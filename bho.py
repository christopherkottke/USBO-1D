import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from numba import njit
import os
import pickle
import cvxpy as cp
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow as gpf
from gpflow.utilities import print_summary, set_trainable

from utilities import *
from tf_utils import *

@tf.function
def expectedImprovement(gpRes,fMin):
	"""
	Tensorflow implementation of the expected improvement acquisition function for Bayesian Hyperparameter Optimization

	Parameters
	----------
	gpRes : tuple (tf.Tensor(shape=[None,1], dtype=float64), tf.Tensor(shape=[None,1], dtype=float64))
		Tuple containing the Gaussian Process means and standard deviations
	fMin : float
		The current optima for Bayesian Hyperparameter Optimization

	Returns
	-------
	tf.Tensor(shape=[None,1], dtype=float64))
		Returns the expected improvement for each input Gaussian Process prediction

	"""

	dist = tfp.distributions.Normal(loc=tf.constant(0.0,dtype=tf.float64),scale=tf.constant(1.0,dtype=tf.float64))

	mu, sigma = gpRes

	# Minimizing instead of maximizing
	mu = -mu
	fMax = -fMin

	zeta = tf.constant(0.01,dtype=tf.float64)

	z = (mu - fMax - zeta) / sigma

	ei = (mu - fMax - zeta) * dist.cdf(z) + sigma * dist.prob(z)

	return ei

class BHO_GM:
	"""
	A class for performing Bayesian Hyperparameter Optimization

	Attributes
	__________
	kernel : gpFlow Kernel
		GPFlow Kernel for Gaussian Process regression

	gpXVar : tf.Variable(shape=[None,dataDim], dtype=float64)
		Tf Variable for holding the Gaussian Process Regression feature vectors

	gpYVar : tf.Variable(shape=[None,dataDim], dtype=float64)
		Tf Variable for holding the Gaussian Process Regression label vector

	gp : gpFlow Gaussian Process Regression
		GPFlow Class for performing Gaussian Process Regression

	gpLoss : gpFlow Gaussian Process Regression Optimization Loss
		GPFlow object holding the loss for optimization of the Gaussian Process

	yMean : tf.Variable(shape=[], dtype=tf.float64)
		Tf Variable for holding the mean of the labels

	yStd : tf.Variable(shape=[], dtype=tf.float64)
		Tf Variable for holding the standard deviation of the labels

	fMin : tf.Variable(shape=[], dtype=tf.float64)
		Tf Variable for holding the optimal value found during Bayesian Hyperparameter Optimization

	optSize : int
		Number of points to be randomly sampled as initializations for acquisition function optimization for each batch entry

	distData : [np.array]
		Array containing the training data for Gaussian Process Regression

	distLabels : [float]
		Array containing the log of the training labels for Gaussian Process Regression

	totLoss = : [float]
		Array containing the log of the observed total losses

	dataDim : int
		Dimensionality of the inputs for the Gaussian Process Regression (twice the number of umbrellas)

	cBounds : (float, float)
		Tuple defining the lower and upper bounds for the placement of the center of each umbrella

	kBounds : (float, float)
		Tuple defining the lower and upper bounds (log10) for the spring constant of each umbrella

	periodic : bool
		Boolean indicating whether the PMF is periodic at 360 degrees

	temp : float
		Temperature at which simulations will be carried out

	savedPoints : int
		Counter keeping track of how many points have been removed from the input data for being the same as another
			point in the input data

	gpNRestarts : int
		Number of times the Gaussian Process should be optimized to obtain the best fit

	"""

	def __init__(self, dataDim, cBounds, kBounds, periodic, temp, mmWeight, optSize=10, gpNRestarts=10):
		"""
		Init function for the BHO_GM class

		Parameters
		----------
		dataDim : int
			Dimensionality of the inputs for the Gaussian Process Regression (twice the number of umbrellas)

		cBounds : (float, float)
			Tuple defining the lower and upper bounds for the placement of the center of each umbrella

		kBounds : (float, float)
			Tuple defining the lower and upper bounds (log10) for the spring constant of each umbrella

		periodic : bool
			Boolean indicating whether the PMF is periodic at 360 degrees

		temp : float
			Temperature at which simulations will be carried out

		mmWeight : float
			Weight for the multimodal loss penalty

		optSize : int
			Number of points to be randomly sampled as initializations for acquisition optimization (default 10)

		gpNRestarts : int
			Number of times the Gaussian Process should be optimized to obtain the best fit
		"""

		# Variables defined to be included in TF Graph computations
		# ---------- TF Variables ---------- #
		self.kernel = gpf.kernels.Matern32(lengthscales=0.1)

		self.gpXVar = tf.Variable(np.zeros((1,dataDim)),shape=(None,dataDim),dtype=tf.float64,trainable=False)
		self.gpYVar = tf.Variable(np.zeros((1,1)),shape=(None,1),dtype=tf.float64,trainable=False)

		self.gp = gpf.models.GPR(data=(self.gpXVar,self.gpYVar),kernel=self.kernel)
		self.gp.likelihood.variance.assign(1e-3)
		self.gp.kernel.variance.assign(1.0)
		set_trainable(self.gp.likelihood.variance, False)
		set_trainable(self.gp.kernel.variance, False)
		self.gpLoss = self.gp.training_loss_closure()

		self.yMean = tf.Variable(0.0,dtype=tf.float64)
		self.yStd = tf.Variable(1.0,dtype=tf.float64)

		# Initialized with unreasonably large value for subsequent minimization
		self.fMin = tf.Variable(1e10,dtype=tf.float64)
		# ---------- END TF Variables ---------- #

		self.optSize = optSize

		self.distData = []
		self.distLabels = []
		self.totLoss = []

		self.dataDim = dataDim
		self.cBounds = cBounds
		self.kBounds = kBounds
		self.periodic = periodic
		self.temp = temp
		self.savedPoints = 0

		self.mmWeight = mmWeight

		self.gpNRestarts = gpNRestarts

	def getGPPred(self,x,yMean,yStd):
		"""
		TF function for returning the Gaussian Process estimate for a set of input points x

		Parameters
		----------
		x : ndArray(shape=[None, dataDim], dtype=float64)
			Set of input points for prediction

		yMean : tf.Variable(shape=[], dtype=float64)
			Mean value of the data labels

		yStd : tf.Variable(shape=[], dtype=float64)
			Standard deviation of the data labels

		Returns
		-------
		(ndArray(shape=[None, dataDim], dtype=float64), ndArray(shape=[None, dataDim], dtype=float64))
			Tuple containing the un-normalized mean and standard deviation of the Gaussian Process Regression at each
				input point.

		"""

		distPred, distStd = self.gp.predict_f(x)

		distPred = distPred * yStd + yMean
		distStd = distStd * yStd

		return distPred, distStd

	def addData(self,data,distLabel,totLoss):
		"""
		Function for adding raw data to the BHO_GM data arrays.

		Sorts umbrellas by increasing center values and scales and shifts input data to be contained within the unit
			cube for Gaussian Process prediction. Removes points in data that are already in distData

		Parameters
		----------
		data : [ndArray]
			Array of raw input points

		distLabel : [float]
			Array of the sum of the distributional and unsampled losses

		totLoss : [float]
			Array of the total losses

		Returns
		-------
		None

		"""

		for d,l,t in zip(data,distLabel,totLoss):
			dataCube, sortInds = self.dataToCube(d)

			lgDistLabel = np.log(l)

			lgTotLoss = np.log(t)

			isOpt = False
			if len(self.totLoss) == 0 or lgTotLoss < np.min(self.totLoss):
				isOpt = True

			novel = True
			if len(self.totLoss) > 0:
				norms = np.linalg.norm(self.distData - dataCube,axis=1)

				for norm, y in zip(norms,self.totLoss):
					if norm < 1e-4 and np.abs(y - lgTotLoss) < 1e-4:
						novel = False
						break

			if isOpt or novel:
				self.distData.append(dataCube)
				self.distLabels.append(lgDistLabel)
				self.totLoss.append(lgTotLoss)

			else:
				self.savedPoints += 1

	def proposePoints(self, xPmf, yPmf, nPoints, verbose=False):
		"""
		Uses the current data to train and query a Gaussian Process model to propose candidate locations for additional sampling

		Parameters
		----------
		nPoints : int
			The number of proposed points desired

		verbose : bool
			Boolean indicating whether information about the Gaussian process regression should be displayed (default: True)

		Returns
		-------
		(ndArray(shape=[nPoints, dataDim], dtype=float64), float)
			First argument is an array of proposed points for future sampling
			Second argument is the largest acquisition score for the proposed points

		"""

		# ---------- Preparing data for GPR ---------- #
		x = np.array(self.distData)
		y = np.array(self.distLabels)[:, np.newaxis]
		self.fMin.assign(np.min(self.totLoss))

		yMean = np.mean(y)
		yStd = np.std(y)

		self.yMean.assign(yMean)
		self.yStd.assign(yStd)

		y = (y - yMean) / yStd

		self.gpXVar.assign(x)
		self.gpYVar.assign(y)

		# ---------- Optimizing the GP parameters ---------- #
		self.gp.kernel.lengthscales.assign(0.1)

		opt = gpf.optimizers.Scipy()
		logs = opt.minimize(self.gpLoss, self.gp.trainable_variables)

		bestLL = logs.fun
		bestParams = [var.numpy() for var in self.gp.trainable_variables]

		for i in range(self.gpNRestarts):
			if i == 0:
				self.gp.kernel.lengthscales.assign(0.1)
			else:
				self.gp.kernel.lengthscales.assign(np.random.uniform() * np.sqrt(self.dataDim))

			opt = gpf.optimizers.Scipy()
			logs = opt.minimize(self.gpLoss, self.gp.trainable_variables)

			if logs.fun < bestLL:
				bestLL = logs.fun
				bestParams = [var.numpy() for var in self.gp.trainable_variables]

		[tv.assign(var) for tv,var in zip(self.gp.trainable_variables,bestParams)]

		# ---------- Maximizing the Acquisition Function for optSize random points ---------- #
		samplePoints = np.random.random((self.optSize * nPoints, self.dataDim))

		samplePoints, _ = self.orderPoints(samplePoints)

		optAccs = []
		optPoints = []
		optPreds = []

		for point in samplePoints:

			# Add the mean predicted value of prior proposed points to the training data go obtain a next best proposal
			x = np.array(self.distData + optPoints)

			y = np.array(self.distLabels + optPreds)[:, np.newaxis]
			self.fMin.assign(np.min(self.totLoss))

			yMean = np.mean(y)
			yStd = np.std(y)

			self.yMean.assign(yMean)
			self.yStd.assign(yStd)

			y = (y - yMean) / yStd

			self.gpXVar.assign(x)
			self.gpYVar.assign(y)

			minRes = minimize(lambda x: self.gpOptWithGrads(x,xPmf,yPmf,self.fMin,self.yMean,self.yStd),x0=point,bounds=[(0,1) for _ in range(self.dataDim)],jac=True)

			optAccs.append(np.exp(-minRes.fun))

			optPoints.append(self.orderPoints(minRes.x[np.newaxis,:])[0][0])

			optPreds.append(self.getGPPred(self.orderPoints(minRes.x[np.newaxis,:])[0],self.yMean,self.yStd)[0].numpy()[0][0])

		optPoints = np.array(optPoints)
		optAccs = np.array(optAccs)

		indices = np.argpartition(-optAccs,nPoints)[:nPoints]

		params = optPoints[indices]
		accs = optAccs[indices]

		if verbose:
			print("-----")
			print("Fit on %d Points"%len(self.totLoss))
			print("Saved Points: %d"%(self.savedPoints))
			print_summary(self.gp)
			print("Current Minima: %f"%np.exp(np.min(self.totLoss)))
			print("Chosen with Optimal Acquisition Score: %E" % np.max(accs))
			print("-----")
			print()

		params = self.cubeToData(params)

		return params, np.max(accs)

	def getLogNormMeanVar(self,lnMean,lnVar):
		"""
		Second order moment matching for approximating parameters a of log-Normal distribution as a Normal distribution

		Parameters
		----------
		lnMean : ndArray(shape=[None], dtype=float64)
			An array of mean parameters of the log-Normal distributions

		lnVar : ndArray(shape=[None], dtype=float64)
			An array of variances of the log-Normal distributions

		Returns
		-------
		(ndArray(shape=[None], dtype=float64), ndArray(shape=[None], dtype=float64))
			The first argument is the resulting mean parameter of the Normal distribution which approximates the log-Normal distribution
			The second argument is the resulting variance parameter of the Normal distribution which approximates the log-Normal distribution

		"""

		mean = tf.exp(lnMean + (lnVar / tf.constant(2.0,dtype=tf.float64)))

		t1 = tf.exp(lnVar) - tf.constant(1.0, tf.float64)
		t2 = tf.exp(lnMean * tf.constant(2.0, tf.float64) + lnVar)

		var = t1*t2

		return mean, var

	@tf.function
	def getGPGMPred(self,x,xPmf,yPmf,yMean,yStd):
		"""
		TF Function for computing the full model output (Through the entire graphical model)

		Parameters
		----------
		x : ndArray(shape=[None, dataDim], dtype=float64)
			Locations to evaluate the GP-GM output

		xPmf : ndArray(shape=[None], dtype=float64)
			x Positions of PMF

		yPmf : ndArray(shape=[None], dtype=float64)
			PMF values

		yMean : tf.Variable(shape=[], dtype=float64)
			Mean value of the data labels

		yStd : tf.Variable(shape=[], dtype=float64)
			Standard deviation of the data labels

		Returns
		-------
		(ndArray(shape=[None], dtype=float64), ndArray(shape=[None], dtype=float64))
			The first argument is the resulting mean parameter of the predicted Normal distribution
			The second argument is the resulting standard deviation of the predicted Normal distribution

		"""

		# ---------- Get GP prediction for distribution + unsampled loss ---------- #
		distPred, distStd = self.getGPPred(x,yMean,yStd)

		meanSum, varSum = self.getLogNormMeanVar(distPred,tf.square(distStd))

		# ---------- Get multimodal loss interpolation ---------- #
		cs = x[:, :self.dataDim // 2] * (self.cBounds[1] - self.cBounds[0]) + self.cBounds[0]
		ks = tf.math.pow(tf.constant(10.0,dtype=tf.float64),x[:, self.dataDim // 2:] * (self.kBounds[1] - self.kBounds[0]) + self.kBounds[0])

		estDists, estExp = estimateFullSampleDist(xPmf,yPmf,cs,ks,temp=self.temp)

		mmPen = getMultimodalLossTf(xPmf,estDists,estExp)

		mmPen = tf.reduce_sum(mmPen,axis=-1,keepdims=True)

		mmInter = self.mmWeight * mmPen

		mmPred = tf.math.log(mmInter + 1e-20)

		mmMean, mmVar = self.getLogNormMeanVar(mmPred, tf.constant(0.0,dtype=tf.float64))

		meanSum += mmMean

		totVar = tf.math.log(varSum/tf.square(meanSum) + 1.0)

		totMean = tf.math.log(meanSum) - (totVar/2.0)

		return totMean, tf.sqrt(totVar)

	def gpOptWithGrads(self,x,xPmf,yPmf,yOpt,yMean,yStd):
		"""
		Calls getGPGrads as a TF operation and returns the result as a numpy array for the scipy optimization

		Parameters
		----------
		x : ndArray(shape=[None, dataDim], dtype=float64)
			Locations to evaluate the GP-GM output

		xPmf : ndArray(shape=[None], dtype=float64)
			x Positions of PMF

		yPmf : ndArray(shape=[None], dtype=float64)
			PMF values

		yOpt : float
			The current optima for Bayesian Hyperparameter Optimization

		yMean : tf.Variable(shape=[], dtype=float64)
			Mean value of the data labels

		yStd : tf.Variable(shape=[], dtype=float64)
			Standard deviation of the data labels

		Returns
		-------
		(ndArray(shape=[None], dtype=float64), ndArray(shape=[None, dataDim], dtype=float64))
			The first argument is the value of the aquisition function at each point
			The second argument is the gradient of the aquisition function with respect to each entry in the input x

		"""

		xTf = tf.convert_to_tensor(x)[tf.newaxis,:]

		xVal, grads = self.getGPGrads(xTf,xPmf,yPmf,yOpt,yMean,yStd)

		return xVal.numpy(), grads.numpy()

	@tf.function
	def getGPGrads(self,x,xPmf,yPmf,yOpt,yMean,yStd):
		"""
		TF function for computing the value and gradient of the acquisition function for a given set of inputs

		Parameters
		----------
		x : ndArray(shape=[None, dataDim], dtype=float64)
			Locations to evaluate the GP-GM output

		xPmf : ndArray(shape=[None], dtype=float64)
			x Positions of PMF

		yPmf : ndArray(shape=[None], dtype=float64)
			PMF values

		yOpt : float
			The current optima for Bayesian Hyperparameter Optimization

		yMean : tf.Variable(shape=[], dtype=float64)
			Mean value of the data labels

		yStd : tf.Variable(shape=[], dtype=float64)
			Standard deviation of the data labels

		Returns
		-------
		(ndArray(shape=[None], dtype=float64), ndArray(shape=[None, dataDim], dtype=float64))
			The first argument is the value of the aquisition function at each point
			The second argument is the gradient of the aquisition function with respect to each entry in the input x

		"""

		with tf.GradientTape() as tape:
			tape.watch(x)

			cs = x[:, :self.dataDim // 2]
			ks = x[:, self.dataDim // 2:]

			sortInd = tf.argsort(cs)

			cs = tf.gather(cs,sortInd,batch_dims=1)
			ks = tf.gather(ks,sortInd,batch_dims=1)

			x = tf.concat([cs, ks], axis=-1)

			gpPred = self.getGPGMPred(x, xPmf, yPmf, yMean, yStd)

			acc = -tf.math.log(expectedImprovement(gpPred, yOpt) + 1e-80)

			accSum = tf.reduce_sum(acc)

		grads = tape.gradient(accSum, x)

		return acc, grads

	def cubeToData(self,x):
		"""
		Takes an input x as coordinates in a unit cube and converts the coordinates to those of the optimization variables (e.g. angstroms, and log kcal/mol/A^2)

		Parameters
		----------
		x : ndArray(shape=[None, dataDim], dtype=float64)
			Coordinates in the unit cube

		Returns
		-------
		ndArray(shape=[None, dataDim], dtype=float64)

		"""

		x = np.copy(x)
		x[:,:self.dataDim // 2] = x[:,:self.dataDim // 2] * (self.cBounds[1] - self.cBounds[0]) + self.cBounds[0]
		x[:,self.dataDim // 2:] = x[:,self.dataDim  // 2:] * (self.kBounds[1] - self.kBounds[0]) + self.kBounds[0]

		return x

	def dataToCube(self,x):
		"""
		Takes an input x in units of the optimization variables (e.g. angstroms, and log kcal/mol/A^2) and converts them to coordinates in the unit cube

		Parameters
		----------
		x : ndArray(shape=[None, dataDim], dtype=float64)
			Parameters in terms of the units of the natural variables

		Returns
		-------
		ndArray(shape=[None, dataDim], dtype=float64)

		"""

		x = np.copy(x)
		x[:len(x) // 2] = (x[:len(x) // 2] - self.cBounds[0]) / (self.cBounds[1] - self.cBounds[0])
		x[len(x) // 2:] = (x[len(x) // 2:] - self.kBounds[0]) / (self.kBounds[1] - self.kBounds[0])

		x, sortInds = self.orderPoints(x[np.newaxis,:])

		x = x[0]

		return x, sortInds

	def orderPoints(self,x):
		"""
		Function to enforce the degeneracy of the input space (that the order of listing the umbrellas doesn't matter)
		This is done by ordering each umbrella by the position of the umbrella center in non-decreasing order

		Parameters
		----------
		x : ndArray(shape=[None, dataDim], dtype=float64)
			Umbrella parameters either as natural variables or in the unit cube

		Returns
		-------
		(ndArray(shape=[None, dataDim], dtype=float64), ndArray(shape=[dataDim]))

		"""

		cVals = x[:,:self.dataDim//2]
		kVals = x[:,self.dataDim//2:]

		sortInds = np.argsort(cVals)

		cVals = np.take_along_axis(cVals,sortInds,axis=-1)
		kVals = np.take_along_axis(kVals,sortInds,axis=-1)

		return np.concatenate([cVals, kVals],axis=-1), sortInds
