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

@tf.function
def tfGetPeriodicDifference(sAngle, eAngle):
	"""
	Tensorflow functon for computing difference between two angles over a periodic boundary at 360 degrees

	Parameters
	----------
	sAngle : ndArray(shape=[None], dtype=float64)
		Array of starting angles

	eAngle : ndArray(shape=[None], dtype=float64)
		Array of ending angles

	Returns
	-------
	ndArray(shape=[None], dtype=float64)

	"""

	degDist = eAngle - sAngle
	degDist = tf.math.floormod((degDist + 180), 360) - 180

	return degDist

def tfLog10(x):
	"""
	Tensorflow function for computing the log base 10 of a number

	Parameters
	----------
	x : ndArray(shape=[None], dtype=float64)

	Returns
	-------
	ndArray(shape=[None], dtype=float64)

	"""

	num = tf.math.log(x)
	denom = tf.math.log(tf.constant(10,dtype=num.dtype))
	return num / denom

@tf.function
def klMultimodalLossTf(xPmf, yPmf, cs, ks, sumBaseDists, distBounds, periodic, temp, nUmbrellas, unsampledCutoff, unsampledWeight, mmWeight):
	"""
	Given a PMF and a set of parameters, computes the total loss using tensorflow operations

	Parameters
	----------
	xPmf : ndArray(shape=[None], dtype=float)
		Array of the PMF reaction coordinate positions

	yPmf : ndArray(shape=[None], dtype=float)
		Array of the PMF reaction coordinates of interests

	cs : ndArray(shape=[None], dtype=float)
		Array of umbrella center positions

	ks : ndArray(shape=[None], dtype=float)
		Array of umbrella force constants

	sumBaseDists : ndArray(shape=[None], dtype=float)
		Array of the sum of pre-existing distributions to be added to estimated distributions

	distBounds : (float, float)
		Tuple of floats describing the desired range of xPmf values

	periodic : bool
		Boolean indicating whether the system is periodic

	temp : float
		Temperature of the system

	nUmbrellas : int
		Number of umbrellas being placed

	unsampledCutoff : float
		Unsampled Cutoff hyperparameter

	unsampledWeight : float
		Hyperparameter for the unsampled penalty

	mmWeight : float
		Hyperparameter for the multimodal penalty

	Returns
	-------
	(float, float, float, float)
		Tuple of the full loss, the distributional loss, the unsampled penalty, and the multimodal penalty

	"""

	absKs = tf.math.pow(tf.constant(10.0,dtype=tf.float64),ks)

	estimatedDists, estExp = estimateFullSampleDist(xPmf, yPmf, cs, absKs, temp=temp)

	mmPen = getMultimodalLossTf(xPmf,estimatedDists,estExp)

	multimodalPen = tf.reshape(mmPen,tf.shape(cs))

	multimodalPen = tf.reduce_mean(multimodalPen,axis=-1)

	# If periodic, create periodic images
	if periodic:
		xSize = tf.shape(xPmf)[0]
		xSize = tf.math.floordiv(xSize, 3)

		estimatedDists = estimatedDists[:,:,:xSize] + estimatedDists[:,:,xSize:2*xSize] + estimatedDists[:,:,2*xSize:3*xSize]
		sumBaseDists = sumBaseDists[:xSize] + sumBaseDists[xSize:2*xSize] + sumBaseDists[2*xSize:3*xSize]
		xPmf = xPmf[xSize:2*xSize]

	inds = tf.transpose(tf.where(tf.logical_and(xPmf >= distBounds[0], xPmf <= distBounds[1])))[0]

	if periodic:
		periodicInds = tf.transpose(tf.where(tf.logical_and(xPmf >= distBounds[0] % 360, xPmf < 360)))[0]

		if distBounds[0] < 0:
			inds = tf.concat([periodicInds,inds],axis=0)

	xPmf = tf.gather(xPmf,inds)
	sumBaseDists = tf.gather(sumBaseDists,inds)
	estimatedDists = tf.gather(estimatedDists,inds,axis=-1)

	klPen = piecewiseKLDiv(xPmf,estDists=estimatedDists,sumBaseDists=sumBaseDists,pieces=2*nUmbrellas)

	unsampledPen = getUnsampledPenalty(estDists=estimatedDists,sumBaseDists=sumBaseDists,unsampledCutoff=unsampledCutoff)

	totLoss = klPen + unsampledWeight * unsampledPen + mmWeight * multimodalPen

	return totLoss, klPen, unsampledWeight * unsampledPen, mmWeight * multimodalPen

@tf.function
def getMultimodalLossTf(xPmf, estimatedDists, normEstExps):
	"""
	This function computes the KL divergence between the obtained distribution and the normal distribution with the same mean and standard deviation.

	Parameters
	----------
	xPmf : ndArray(shape=[None], dtype=float64)
		Array of the PMF reaction coordinate positions

	estimatedDists : ndArray(shape=[None], dtype=float64)
		Array of the probability density of the resulting distribution

	normEstExps : ndArray(shape=[None], dtype=float64)
		Array of the normalized logarithm of the resulting distribution

	Returns
	-------
	float

	"""

	yMean, yVar = tf.nn.weighted_moments(xPmf,[-1],estimatedDists,keepdims=True)

	yStd = tf.math.sqrt(yVar)

	normDists = tfp.distributions.Normal(loc=yMean, scale=yStd)

	logProbs = normDists.log_prob(xPmf)

	normLogProbs = logProbs - tf.math.log(tf.reduce_sum(tf.exp(logProbs),axis=-1,keepdims=True))

	normProbs = tf.exp(normLogProbs)

	penalty = tf.reduce_mean(normProbs * (normLogProbs - normEstExps) + estimatedDists * (normEstExps - normLogProbs), axis=-1)

	return penalty

@tf.function
def piecewiseKLDiv(xPmf,estDists,sumBaseDists,pieces):
	"""
	This function computes the piecewise KL divergence of the total resulting distribution.

	Parameters
	----------
	xPmf : ndArray(shape=[None], dtype=float64)
		Array of the PMF reaction coordinate positions

	estDists : ndArray(shape=[None], dtype=float64)
		Array of the probability density of the resulting distribution

	sumBaseDists : ndArray(shape=[None], dtype=float64)
		Array of the summed distribution of umbrellas whose parameters are fixed

	pieces : int
		How many pieces to break the distribution into

	Returns
	-------
	float

	"""

	delta = tf.shape(xPmf)[0] / pieces

	slices = tf.range(0,tf.math.ceil(delta),1,dtype=tf.int32)

	slices = tf.repeat(slices[tf.newaxis,:],2*pieces - 1,axis=0)

	offset = tf.cast(tf.floor(tf.range(0,2*pieces - 1,1,dtype=tf.float64) * (delta / 2)),tf.int32)
	slices = slices + offset[:,np.newaxis]

	totDist = sumBaseDists + tf.reduce_sum(estDists, axis=1)

	td = tf.gather(totDist,slices,axis=-1)

	kld = klDivTfFlat(f1=td)

	return kld

@tf.function
def klDivTfFlat(f1):
	"""
	Computes the KL divergence between the provided distribution and the uniform distribution.

	Parameters
	----------
	f1 : ndArray(shape=[None], dtype=float64)
		Array of the probability density

	Returns
	-------
	float

	"""

	f1Prob = f1 / (tf.reduce_sum(f1,keepdims=True,axis=-1) + 1e-8) + 1e-8

	f2Prob = tf.ones_like(f1) / tf.cast(tf.shape(f1)[2],dtype=tf.float64)

	klDivs = tf.reduce_sum(f1Prob * tf.math.log(f1Prob/f2Prob) + f2Prob * tf.math.log(f2Prob/f1Prob),axis=-1)

	return tf.reduce_mean(klDivs,axis=-1)

@tf.function
def getUnsampledPenalty(estDists,sumBaseDists, unsampledCutoff):
	"""
	This function computes unsampled penalty of the resulting distribution

	Parameters
	----------
	estDists : ndArray(shape=[None], dtype=float64)
		Array of the probability density of the resulting distribution

	sumBaseDists : ndArray(shape=[None], dtype=float64)
		Array of the summed distribution of umbrellas whose parameters are fixed

	unsampledCutoff : float
		Unsampled Cutoff hyperparameter

	Returns
	-------
	float

	"""

	dist = sumBaseDists + tf.reduce_sum(estDists, axis=1)

	penalty = tf.square((unsampledCutoff - tf.minimum(dist,unsampledCutoff)) / unsampledCutoff)

	return tf.reduce_mean(penalty,axis=-1)

@tf.function
def estimateFullSampleDist(xPmf,yPmf,cs,ks,temp):
	"""
	Given the underlying PMF and the umbrella parameters, this function computes the expected resulting distributions

	Parameters
	----------
	xPmf : ndArray(shape=[None], dtype=float)
		Array of the PMF reaction coordinate positions

	yPmf : ndArray(shape=[None], dtype=float)
		Array of the PMF reaction coordinates of interests

	cs : ndArray(shape=[None], dtype=float)
		Array of umbrella center positions

	ks : ndArray(shape=[None], dtype=float)
		Array of umbrella force constants

	temp : float
		Temperature of the system

	Returns
	-------
	(ndArray(shape=[None], dtype=float64), ndArray(shape=[None], dtype=float64))
		Tuple of the array of the probability density of the resulting distribution, and the log of that probability density
	"""
	dist = cs[:,:,tf.newaxis] - xPmf[tf.newaxis,tf.newaxis,:]

	wellE = (ks[:,:,tf.newaxis] / 2) * tf.square(dist)

	pot = (yPmf + wellE) * 1000  # cal/mol

	pot = pot / NA  # cal
	beta = 1 / (KB * temp)

	exp = -beta * pot

	prob = tf.exp(exp)

	probSum = tf.reduce_sum(prob,keepdims=True,axis=-1)

	exp = exp - tf.math.log(probSum)

	prob = prob / probSum

	return prob, exp

@tf.function
def bbOptFnTf(params, xPmf, yPmf, sumBaseDists, distBounds, temp, periodic, unsampledCutoff, unsampledWeight, mmWeight):
	"""
	This function interfaces between the non-tensorflow operation functions and the tensorflow functions. It is called to evaluate the loss function on a set of parameters

	Parameters
	----------
	params : ndArray(shape=[None], dtype=float)
		Array of the umbrella parameters

	xPmf : ndArray(shape=[None], dtype=float)
		Array of the PMF reaction coordinate positions

	yPmf : ndArray(shape=[None], dtype=float)
		Array of the PMF reaction coordinates of interests

	sumBaseDists : ndArray(shape=[None], dtype=float)
		Array of the sum of pre-existing distributions to be added to estimated distributions

	distBounds : (float, float)
		Tuple of floats describing the desired range of xPmf values

	temp : float
		Temperature of the system

	periodic : bool
		Boolean indicating whether the system is periodic

	unsampledCutoff : float
		Unsampled Cutoff hyperparameter

	unsampledWeight : float
		Hyperparameter for the unsampled penalty

	mmWeight : float
		Hyperparameter for the multimodal penalty

	Returns
	-------
	(float, float, float, float)
		Tuple of the full loss, the distributional loss, the unsampled penalty, and the multimodal penalty

	"""

	nUmbrellas = tf.shape(params)[1] // 2
	centers = params[:,:nUmbrellas]
	ks = params[:,nUmbrellas:]

	return klMultimodalLossTf(xPmf, yPmf, centers, ks, sumBaseDists, distBounds, temp=temp, periodic=periodic, nUmbrellas=nUmbrellas, unsampledCutoff=unsampledCutoff, unsampledWeight=unsampledWeight, mmWeight=mmWeight)

@tf.function
def tfGrads(x0, xPmf, yPmf, sumBaseDists, distBounds, temp, periodic, unsampledCutoff, unsampledWeight, mmWeight):
	"""
	This function is used to compute the loss and derivative of the loss with respect to the umbrella parameters

	Parameters
	----------
	x0 : ndArray(shape=[None], dtype=float)
		Array of the umbrella parameters

	xPmf : ndArray(shape=[None], dtype=float)
		Array of the PMF reaction coordinate positions

	yPmf : ndArray(shape=[None], dtype=float)
		Array of the PMF reaction coordinates of interests

	sumBaseDists : ndArray(shape=[None], dtype=float)
		Array of the sum of pre-existing distributions to be added to estimated distributions

	distBounds : (float, float)
		Tuple of floats describing the desired range of xPmf values

	temp : float
		Temperature of the system

	periodic : bool
		Boolean indicating whether the system is periodic

	unsampledCutoff : float
		Unsampled Cutoff hyperparameter

	unsampledWeight : float
		Hyperparameter for the unsampled penalty

	mmWeight : float
		Hyperparameter for the multimodal penalty

	Returns
	-------
	(float, ndArray(shape=[None], dtype=float))
		Tuple of the full loss and the derivative of the loss with respect to x0

	"""

	with tf.GradientTape() as tape:
		tape.watch(x0)
		loss = bbOptFnTf(x0,xPmf, yPmf, sumBaseDists, distBounds, temp, periodic, unsampledCutoff, unsampledWeight, mmWeight)[0]

	grads = tape.gradient(loss,x0)

	return loss, grads