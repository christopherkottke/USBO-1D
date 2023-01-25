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
from mcmc import *
from bho import *

tf.keras.backend.set_floatx('float64')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

plt.rcParams.update({'font.size': 10})

MM_WEIGHT=None
UNSAMPLED_WEIGHT=None
UNSAMPLED_CUTOFF=None

'''
TODO
	Post Paper
		Code organization and documentation
		Arbitrary periodicity?
		Optimize simulation time (assuming necessary sim time scales 1/k)
		
	Variable diffusion rate along RC (model diffusion constant for each bin; weight positions by frequency of each bin in each simulations)
	2D PMF support
'''

def addPointsToBHO(pointsArr, bho, xPmf, yPmf, sumBaseDists, distBounds, temp, periodic):
	"""
	This function takes proposed umbrella parameters, evaluates the fitness of those parameters, and adds the data to a provided BHO class

	Parameters
	----------
	pointsArr : ndArray(shape=[None, None], dtype=float64)
		Array of umbrella parameterizations

	bho : BHO_GM class instance
		An instance of the BHO_GM class

	xPmf : ndArray(shape=[None], dtype=float)
		Array of the PMF reaction coordinate positions

	yPmf : ndArray(shape=[None], dtype=float)
		Array of the PMF free energies

	sumBaseDists : ndArray(shape=[None], dtype=float)
		Array of the sum of pre-existing distributions to be added to estimated distributions

	distBounds : (float, float)
		Lower and upper bound of the PMF coordinates for which the distributional loss should be evaluated

	temp : float
		Temperature of the system

	periodic : bool
		Indicates whether the system of interest is periodic

	Returns
	-------
	None

	"""

	myPoints = np.concatenate(pointsArr,axis=0)

	tfPoints = tf.convert_to_tensor(myPoints,dtype=tf.float64)

	val, klPen, unsampledPen, multimodalPen = bbOptFnTf(tfPoints, xPmf, yPmf, sumBaseDists, distBounds, temp=temp, periodic=periodic, unsampledCutoff=UNSAMPLED_CUTOFF, unsampledWeight=UNSAMPLED_WEIGHT, mmWeight=MM_WEIGHT)

	bho.addData(myPoints,klPen+unsampledPen,val)

def runBBOpt(paramDict):
	"""
	This function takes as input parameters for running the USBO method and runs the method using those parameters

	Parameters
	----------
	paramDict : dict
		A dictionary of parameters for running the USBO algorithm

	Returns
	-------
	None

	"""

	# ---------- Unpacking paramDict ---------- #

	base = paramDict["base_path"]
	pmfFolder = paramDict["pmf_folder"]
	pmfPath = paramDict["pmf_path"]
	metaPath = paramDict["meta_path"]
	simDataPath = paramDict["sim_data_path"]
	outputName = paramDict["output_name"]

	autoCLB = float(paramDict["clb"])
	autoCUB = float(paramDict["cub"])
	distCLB = float(paramDict["dist_clb"])
	distCUB = float(paramDict["dist_cub"])

	baseKLB = float(paramDict["base_klb"])
	baseKUB = float(paramDict["base_kub"])

	nUmbrellas = int(paramDict["n_umbrellas"])

	# Simulation Parameters
	simIntTime = float(paramDict["sim_int_time"])
	periodic = int(paramDict["periodic"]) == 1
	temp = float(paramDict["temp"])
	timeBetweenPoints = float(paramDict["time_between_points"])

	# Loss Parameters
	mmWeight = float(paramDict["mm_weight"])
	unsampledWeight = float(paramDict["unsampled_weight"])
	unsampledCutoff = float(paramDict["unsampled_cutoff"])

	global UNSAMPLED_CUTOFF
	UNSAMPLED_CUTOFF = unsampledCutoff
	global UNSAMPLED_WEIGHT
	UNSAMPLED_WEIGHT = unsampledWeight
	global MM_WEIGHT
	MM_WEIGHT = mmWeight

	# BHO Parameters
	bhoOptSize = int(paramDict["bho_opt_size"])
	bhoGPNRestarts = int(paramDict["bho_gp_n_restarts"])

	# MCMC Simulation Parameters
	nReplicas = int(paramDict["n_replicas"])
	minStep = float(paramDict["min_step"])
	simConvTol = float(paramDict["sim_conv_tol"])
	simConvFactor = int(paramDict["sim_conv_factor"])
	intTime = float(paramDict["int_time"])
	saveSteps = int(paramDict["save_steps"])
	saveApprox = int(paramDict["save_approx"]) == 1

	# Optimizing US Parameters
	nXPoints = int(paramDict["n_x_points"])
	umbNIters = int(paramDict["umb_search_iters"])
	umbBSize = int(paramDict["umb_search_bSize"])
	usePriorDists = int(paramDict["use_prior_dists"]) == 1
	lastParamFile = paramDict["last_param_file"]

	# ---------- Loading/Computing Simulation Information ---------- #
	meta = readMeta("%s/%s/%s" % (base, pmfFolder, metaPath))

	if os.path.exists("%s/%s" % (base, simDataPath)):
		print("Loading Simulation Data")
		with open("%s/%s" % (base, simDataPath),'rb') as f:
			simData = pickle.load(f)
			points = simData["points"]

	else:
		print("Getting Simulation Data from File")
		points = getPoints(base, pmfFolder, meta)

		with open("%s/%s"%(base, simDataPath),'wb') as f:
			simData = {"points":points}

			pickle.dump(simData,f)

	xs, ys = getPMF("%s/%s/%s" % (base, pmfFolder, pmfPath))

	centers = np.array([m["c"] for m in meta])
	ks = np.array([m["k"] for m in meta])

	diffC = computeDiffusionConstant(xs, ys, points, centers, ks, timeBetweenPoints, temp=temp, intTime=simIntTime, periodic=periodic)
	diffC = np.average(diffC)

	# ---------- Setting Up Interpolations ---------- #

	if periodic:
		nPoints = int(nXPoints * (360 / (distCUB - distCLB)))
		xsNew = np.linspace(0, 360, nPoints)

		ys = tfp.math.interp_regular_1d_grid(xsNew, x_ref_min=0, x_ref_max=360, y_ref=ys)

		xsNew = xsNew[:-1]
		ys = ys[:-1]

		xs = np.concatenate([xsNew-360,xsNew,xsNew+360])
		ys = np.concatenate([ys,ys,ys],axis=0)

	else:
		xsNew = np.linspace(distCLB,distCUB,nXPoints)

		ys = tfp.math.interp_regular_1d_grid(xsNew, x_ref_min=xs[0], x_ref_max=xs[-1], y_ref=ys)
		xs = xsNew

	hists = pointsToHist(xs, points)
	hists = np.array(hists)

	priorDists = None
	if usePriorDists:
		priorDists = hists

	# ---------- Setting K Bounds ---------- #
	if lastParamFile is not None:
		lastParams = loadParamsFromFile("%s/%s" % (base,lastParamFile))
		autoKLB = getKLB(xs, ys, baseKLB, baseKUB, lastParams=lastParams, temp=temp)

		_, lastKs = lastParams
		lastKs = np.unique(lastKs)
		ksSorted = np.sort(lastKs)

		if autoKLB < ksSorted[0]:
			autoKLB = ksSorted[1]

		print("New K Lower Bound: %f" % autoKLB)

	else:
		autoKLB = baseKLB

	# ---------- Running USBO Method ---------- #

	if nUmbrellas > 0:
		bestRes, totalLoss, finalKL, finalUP, finalMM = proposeNNewUmbrellas(xPmf=xs,
																						  yPmf=ys,
																						  startDists=priorDists,
																						  xRange=(autoCLB,autoCUB),
																						  kRange=(autoKLB,baseKUB),
																						  distBounds=(distCLB, distCUB),
																						  nUmbrellas=nUmbrellas,
																						  nIters=umbNIters,
																						  bSize=umbBSize,
																						  bhoOptSize=bhoOptSize,
																						  bhoRestarts=bhoGPNRestarts,
																						  verbose=True,
																						  periodic=periodic,
																						  temp=temp)

		bestCs = bestRes[:nUmbrellas]
		bestKs = bestRes[nUmbrellas:]

		plotExpectedHistograms(xs, ys, bestCs, bestKs, temp, periodic, distBounds=(distCLB, distCUB), startDists=priorDists, path="%s/%s.png" % (base, outputName), title="Placement of %d Umbrellas" % nUmbrellas, xLabel="Coordinate Position")

		sortInd = np.argsort(bestCs)
		bestCs = bestCs[sortInd]
		bestKs = bestKs[sortInd]
	else:
		bestCs, bestKs, totalLoss, finalKL, finalUP, finalMM = loadParamsFromFile("%s/%s.txt"%(base,outputName))

	# ---------- Reporting USBO Results ---------- #

	simTimes = [0 for _ in bestCs]
	acTimes = [0 for _ in bestCs]
	klDivs = [0 for _ in bestCs]
	convScores = [0 for _ in bestCs]

	print("%E, %E, %E, %E" % (totalLoss, finalKL, finalUP, finalMM))

	with open("%s/%s.txt" % (base,outputName), 'w') as f:
		for c, k, sTime, acTime in zip(bestCs, bestKs, simTimes, acTimes):
			f.write("%f, %f\n" % (c, k))
		f.write("%f, %f, %E, %E, %f\n" % (totalLoss, finalKL, finalUP, finalMM, np.sum(simTimes) / 1e-15))

	# ---------- Estimating Simulation Time ---------- #

	targetDists, estExp = estimateFullSampleDist(xs,ys,bestCs[np.newaxis,:],bestKs[np.newaxis,:],temp=temp)
	targetDists = targetDists[0]

	for i, (c, k, d) in enumerate(zip(bestCs, bestKs, targetDists)):
		print(i+1)
		simTime, acTime, hists, mcmcPoints, klDiv, convScore = findMaxSimTime(xs, ys, d, c, k, diffC, numReplicas=nReplicas, minStep=minStep, tol=simConvTol, convFactor=simConvFactor, intTime=intTime,saveSteps=saveSteps,saveApprox=saveApprox)
		simTimes[i] = simTime
		acTimes[i] = acTime
		klDivs[i] = klDiv
		convScores[i] = convScore

	# ---------- Reporting USBO Results ---------- #

	with open("%s/%s.txt" % (base,outputName), 'w') as f:
		for c, k, sTime, acTime, klDiv, convScore in zip(bestCs, bestKs, simTimes, acTimes, klDivs, convScores):
			f.write("%f, %f, %d, %f, %E, %E\n" % (c, k, sTime / 1e-15, acTime, klDiv, convScore))
		f.write("%f, %f, %E, %E, %d\n" % (totalLoss, finalKL, finalUP, finalMM, np.sum(simTimes) / 1e-15))

def optWithGrads(x0, xPmf, yPmf, sumBaseDists, distBounds, temp, periodic):
	"""
	This function interfaces the python section of the program with the tensorflow sections to provide the value and gradients of the loss function with respect to the umbrella parameters

	Parameters
	----------
	x0 : ndArray(shape=[None], dtype=float)
		Initial values of umbrella parameters for optimization

	xPmf : ndArray(shape=[None], dtype=float)
		Array of the PMF reaction coordinate positions

	yPmf : ndArray(shape=[None], dtype=float)
		Array of the PMF free energies

	sumBaseDists : ndArray(shape=[None], dtype=float)
		Array of the sum of pre-existing distributions to be added to estimated distributions

	distBounds : (float, float)
		Lower and upper bound of the PMF coordinates for which the distributional loss should be evaluated

	temp : float
		Temperature of the system

	periodic : bool
		Indicates whether the system of interest is periodic

	Returns
	-------
	(float, ndArray(shape=[None], dtype=float))
		Tuple containing the value of the loss fuction at the provided x0 and the gradient of the loss function with respect to the input parameters.

	"""

	x0 = tf.convert_to_tensor(x0,dtype=tf.float64)[tf.newaxis,:]

	loss, grads = tfGrads(x0, xPmf, yPmf, sumBaseDists, distBounds, temp, periodic, UNSAMPLED_CUTOFF, UNSAMPLED_WEIGHT, MM_WEIGHT)

	npGrads = grads.numpy()

	return loss.numpy(), npGrads

def proposeNNewUmbrellas(xPmf,yPmf,xRange,kRange,nUmbrellas,periodic,temp,distBounds=None,startDists=None,nIters=1,bSize=1,bhoOptSize=1,bhoRestarts=1,verbose=False):
	"""
	This function runs the USBO umbrella parameter search for a specified umbrella placement problem.

	Parameters
	----------
	xPmf : ndArray(shape=[None], dtype=float)
		Array of the PMF reaction coordinate positions

	yPmf : ndArray(shape=[None], dtype=float)
		Array of the PMF free energies

	xRange : (float, float)
		Lower and upper bound for the centers of placed umbrellas

	kRange : (float, float)
		Lower and upper bound for the force constant of placed umbrellas

	nUmbrellas : int
		The number of umbrellas to be placed

	periodic : bool
		Indicates whether the system of interest is periodic

	temp : float
		Temperature of the system

	distBounds : (float, float)
		Lower and upper bound of the PMF coordinates for which the distributional loss should be evaluated

	startDists : ndArray(shape=[None, None], dtype=float)
		Arrays of the pre-existing distributions to be added to estimated distributions

	nIters : int
		Number of iterations to perform the outer search loop

	bSize : int
		Number of candidate points to search at once per outer search iteration

	bhoOptSize : int
		Number of points to be sampled for BHO acquisition function maximization

	bhoRestarts : int
		Number of GP models trained to obtain the best model

	verbose : bool
		Whether or not the function should provide information about its run status.

	Returns
	-------
	(ndArray(shape=[None], dtype=float), float, float, float, float)
		Tuple containing the best set of parameters, the value of the loss function at that parameter set, and the values of the component losses.

	"""

	global pointsArr

	sumBaseDists = np.zeros(len(xPmf))

	if startDists is not None:
		sumBaseDists = njitSumAxis0(startDists)

	posBounds = [xRange for _ in range(nUmbrellas)]
	kBounds = [(np.log10(kRange[0]),np.log10(kRange[1])) for _ in range(nUmbrellas)]

	if distBounds is None:
		distBounds = xRange

	bestRes = None
	bestVal = np.infty

	improved = False
	count = 0

	bho = BHO_GM(dataDim=2*nUmbrellas,cBounds=xRange,kBounds=(np.log10(kRange[0]),np.log10(kRange[1])),periodic=periodic, temp=temp, mmWeight=MM_WEIGHT, optSize=bhoOptSize, gpNRestarts=bhoRestarts)

	with tqdm(total=nIters) as pbar:
		print()
		done = False
		while not done:
			bounds = posBounds + kBounds

			# If there are no points sampled, randomly select points
			if len(bho.totLoss) == 0:
				x0 = np.zeros(shape=(bSize,2*nUmbrellas))

				x0[:,:nUmbrellas] = np.random.random(size=(bSize,nUmbrellas)) * (xRange[1] - xRange[0]) + xRange[0]
				x0[:,nUmbrellas:] = np.random.random(size=(bSize,nUmbrellas)) * (np.log10(kRange[1]) - np.log10(kRange[0])) + np.log10(kRange[0])

			# Otherwise select the points via BHO
			else:
				x0, maxAcc = bho.proposePoints(xPmf,yPmf,bSize,verbose=True)

			pointsArr = [x0]

			optVals = []
			optRes = []

			# For each point, optimize over the true loss surface
			for point in x0:
				minRes = minimize(optWithGrads, x0=point, args=(xPmf, yPmf, sumBaseDists, distBounds, temp, periodic), bounds=bounds, method='L-BFGS-B', jac=True)

				optVals.append(minRes.fun)
				pointsArr.append([minRes.x])
				optRes.append(minRes.x)

				count += 1

			# Add the stand and end points to the BHO GP
			addPointsToBHO(pointsArr, bho, xPmf, yPmf, sumBaseDists, distBounds, temp=temp, periodic=periodic)

			bestInd = np.argmin(optVals)
			val = optVals[bestInd]
			res = optRes[bestInd]

			if val < bestVal:
				improved = True
				bestVal = val
				bestRes = res

			if improved:
				if verbose:
					pbar.update(1)
					print()

			if count == nIters * bSize:
				done = True

	bestResTf = tf.convert_to_tensor(bestRes[np.newaxis,:],dtype=tf.float64)

	totalLoss, finalKL, finalUP, finalMM = bbOptFnTf(bestResTf,xPmf,yPmf,sumBaseDists,distBounds=distBounds,temp=temp,periodic=periodic, unsampledCutoff=UNSAMPLED_CUTOFF, unsampledWeight=UNSAMPLED_WEIGHT, mmWeight=MM_WEIGHT)

	bestRes[nUmbrellas:] = np.power(10, bestRes[nUmbrellas:])

	return bestRes, totalLoss, finalKL, finalUP, finalMM