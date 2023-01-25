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

@njit(cache=True)
def njitRunMCMetroSim(xs,ys,c,k,std,states,steps,temp,saveSteps,approx=False):
	"""
	Runs Monte Carlo Metropolis-Hastings simulations of diffusion along the PMF reaction coordinate

	Parameters
	----------
	xs : ndArray(shape=[None], dtype=float)
		Array of the PMF reaction coordinate positions

	ys : ndArray(shape=[None], dtype=float)
		Array of the PMF free energies

	c : float
		Center of the umbrella

	k : float
		Force constant of the umbrella

	std : float
		Standard deviation of the normal distribution which describes the diffusion rate of the system of interest

	states : ndArray(shape=[None], dtype=float)
		Array tracking the current "location" of parallel MCMC simulations

	steps : int
		The number of MCMC steps to be performed

	temp : float
		Temperature of the system

	saveSteps : int
		Number of steps between saved points

	approx : bool
		Approximates the diffusion of multiple steps (saveSteps) into a single step. This provides a speedup by a factor approximately equal to saveSteps

	Returns
	-------
	(ndArray(shape=[None,steps // saveSteps], dtype=float), ndArray(shape=[None], dtype=float))
		The first argument is a matrix detailing the trajectory of the position of the system along the PMF RC for each parallel simulation
		The second argument is the vector detailing the state of the systems at the end of the simulation (for continuing the simulation if desired)

	"""
	bias = (k / 2) * np.square(states - c)

	currentEs = njitGetPointEs(xs,ys,states) + bias

	if approx:
		std = np.sqrt(np.square(std) * saveSteps)
		steps = steps // saveSteps
		saveSteps = 1

	result = np.zeros((len(states),steps + 1))

	result[:,0] = states

	for step in range(steps):
		randSteps = np.random.normal(0, scale=std, size=len(states))

		bias = (k / 2) * np.square(states+randSteps - c)

		newEs = njitGetPointEs(xs,ys,states+randSteps) + bias

		deltaEs = (newEs - currentEs) * 1000

		probs = np.exp(-(deltaEs/NA)/(KB*temp))

		for i,(prob,randStep,newE) in enumerate(zip(probs,randSteps,newEs)):
			uniform = np.random.rand()
			if uniform < prob:
				states[i] += randStep
				currentEs[i] = newE

			result[i,step+1] = states[i]

	return result[:,::saveSteps], states

@njit(cache=True)
def njitComputeACTime(arr):
	"""
	Given a sequence of temporally related values, computes the autocorrelation time of that sequence.

	Parameters
	----------
	arr : ndArray(shape=[None], dtype=float)
		Array of temporal values

	Returns
	-------
	float

	"""

	acMean = np.expand_dims(njitSumAxis1(arr),1) / arr.shape[1]

	acVar = np.expand_dims(njitSumAxis1(np.square(arr - acMean)),1) / arr.shape[1]

	acStd = np.sqrt(acVar)

	normed = (arr - acMean) / acStd

	# To prevent overflowing memory, the sequence is subsampled to 100,000 values
	skip = int(np.ceil(arr.shape[1] / 100000))

	normed = normed[:,::skip]

	m = int(np.ceil(5 * normed.shape[1] / 1000))

	acTime = np.ones(arr.shape[0])

	for i in np.arange(1,m):
		covar = normed[:,:-i] * normed[:,i:]
		corr = njitSumAxis1(covar) / covar.shape[1]

		acTime += 2 * corr * skip

	return np.mean(acTime)

def findMaxSimTime(xPmf,yPmf,targetDist,c,k,diffC,tol=0.01,temp=300,minStep=1e-10,intTime=1e-15,numReplicas=1,percentile=None,convFactor=1,saveSteps=100,saveApprox=False):
	"""
	Outer wrapper for running MC Metropolis simulations until convergence.

	Parameters
	----------
	xPmf : ndArray(shape=[None], dtype=float)
		Array of the PMF reaction coordinate positions

	yPmf : ndArray(shape=[None], dtype=float)
		Array of the PMF free energies

	targetDist : ndArray(shape=[None], dtype=float)
		Array of the expected distribution given the umbrella parameters and PMF

	c : float
		Center of the umbrella

	k : float
		Force constant of the umbrella

	diffC : float
		Standard deviation of the normal distribution which describes the diffusion rate of the system of interest

	tol : float
		The tolerance for the convergence criteria (should be > 0)

	temp : float
		The temperature of the system

	minStep : float
		The smallest amount of time to run a simulation

	intTime : float
		The integration time of the simluation

	numReplicas : int
		Number of simulation systems to run in parallel

	percentile : float
		The percentile (0,1] of simulation replicas which are necessary to be converged for the loop to halt.

	convFactor : int
		A convergence factor within simulations. Each simulation will be split into _convFactor_ partitions and convergence across those partitions will be evaluated.

	saveSteps : int
		Number of steps between recorded points (where each step is the integration time)

	saveApprox : bool
		Boolean indicating whether saveSteps should be subsumed into a single step by altering _diffC_. If true, speeds up calculations at the cost of accuracy.

	Returns
	-------
	count, acTime, hists, dists, klDivs[ind], convScore

	(float, float, ndArray(shape=[None, None], dtype=float), ndArray(shape=[None, None], dtype=float), float, float)
		1. The amount of time for the specified percentile of simulations to achieve convergence
		2. The average autocorrelation time of the simulations
		3. An array of histograms of the obtained distributions for each replica
		4. An array of points sampled for each replica
		5. The KL Divergence between the expected and real distributions at the specified percentile
		6. The KL Divergence between the individual simulation partitions at the specified percentile

	"""

	# If no percentile is provided set to the leave one out percentile
	if percentile is None:
		percentile = 1 - (1 / numReplicas)

	maxInd = np.argmax(targetDist)

	# Initialize the simulations at the most probable location
	states = np.array([xPmf[maxInd] for _ in range(numReplicas)])
	states = np.minimum(np.max(xPmf),states)
	states = np.maximum(np.min(xPmf),states)

	hists = np.zeros((numReplicas,len(xPmf)))
	dists = None

	count = 0

	done = False
	while not done:
		score = 0

		# Run the simulation for minStep steps
		newDists, states = njitRunMCMetroSim(xPmf,yPmf,c,k,diffC,states,int(minStep/intTime),temp=temp,saveSteps=saveSteps,approx=saveApprox)

		if dists is None:
			dists = newDists
		else:
			dists = np.concatenate([dists,newDists],axis=1)

		newHists = pointsToHist(xPmf, newDists, norm=False)

		hists += newHists

		# Check the divergence between the obtained and expected distributions
		klDivs = []
		for d in hists:
			klDivs.append(klDiv(targetDist,d))

		count += minStep
		minStep *= 1.1

		argsort = np.argsort(klDivs)

		ind = argsort[int(numReplicas * percentile - 1)]

		score += klDivs[ind]

		numSteps = dists.shape[1]

		# Check the divergences within each simulation's partition
		if convFactor > 1:
			replicaScores = []
			for replica in dists:
				worstScore = 0
				for i in range(0,convFactor - 1):
					startIInd = int((i / convFactor) * numSteps)
					endIInd = int(((i+1) / convFactor) * numSteps)
					distI = pointsToHist(xPmf,[replica[startIInd:endIInd]], norm=False)[0]
					for j in range(i+1, convFactor):
						startJInd = int((j / convFactor) * numSteps)
						endJInd = int(((j + 1) / convFactor) * numSteps)
						distJ = pointsToHist(xPmf, [replica[startJInd:endJInd]], norm=False)[0]

						myScore = klDiv(distI, distJ)
						if myScore > worstScore:
							worstScore = myScore
				replicaScores.append(worstScore)

			convArgsort = np.argsort(replicaScores)
			convInd = convArgsort[int(len(convArgsort) * percentile - 1)]
			convScore = replicaScores[convInd]

			score = np.maximum(score,convScore)

			print("%d, %.4f, %.4f"%(count/intTime, klDivs[ind], convScore),end='\r')
		else:
			print("%d, %.4f"%(count/intTime, klDivs[ind]))

		# If both convergence measures have been achieved, or 50ns simulation time has elapsed, exit the loop.
		if score <= tol or count > 50000000 * intTime:
			done = True

	hists = hists / np.sum(hists,axis=-1,keepdims=True)

	acTime = njitComputeACTime(dists)

	acTime *= saveSteps

	print()
	print()
	return count, acTime, hists, dists, klDivs[ind], convScore

def computeDiffusionConstant(xs, ys, points, centers, ks, timeBetweenPoints, temp, intTime=1e-15, periodic=False):
	"""
	Computes the diffusion constant arising from treating simulations as MCMC simulations

	Parameters
	----------
	xs : ndArray(shape=[None], dtype=float)
		Array of the PMF reaction coordinate positions

	ys : ndArray(shape=[None], dtype=float)
		Array of the PMF free energies

	points : ndArray(shape=[None, None], dtype=float)
		A series of recorded PMF coordinates over the course of a simulation. Each entry in axis 0 corresponds to one series to be binned, and each entry in axis 1 is an individual coordinate

	centers : ndArray(shape=[None], dtype=float)
		An array of values corresponding to the umbrella centers used for each simulation

	ks : ndArray(shape=[None], dtype=float)
		An array of values corresponding to the umbrella spring constants used for each simulation

	timeBetweenPoints : float
		The time difference between each adjacent pair of points.

	temp : float
		Temperature used for simulation

	intTime : float
		Integration time used for simulation

	periodic : bool
		Boolean indicating whether the system is periodic

	Returns
	-------
	float

	"""

	avgStd = []

	for entry, c, k in zip(points,centers,ks):
		if periodic:
			diff = njitGetPeriodicDifference(entry[1:], entry[:-1])
		else:
			diff = entry[1:] - entry[:-1]

		myDxs = diff / np.sqrt(timeBetweenPoints / intTime)

		bias = (k/2) * np.square(entry - c)
		es = njitGetPointEs(xs,ys,entry) + bias

		deltaEs = (es[1:] - es[:1]) * 1000

		weights = 1 / np.minimum(1,np.exp(-(deltaEs/NA)/(KB*temp)))

		mean = np.average(myDxs,weights=weights)
		var = np.average(np.square(myDxs - mean),weights=weights)

		avgStd.append(np.sqrt(var))

	return np.array(avgStd)