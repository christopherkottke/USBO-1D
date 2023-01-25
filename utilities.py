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

KB = 3.297623483e-24 # Boltzmann Constant (cal/K)
NA = 6.02214e23 # Avogadro's number

from tf_utils import estimateFullSampleDist

@njit(cache=True)
def njitSumAxis0(a):
	"""
	An njit function for summing along the zero axis of a 2D array.

	Note: This is done as it is significantly faster than the native numpy.sum(a, axis=0) call

	Parameters
	----------
	a : ndArray(shape=[None,None], dtype=float)
		2D array to be summed over

	Returns
	-------
	ndArray(shape=[None], dtype=float)

	"""

	result = np.zeros(a.shape[1:])

	for entry in a:
		result += entry

	return result

@njit(cache=True)
def njitSumAxis1(a):
	"""
	An njit function for summing along the one axis of a 2D array.

	Note: This is done as it is significantly faster than the native numpy.sum(a, axis=1) call

	Parameters
	----------
	a : ndArray(shape=[None,None], dtype=float)
		2D array to be summed over

	Returns
	-------
	ndArray(shape=[None], dtype=float)

	"""

	result = np.zeros(a.shape[0])

	for entry in a.T:
		result += entry

	return result

@njit(cache=True)
def njitD1Sum(a):
	"""
	An njit function for summing a 1D array.

	Note: This is done as it is significantly faster than the native numpy.sum(a) call

	Parameters
	----------
	a : ndArray(shape=[None], dtype=float)
		1D array to be summed over

	Returns
	-------
	float

	"""

	result = 0
	for entry in a:
		result += entry
	return result

@njit(cache=True)
def njitGetPointEs(xs,ys,points):
	"""
	Uses the PMF at defined points to interpolate between them to provide a continuous energy surface

	Parameters
	----------
	xs : ndArray(shape=[None], dtype=float)
		Array of the PMF reaction coordinate positions

	ys : ndArray(shape=[None], dtype=float)
		Array of the PMF free energies

	points : ndArray(shape=[None], dtype=float)
		Array of the PMF reaction coordinates of interests

	Returns
	-------
	ndArray(shape=[None], dtype=float)

	"""

	result = []
	for point in points:
		ind = np.where(point - xs > 0, point - xs, np.inf).argmin()
		frac = (point - xs[ind])/(xs[1] - xs[0])

		if frac < 0:
			myE = ys[0]
		else:
			startE = ys[ind]
			if ind == len(ys) - 1:
				endE = ys[ind]
			else:
				endE = ys[ind+1]

			myE = startE + (endE - startE) * frac

		result.append(myE)

	return np.array(result)

@njit(cache=True)
def njitGetPeriodicDifference(sAngle, eAngle):
	"""
	Computes the difference between two angles over a periodic boundary at 360 degrees

	Parameters
	----------
	sAngle : float
		The starting angle

	eAngle : float
		The ending angle

	Returns
	-------
	float

	"""

	degDist = eAngle - sAngle
	degDist = (degDist + 180) % 360 - 180

	return degDist

def klDiv(f1,f2=None):
	"""
	Computes the symmetric KL divergence between a provided distribution and another distribution (uniform if None)

	Parameters
	----------
	f1 : ndArray(shape=[None], dtype=float)
		Array of the first probability density

	f2 : ndArray(shape=[None], dtype=float)
		Array of the first probability density

	Returns
	-------
	float
		The symmetric KL Divergence

	"""

	f1Prob = f1 / (njitD1Sum(f1) + 1e-8) + 1e-8

	if f2 is None:
		f2Prob = np.ones(len(f1)) / len(f1)
	else:
		f2Prob = f2 / (njitD1Sum(f2) + 1e-8) + 1e-8

	return njitD1Sum(f1Prob * np.log(f1Prob/f2Prob) + f2Prob * np.log(f2Prob/f1Prob))

def plotExpectedHistograms(xPmf,yPmf,cs,ks,temp,periodic,distBounds,startDists=None,path="",title="",xLabel=""):
	"""
	Given a PMF and a set of parameters, creates a plot of the expected distributions resulting from the umbrellas created by those parameters

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

	periodic : bool
		Boolean indicating whether the system is periodic

	distBounds : (float, float)
		Tuple of floats describing the desired range of xPmf values

	startDists : ndArray(shape=[None, None], dtype=float)
		Arrays of the pre-existing distributions to be added to estimated distributions

	path : String
		Path to save the resulting plot

	title : String
		Plot title

	xLabel : String
		Label for the plot x-axis

	Returns
	-------
	None

	"""

	estDists, estExp = estimateFullSampleDist(xPmf,yPmf,cs[np.newaxis,:],ks[np.newaxis,:],temp=temp)

	estDists = estDists[0]

	if periodic:
		xSize = len(xPmf) // 3
		xPmf = xPmf[xSize:2 * xSize]
		estDists = estDists[:, :xSize] + estDists[:, xSize:2 * xSize] + estDists[:, 2 * xSize:3 * xSize]

	inds = tf.transpose(tf.where(tf.logical_and(xPmf >= distBounds[0], xPmf <= distBounds[1])))[0]

	xPmf = tf.gather(xPmf,inds)
	estDists = tf.gather(estDists,inds,axis=-1)

	fig, ax = plt.subplots(figsize=(3.25, 3.25),dpi=1000,layout="constrained")

	if startDists is not None:
		for entry in startDists:
			if periodic:
				entry = entry[:, :xSize] + entry[:, xSize:2 * xSize] + entry[:, 2 * xSize:3 * xSize]
				entry = entry[inds]
			ax.plot(xPmf, entry, c="black", linewidth=3)

	for entry in estDists:
		ax.plot(xPmf, entry, linewidth=3)

	ax.set_title(title)
	ax.set_ylabel("Relative Frequency")
	ax.set_xlabel(xLabel)
	plt.savefig(path)
	plt.close()

def loadParamsFromFile(file):
	"""
	Provided a path to a USBO output file, loads the umbrella centers, force constants, and loss values listed in the file

	Parameters
	----------
	file : String
		Path to a USBO output file

	Returns
	-------
	(ndArray(shape=[None], dtype=float), ndArray(shape=[None], dtype=float), float, float, float, float)
		Returns the umbrella centers, umbrella force constants, total loss under those parameters, and component losses.

	"""

	cs = []
	ks = []

	with open(file,'r') as f:
		lines = f.readlines()
		lossArr = [float(s) for s in lines[-1].split(",")]

		totLoss = lossArr[0]
		klLoss = lossArr[1]
		upLoss = lossArr[2]
		mmLoss = lossArr[3]

		for line in lines[:-1]:
			lineArr = [float(s) for s in line.split(",")]
			cs.append(lineArr[0])
			ks.append(lineArr[1])

	return np.array(cs), np.array(ks), totLoss, klLoss, upLoss, mmLoss

def getPoints(basePath, pmfFolder, meta):
	"""
	Loads the values monitored along the PMF RC axis during simulations.

	Parameters
	----------
	basePath : String
		Path to the outer folder

	pmfFolder : String
		Path from the outer folder to the folder containing the PMF information

	meta : [dict]
		Array of dictionaries where each dictionary entry contains the subpath to the file containing the simulation PMF RC values and the umbrella center and force constants used for that simulation.

	Returns
	-------
	[ndArray(shape=[None], dtype=float)]
		Array of simulation PMF RC values where each entry in the array is a numpy array of the values obtained from a single simulation.

	"""

	result = []

	for entry in meta:
		myName = entry["fName"]

		myArr = []
		with open("%s/%s/%s"%(basePath,pmfFolder,myName), 'r') as f:
			lines = f.readlines()

		for line in lines:
			lineArr = line.split()

			entry = float(lineArr[1])

			myArr.append(entry)

		result.append(np.array(myArr))

	return result

def readMeta(metaPath):
	"""
	Provided a path to the meta file used to generate a PMF using WHAM, stores the paths to the individual simulations as well as the simulation umbrella parameters

	Parameters
	----------
	metaPath : String
		Path to the meta file.

	Returns
	-------
	[dict]
		Array of dictionaries where each dictionary entry contains the subpath to the file containing the simulation PMF RC values and the umbrella center and force constants used for that simulation.

	"""

	with open(metaPath,'r') as f:
		lines = f.readlines()

	result = []

	for line in lines:
		if line != "\n":
			lineArr = line.split()
			result.append({"fName": lineArr[0], "c": float(lineArr[1]), "k": float(lineArr[2])})

	return result

def getPMF(path):
	"""
	Provided a path to a PMF file, loads that PMF

	Parameters
	----------
	path : String
		Path to a PMF file

	Returns
	-------
	(ndArray(shape=[None], dtype=float), ndArray(shape=[None], dtype=float))
		Returns a tuple containing the RC coordinates of the PMF and the energy values at each RC coordinate.

	"""

	with open("%s"%path,'r') as f:
		lines = f.readlines()

	xs = []
	ys = []

	for line in lines[1:]:
		lineArr = line.replace(","," ").split()
		if lineArr[1] != "inf" and lineArr[0][0] != "#":
			xs.append(float(lineArr[0]))
			ys.append(float(lineArr[1]))

	return np.array(xs), np.array(ys)

def getKLB(xs, ys, baseKLB, baseKUB, temp, lastParams, relativeHeight=0.6):
	"""
	This function computes the minimum value of k which produces a distribution with a peak at least as high as relativeHeight * the largest peak produced by the provided parameters

	Parameters
	----------
	xs : ndArray(shape=[None], dtype=float)
		Array of the PMF reaction coordinate positions

	ys : ndArray(shape=[None], dtype=float)
		Array of the PMF free energies

	baseKLB : float
		Lower bound of the umbrella spring constant

	baseKUB : (float, float)
		Upper bound of the umbrella spring constant

	temp : float
		Temperature of the system

	lastParams : (ndArray(shape=[None], dtype=float), ndArray(shape=[None], dtype=float))
		Tuple where the first argument is the array of umbrella centers from the last iteration, and the second argument is the array of spring constants from the last iteration

	relativeHeight : float
		Float determining how low a distribution's peak should be in proportion to the largest peak resulting from lastParams

	Returns
	-------
	float

	"""

	lastCs, lastKs = lastParams

	estDist, estExp = estimateFullSampleDist(xs,ys,lastCs[np.newaxis,:],lastKs[np.newaxis,:],temp=temp).numpy()[0]

	max1 = np.max(estDist)

	def getDistMax(k):
		"""
		This function takes a value of k and computes the highest peak obtainable given an underlying PMF using that value of k and any arbitrary value of c

		Parameters
		----------
		k : float
			Spring constant

		Returns
		-------
		float

		"""
		myDist, estExp = estimateFullSampleDist(xs, ys, np.array([[(xs[-1] + xs[0]) / 2]]), k[np.newaxis,:],temp=temp).numpy()[0]
		return np.max(myDist)

	res = minimize(lambda x: np.square(getDistMax(x) - relativeHeight * max1),x0=np.array(baseKLB),bounds=[(baseKLB,baseKUB)],method="L-BFGS-B")

	autoKLB = res.x[0]

	return autoKLB

def pointsToHist(xs, points, norm=True):
	"""
	This function takes a series of points and bins them according to xs in order to make a histogram

	Parameters
	----------
	xs : ndArray(shape=[None], dtype=float)
		Array of the PMF reaction coordinate positions

	points : ndArray(shape=[None, None], dtype=float)
		A series of recorded PMF coordinates over the course of a simulation. Each entry in axis 0 corresponds to one series to be binned, and each entry in axis 1 is an individual coordinate

	norm : bool
		A boolean to indicate whether the generated histograms should sum to one. (default True)

	Returns
	-------
	ndArray(shape=[None, None], dtype=float)
		The histograms resulting from binning. Entries in axis 0 correspond to histograms for each series. Entries in axis 1 correspond to individual bins according to xs.

	"""

	hists = []
	histBins = np.concatenate([xs - (xs[1] - xs[0]) / 2, [xs[-1] + (xs[1] - xs[0]) / 2]])

	for entry in points:
		myData, _ = np.histogram(entry, bins=histBins)

		if norm:
			hists.append(myData / np.sum(myData))
		else:
			hists.append(myData)

	hists = np.array(hists)
	return hists