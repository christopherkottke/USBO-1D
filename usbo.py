from readParams import Params
from findNewUmbrellas import runBBOpt
import os
import sys

if __name__ == "__main__":
	pathToDat = sys.argv[1]

	paramDict = Params()
	paramDict.loadParams(pathToDat)

	os.chdir(paramDict["base_path"])

	runBBOpt(paramDict)