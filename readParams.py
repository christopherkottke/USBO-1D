import copy

class Params():
	"""
	A class for reading input files for the USBO method

	Attributes
	__________
	defaultArgDict : dict
		Dictionary containing default parameter values for the USBO method. Parameters which must be set by the user default to 'None'.
	"""

	def __init__(self):
		self.defaultArgDict = {
			'base_path': None,
			'n_umbrellas': None,
			'clb': None,
			'cub': None,
			'dist_clb': None,
			'dist_cub': None,
			'base_klb': None,
			'base_kub': None,
			'sim_data_path': None,
			'pmf_folder' : None,
			'pmf_path' : None,
			'meta_path' : None,
			'output_name' : None,
			'mm_weight': '1e6',
			'unsampled_weight': '1e2',
			'unsampled_cutoff': '1e-4',
			'bho_opt_size': '10',
			'bho_gp_n_restarts': '10',
			'n_replicas': '100',
			'min_step': '1e-11',
			'sim_conv_tol': '0.05',
			'sim_conv_factor': '2.0',
			'n_x_points': '800',
			'temp': '300.0',
			'int_time': '1e-15',
			'save_steps': '10',
			'save_approx': '0',
			'umb_search_iters': '10',
			'umb_search_bSize': '10',
			'periodic': '0',
			'sim_int_time' : '1e-15',
			'time_between_points' : '1e-15',
			'last_param_file' : 'None',
			'use_prior_dists' : '0'
		}

	def loadParams(self, path):
		"""
		Function for loading USBO parameters from file.

		Parameters
		----------
		path : String
			Path to USBO input file

		Returns
		-------
		None
		"""
		self.argDict = copy.copy(self.defaultArgDict)

		with open(path, 'r') as f:
			lines = f.readlines()

			for line in lines:
				if line[0] != "#" and len(line) > 1:
					lineArr = [s.strip() for s in line.split("=")]
					print(lineArr)
					self.argDict[lineArr[0]] = lineArr[1]

		self.checkParams()

	def __getitem__(self, arg):
		"""
		Function for referencing specific parameters in the self.defaultArgDict dictionary.

		Parameters
		----------
		arg : String
			Dictionary Key

		Returns
		-------
		Object corresponding to the stored dictionary entry

		"""
		if arg not in self.defaultArgDict.keys():
			raise KeyError("%s is not a recognized argument."%arg)
		else:
			return self.argDict[arg]

	def checkParams(self):
		"""
		Function for checking that the mandatory user settings are set.

		Parameters
		----------
		None

		Returns
		-------
		None
		"""

		manditoryParams = ["base_path", "n_umbrellas", "periodic", "clb", "cub", "dist_clb", "dist_cub", "base_klb", "base_kub", "sim_data_path", "pmf_folder", "pmf_path", "meta_path", "output_name"]

		for param in manditoryParams:
			if self.argDict[param] is None:
				raise Exception("%s was not specified in the provided input file."%param)

		if self.argDict["last_param_file"] == "None":
			self.argDict["last_param_file"] = None