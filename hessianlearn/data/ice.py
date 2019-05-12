from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import random
from ..data.data import *

import math
import time

# from statsmodels import robust

def dir_check(dir):
	try:
		os.stat(dir)
	except:
		os.mkdir(dir)

def load_ice(path_to_ice = ''):
	# read from file
	ms = np.load(path_to_ice+'ice_iid_all_m_fulls.npy')
	qois = np.load(path_to_ice+'ice_iid_all_qois.npy')
	return [ms,qois]

	





