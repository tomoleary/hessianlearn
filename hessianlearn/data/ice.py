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

def load_ice(path_to_ice = '',type = 'hessian',m_full = False):
	print('M full?',m_full)
	try: 
		# read from file
		if type == 'hessian':
			if m_full:
				ms = np.load(path_to_ice+'ice_hessian_all_m_fulls.npy')
			else:
				ms = np.load(path_to_ice+'ice_hessian_all_ms.npy')
			qois = np.load(path_to_ice+'ice_hessian_all_qois.npy')
		elif type == 'iid':
			if m_full:
				ms = np.load(path_to_ice+'ice_iid_all_m_fulls.npy')
			else:
				ms = np.load(path_to_ice+'ice_iid_all_ms.npy')
			qois = np.load(path_to_ice+'ice_iid_all_qois.npy')
		elif type == 'test':
			if m_full:
				ms = np.load(path_to_ice+'ice_test_all_m_fulls.npy')
			else:
				ms = np.load(path_to_ice+'ice_test_all_ms.npy')
			qois = np.load(path_to_ice+'ice_test_all_qois.npy')
		else:
			raise ValueError(type)
	except:
		# write to file
		print(80*'#')
		print('Did not load')
		print(80*'#')

		ms = []
		qois = []
		if type == 'hessian':
			ice_folder = path_to_ice + 'hessian_vecs/'
		elif type == 'iid':
			ice_folder = path_to_ice + 'iid_vecs/'
		elif type == 'test':
			ice_folder = path_to_ice + 'test_vecs/'
		else:
			raise ValueError(type)
		if not os.path.isdir(ice_folder):
			print('Make sure to execute from a directory containing greedy_vecs/ or iid_vecs/ or to explicitly link the path')

		if m_full:
			m_folder = 'm_full/'
		else:
			m_folder = 'm/'
		for file_index in os.listdir(ice_folder + m_folder):
			if os.path.exists(ice_folder+m_folder+file_index):
				ms.append(np.load(ice_folder + m_folder+file_index))
				qois.append(np.load(ice_folder+'qoi/'+file_index))
				print('Loaded ',file_index)
			else:
				print('Oopsy doopsy')

		ms = np.array(ms)
		qois = np.expand_dims(qois,axis = -1)
		m_name = path_to_ice
		qoi_name = path_to_ice
		if type == 'hessian':
			if m_full:
				m_name += 'ice_hessian_all_m_fulls.npy'
			else:
				m_name += 'ice_hessian_all_ms.npy'
			qoi_name += 'ice_hessian_all_qois.npy'
		elif type == 'iid':
			if m_full:
				m_name += 'ice_iid_all_m_fulls.npy'
			else:
				m_name += 'ice_iid_all_ms.npy'
			qoi_name += 'ice_iid_all_qois.npy'
		elif type == 'test':
			if m_full:
				m_name += 'ice_test_all_m_fulls.npy'
			else:
				m_name += 'ice_test_all_ms.npy'
			qoi_name += 'ice_test_all_qois.npy'

		np.save(m_name,ms)
		qois = np.array(qois)
		qois = np.expand_dims(qois,axis = -1)
		if len(qois.shape) == 3:
			print('Expanding qoi dimension again')
			qois = np.expand_dims(qois,axis = -1)
		np.save(qoi_name,qois)
	return [ms,qois]





