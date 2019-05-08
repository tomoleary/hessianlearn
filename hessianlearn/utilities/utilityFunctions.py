from __future__ import absolute_import, division, print_function
import numpy as np




def zeros_like(b):
	zeros = []
	for item in b:
		zeros.append(np.zeros_like(item))
	return zeros




