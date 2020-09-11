# This file is part of the hessianlearn package
#
# hessianlearn is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
#
# hessianlearn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# If not, see <http://www.gnu.org/licenses/>.
#
# Author: Tom O'Leary-Roseberry
# Contact: tom.olearyroseberry@utexas.edu
from __future__ import absolute_import, division, print_function

import unittest 
import numpy as np
import sys

sys.path.append('../../')
from hessianlearn import (randomized_eigensolver)

class TestRandomizedEigensolver(unittest.TestCase):

	def test_basic(self):
		my_state = np.random.RandomState(seed=0)
		n = 100
		Q,_ = np.linalg.qr(my_state.randn(n,n))
		d = np.concatenate((np.ones(10),np.exp(-np.arange(n-10))))
		Aop = lambda x: Q@np.diag(d)@Q.T@x
		d_hl, Q_hl = randomized_eigensolver(Aop,100, 100)
		assert np.linalg.norm(d[:50] - d_hl[0:50]) < 1e-10

if __name__ == '__main__':
    unittest.main()