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
from hessianlearn import (block_range_finder)

class TestRangeFinders(unittest.TestCase):

	def test_basic(self):
		my_state = np.random.RandomState(seed=0)
		n = 100
		Q,_ = np.linalg.qr(my_state.randn(n,n))
		d = np.concatenate((np.ones(10),np.exp(-np.arange(n-10))))
		Aop = lambda x: Q@np.diag(d)@(Q.T@x)

		Q_range = block_range_finder(Aop,100,1e-5,10)
		assert Q_range.shape[-1] <=40
		w_action = my_state.randn(100,1)
		action = Aop(w_action)
		error = np.linalg.norm(action - Q_range@(Q_range.T@ action))
		print(error)
		assert error < 1e-5

if __name__ == '__main__':
    unittest.main()