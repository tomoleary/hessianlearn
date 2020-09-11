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
import time
import sys
import numpy as np


from scipy.linalg import cholesky, eigh, solve_triangular, qr, rq

import time




def adaptive_range_finder(Aop,n,rank_guess,tolerance, seed = 0):
    # This case handles Hessians (symmetric matrices),
    # So operation Aop(\cdot) is taken to also be its transpose
    # In general case (SVD) one needs to work out matmult and transpmult
    # and pass these into the range finder as well



    pass

def noise_aware_adaptive_range_finder(Aop,n,rank_guess,tolerance, noise_tolerance,seed = 0):

    # Check variance of last eigenvector
    # If its less than noise_tolerance, and error is greater than tolerance then sample more
    # If its less than noise_tolerance and error is less than tolerance then stop, return range
    # If its greater than noise tolerance, stop and use binary search to work backwards for 
    # vector with RQ variance below noise tolerance, and return only these first columns



    pass



