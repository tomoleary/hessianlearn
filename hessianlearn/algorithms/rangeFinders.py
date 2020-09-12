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


def block_range_finder(A_op,n,epsilon,block_size):
    # Taken from http://people.maths.ox.ac.uk/martinsson/Pubs/2015_randQB.pdf
    # Assumes A is symmetric
    my_state = np.random.RandomState(seed=0)
    W = my_state.randn(n,1)
    Action = A_op(W)
    big_Q = None
    converged = False
    iterator = 0
    while not converged:
        # Sample Gaussian random matrix
        Omega = my_state.randn(n,block_size)
        # Perform QR on action
        Q,_ = np.linalg.qr(A_op(Omega))
        # Update basis
        if big_Q is None:
            big_Q = Q
        else:
            Q -= big_Q@(big_Q.T@Q)
            big_Q = np.concatenate((big_Q,Q),axis = 1)
            big_Q,_ = np.linalg.qr(big_Q)
        # Error estimation
        Approximate_Error = Action - big_Q@(big_Q.T@Action)
        error = np.linalg.norm(Approximate_Error)
        converged = error < epsilon
        iterator+=1 
        if iterator > n//block_size:
            break
    return big_Q



def noise_aware_adaptive_range_finder(Aop,n,rank_guess,tolerance, noise_tolerance,seed = 0):

    # Check variance of last eigenvector
    # If its less than noise_tolerance, and error is greater than tolerance then sample more
    # If its less than noise_tolerance and error is less than tolerance then stop, return range
    # If its greater than noise tolerance, stop and use binary search to work backwards for 
    # vector with RQ variance below noise tolerance, and return only these first columns



    pass



