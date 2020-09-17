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


def block_range_finder(A_op,n,epsilon,block_size,verbose = False,seed = 0):
    # Taken from http://people.maths.ox.ac.uk/martinsson/Pubs/2015_randQB.pdf
    # Assumes A is symmetric
    my_state = np.random.RandomState(seed=seed)
    w = my_state.randn(n,1)
    Action = A_op(w)
    big_Q = None
    converged = False
    iteration = 0
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
            # This QR gets slow after many iterations, only last columns
            # need to be orthonormalized
            big_Q,_ = np.linalg.qr(big_Q)
        # Error estimation
        Approximate_Error = Action - big_Q@(big_Q.T@Action)
        error = np.linalg.norm(Approximate_Error)
        converged = error < epsilon
        iteration+=1 
        if verbose:
            print('At iteration', iteration, ' error is ',error,' converged = ',converged)
        if iteration > n//block_size:
            break
    # I believe that the extra action of A_op in forming B for the QB factorization 
    # is cheaper to do once after the fact, and is not needed for the matrix 
    # free randomized error estimator. For this reason I just return Q, and 
    # do not form B.
    return big_Q




def noise_aware_adaptive_range_finder(Hessian,hessian_feed_dict,rq_estimator_dict_list,\
        block_size = None,noise_tolerance = 1e-1,epsilon = 1e-1, verbose = False,seed = 0):
    # A_op(w_hat,feed_dict) is a function that implements the action of the [n x n] 
    # symmetric finite sum operator A on an [n x k] ndarray
    # the second argument is the data dictionary for the finite sum operator
    # n is the dimension of A
    # hessian_feed_dict is the data used in the Hessian action
    # noise_tolerance is the truncation condition for spectral variance
    # epsilon is the truncation condition for the operator error estimator
    # block_size is the chunk size used in the iterative range estimator
    # rq_estimator_dict is a secondary data dictionary used for estimation of the 
    # spectral noise, if None then hessian_feed_dict will be used
    # partitions is the number of partitions that will be used in the data
    # for the spectral noise truncation condition


    # Check variance of last eigenvector
    # If its less than noise_tolerance, and error is greater than tolerance then sample more
    # If its less than noise_tolerance and error is less than tolerance then stop, return range
    # If its greater than noise tolerance, stop and use binary search to work backwards for 
    # vector with RQ variance below noise tolerance, and return only these first columns
    ###################################################################################
    assert type(rq_estimator_dict_list) is list
    n = Hessian.dimension
    if block_size is None:
        block_size = int(0.01*n)
    my_state = np.random.RandomState(seed=seed)
    w = my_state.randn(n,1)

    H = lambda x: Hessian(x,hessian_feed_dict,verbose = verbose)
    Action = H(w)
    big_Q = None
    converged = False
    iteration = 0
    rq_noise = 0.

    while not converged:
        # Sample Gaussian random matrix
        Omega = my_state.randn(n,block_size)
        # Perform QR on action
        Q,_ = np.linalg.qr(H(Omega))
        # Update basis
        if big_Q is None:
            big_Q = Q
        else:
            Q -= big_Q@(big_Q.T@Q)
            big_Q = np.concatenate((big_Q,Q),axis = 1)
            # This QR gets slow after many iterations, only last columns
            # need to be orthonormalized
            big_Q,_ = np.linalg.qr(big_Q)
        # Error estimation is both for operator error
        # as well as spectral noise
        # Operator error estimation
        Approximate_Error = Action - big_Q@(big_Q.T@Action)
        operator_error = np.linalg.norm(Approximate_Error)
        # Noise error estimation    
        rq_direction = big_Q[:,-block_size:]
        try:
            RQ_samples = np.zeros((len(rq_estimator_dict_list),rq_direction.shape[1]))
        except:
            RQ_samples = np.zeros(len(rq_estimator_dict_list))
        if verbose:
            try:
                from tqdm import tqdm
                for samp_i,sample_dictionary in enumerate(tqdm(rq_estimator_dict_list)):
                    RQ_samples[samp_i] = Hessian.quadratics(rq_direction,sample_dictionary)
            except:
                print('Issue with tqdm')
                for samp_i,sample_dictionary in enumerate(rq_estimator_dict_list):
                    RQ_samples[samp_i] = Hessian.quadratics(rq_direction,sample_dictionary)
        else:
            for samp_i,sample_dictionary in enumerate(rq_estimator_dict_list):
                RQ_samples[samp_i] = Hessian.quadratics(rq_direction,sample_dictionary)

        rq_noise = np.max(np.std(RQ_samples,axis = 0))
        # print(80*'#')
        # print('Rayleigh quotient noise for the last direction is ',rq_noise)
        # print(80*'#')

        converged = (operator_error < epsilon) or (rq_noise > noise_tolerance)
        iteration+=1 
        if verbose:
            print('At iteration', iteration, 'operator error is ',operator_error,' convergence = ',(operator_error < epsilon))
            print('At iteration', iteration, 'RQ noise is ',rq_noise,' exit condition = ',(rq_noise > noise_tolerance))

        if iteration > n//block_size:
            break
    # I believe that the extra action of A_op in forming B for the QB factorization 
    # is cheaper to do once after the fact, and is not needed for the matrix 
    # free randomized error estimator. For this reason I just return Q, and 
    # do not form B.
    return big_Q



