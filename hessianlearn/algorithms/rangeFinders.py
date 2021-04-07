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
    """
    Randomized algorithm for block range finding    
    
    Parameters:
    -----------
    Aop : {Callable} n x n symmetric matrix
          Hermitian matrix operator whose eigenvalues need to be estimated
          y = Aop(dw) is the action of A in the direction dw 
    n   : size of matrix A
            
    Returns:
    --------
    Q : range for Aop
    """
    # Taken from http://people.maths.ox.ac.uk/martinsson/Pubs/2015_randQB.pdf

    my_state = np.random.RandomState(seed=seed)
    w = my_state.randn(n,1)
    w /= np.linalg.norm(w)
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
        block_size = None,noise_tolerance = 1.0,epsilon = 1e-1, max_vectors = 20, verbose = False,seed = 0):
    """
    Randomized algorithm for noise aware block range finding  (N.A.A.R.F.)
    
    Parameters:
    -----------
    Hessian : 
    hessian_feed_dict : 
    rq_estimator_dict : 
    block_size :
    noise_tolerance :
    epsilon : 
    verbose : 
    seed :
            
    Returns:
    --------
    Q : range for dominant eigenmodes of Hessian
    """

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

        rq_snr = np.abs(np.mean(RQ_samples,axis=0))/np.std(RQ_samples,axis = 0)
        too_noisy = (rq_snr < noise_tolerance).any()
        converged = (operator_error < epsilon) or too_noisy
        # print(80*'#')
        # print('rq_snr = ',rq_snr)
        # print('rq_snr < noise_tolerance = ',rq_snr < noise_tolerance)
        # print('too noisy? = ',too_noisy)
        # print('(operator_error < epsilon) = ',(operator_error < epsilon))
        # print(80*'#')
        
        iteration+=1 
        if verbose:
            print('At iteration', iteration, 'operator error is ',operator_error,' convergence = ',(operator_error < epsilon))
        if big_Q.shape[-1] >= max_vectors:
            break

        if iteration > n//block_size:
            break
    # I believe that the extra action of A_op in forming B for the QB factorization 
    # is cheaper to do once after the fact, and is not needed for the matrix 
    # free randomized error estimator. For this reason I just return Q, and 
    # do not form B.
    return big_Q



