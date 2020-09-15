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


def low_rank_hessian(optimizer,feed_dict,k,p=None,verbose = False):
    H = lambda x: optimizer.H_w_hat(x,feed_dict)
    n = optimizer.problem.dimension
    return randomized_eigensolver(H, n, k,p = p,verbose = verbose)




def randomized_eigensolver(Aop, n, k, p = None,seed = 0,verbose = False):
    """
    Randomized algorithm for Hermitian eigenvalue problems
    Returns k largest eigenvalues computed using the randomized algorithm
    
    
    Parameters:
    -----------
    Aop : {Callable} n x n
          Hermitian matrix operator whose eigenvalues need to be estimated
          y = Aop(w_hat) is the action of A in the direction w_hat 
          
    n : int,
           number of row/columns of the operator A
        
    k :  int, 
        number of eigenvalues/vectors to be estimated
    p :  int, optional
        oversampling parameter which can improve accuracy of resulting solution
        Default: 20
            
    Returns:
    --------
    
    d : ndarray, (k,)           
        eigenvalues arranged in descending order
    U : ndarray, (n,k)
        eigenvectors arranged according to eigenvalues
    
    References:
    -----------
    .. [1] Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp. "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions." SIAM review 53.2 (2011): 217-288.
    Examples:
    ---------
    >>> import numpy as np
    >>> n = 100
    >>> A = np.diag(0.95**np.arange(n))
    >>> Aop = lambda w_hat: np.dot(A,w_hat)
    >>> k = 10
    >>> p = 5
    >>> lmbda, U = randomized_eigensolver(Aop, n, k, p)
    """
    if n == k:
        p = 0
    elif p is None:
        p = int(0.01*k)
        if k+p > n:
            p = n - k
    random_state = np.random.RandomState(seed=seed)
    Omega = random_state.randn(n,k+p)
    n  = Omega.shape[0]  

    assert(n >= k )
    
    m = Omega.shape[1]

    # Y = np.zeros(Omega.shape, dtype = 'd')
    Y = np.zeros(Omega.shape)

    if verbose:
        print('Applying Hessian')
        try:
            from tqdm import tqdm
            for i in tqdm(range(m)):
                # print('i = ',i,' m = ',m)
                Y[:,i] = Aop(Omega[:,i])
                # print('Y[:,i] max = ',np.max(Y[:,i]))
                # print('Y[:,i] min = ',np.min(Y[:,i]))

        except:
            print('No progress bar :(')
            for i in range(m):
                # print(i,)
                Y[:,i] = Aop(Omega[:,i])
        # print('condition number for Y = ',np.linalg.cond(Y))
        Q,_ = qr(Y, mode = 'economic')
        T = np.zeros((m,m),dtype = 'd')
        print('Forming small square matrix')
        try:
            for i in tqdm(np.arange(m)):
                # print( i,)
                Aq = Aop(Q[:,i])    
                for j in np.arange(m):
                    T[i,j] = np.dot(Q[:,j].T,Aq)
        except:
            for i in np.arange(m):
                # print( i,)
                Aq = Aop(Q[:,i])    
                for j in np.arange(m):
                    T[i,j] = np.dot(Q[:,j].T,Aq)
    else:

        for i in range(m):
            # print(i,)
            Y[:,i] = Aop(Omega[:,i])
        Q,_ = qr(Y, mode = 'economic')
        
        T = np.zeros((m,m),dtype = 'd')

        for i in np.arange(m):
            # print( i,)
            Aq = Aop(Q[:,i])    
            for j in np.arange(m):
                T[i,j] = np.dot(Q[:,j].T,Aq)
                
    #Eigen subproblem
    if verbose:
        print('Computing eigenvalue decomposition')
    d, V = eigh(T)
    d_abs = np.abs(d) #sort by absolute value (we want the k largest eigenvalues regardless of sign)
    sort_perm = d_abs.argsort()
        
    sort_perm = sort_perm[::-1]
    
    d = d[sort_perm[0:k]]
    V = V[:, sort_perm[0:k]] 
    
    #Compute eigenvectors        
    U = np.dot(Q, V[:,::-1])    

    return d[:k], U[:,:k]


def randomized_eigensolver_from_range(Aop, Q,verbose = False):
    """
    Randomized algorithm for Hermitian eigenvalue problems
    Returns k largest eigenvalues computed using the randomized algorithm
    
    
    Parameters:
    -----------
    Aop : {Callable} n x n
          Hermitian matrix operator whose eigenvalues need to be estimated
          y = Aop(w_hat) is the action of A in the direction w_hat 
    Q : Array n x r
          
            
    Returns:
    --------
    
    d : ndarray, (k,)           
        eigenvalues arranged in descending order
    U : ndarray, (n,k)
        eigenvectors arranged according to eigenvalues
    
    References:
    -----------
    .. [1] Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp. "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions." SIAM review 53.2 (2011): 217-288.
    Examples:
    ---------
    >>> import numpy as np
    >>> n = 100
    >>> A = np.diag(0.95**np.arange(n))
    >>> Aop = lambda w_hat: np.dot(A,w_hat)
    >>> k = 10
    >>> p = 5
    >>> lmbda, U = randomized_eigensolver(Aop, n, k, p)
    """

    
    m = Q.shape[1]

    # Y = np.zeros(Omega.shape, dtype = 'd')
    Y = np.zeros(Omega.shape)

    if verbose:
        T = np.zeros((m,m),dtype = 'd')
        print('Forming small square matrix')
        try:
            for i in tqdm(np.arange(m)):
                # print( i,)
                Aq = Aop(Q[:,i])    
                for j in np.arange(m):
                    T[i,j] = np.dot(Q[:,j].T,Aq)
        except:
            for i in np.arange(m):
                # print( i,)
                Aq = Aop(Q[:,i])    
                for j in np.arange(m):
                    T[i,j] = np.dot(Q[:,j].T,Aq)
    else:

        for i in range(m):
            # print(i,)
            Y[:,i] = Aop(Omega[:,i])
        Q,_ = qr(Y, mode = 'economic')
        
        T = np.zeros((m,m),dtype = 'd')

        for i in np.arange(m):
            # print( i,)
            Aq = Aop(Q[:,i])    
            for j in np.arange(m):
                T[i,j] = np.dot(Q[:,j].T,Aq)
                
    #Eigen subproblem
    if verbose:
        print('Computing eigenvalue decomposition')
    d, V = eigh(T)
    d_abs = np.abs(d) #sort by absolute value (we want the k largest eigenvalues regardless of sign)
    sort_perm = d_abs.argsort()
        
    sort_perm = sort_perm[::-1]
    
    d = d[sort_perm[0:k]]
    V = V[:, sort_perm[0:k]] 
    
    #Compute eigenvectors        
    U = np.dot(Q, V[:,::-1])    

    return d[:k], U[:,:k]

def randomized_double_pass_eigensolver(Aop, Y, k):
    """
    Randomized algorithm for Hermitian eigenvalue problems
    Returns k largest eigenvalues computed using the randomized algorithm
    
    Parameters:
    -----------
    Aop : {Callable} n x n
          Hermitian matrix operator whose eigenvalues need to be estimated
          y = Aop(w_hat) is the action of A in the direction w_hat 
    Y = Aop(Omega) : precomputed action of Aop on Omega, a m x n Array of (presumably) sampled Gaussian or l-percent sparse random vectors (row)
    k :  int, 
        number of eigenvalues/vectors to be estimated, 0 < k < m
    Returns:
    --------
    
    lmbda : ndarray, (k,)           
        eigenvalues arranged in descending order
    Ut : ndarray, (k, n)
        eigenvectors arranged according to eigenvalues, rows are eigenvectors
    
    References:
    -----------
    .. [1] Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp. "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions." SIAM review 53.2 (2011): 217-288.
    .. [2] Algorithm 2 of Arvind paper
    Examples:
    ---------
    >>> import numpy as np
    >>> n = 100
    >>> A = np.diag(0.95**np.arange(n))
    >>> Aop = lambda w_hat: np.dot(A,w_hat)
    >>> k = 10
    >>> p = 5
    >>> Omega = np.random.randn(n, k+p)
    >>> lmbda, Ut = randomized_eigensolver(Aop, Omega, k)
    """
    raise Exception("Need to reimplement this function")
    m, n = Y.shape 
    assert(n >= m >= k) #m = k + p ( p is the oversampling for Omega, to ensure we get a good random projection basis)
    Q, _ = qr(Y.T, mode='economic')
    T =  (Aop(Q.T) @ Q).T #m foward problems , m x m small matrix
    # T = .5*T + .5*T.T

    #Eigen subproblem
    lmbda, V = eigh(T, turbo=True, overwrite_a=True, check_finite=False)
    inds = np.abs(lmbda).argsort()[::-1]
    lmbda = lmbda[inds[0:k]]
    V = V[:, inds[0:k]] #S in the original paper m x m

    #Compute eigenvectors
    Ut = (Q @  V).T 
    return lmbda, Ut
