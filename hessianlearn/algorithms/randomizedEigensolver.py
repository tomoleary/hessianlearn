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
    
    w : ndarray, (k,)           
        eigenvalues arranged in descending order
    u : ndarray, (n,k)
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
    if p is None:
        p = int(0.01*k)
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

def randomized_single_pass_generalized_eigensolver(Aop, Bop, Binvop, Omega, k): 
    """
    The single pass algorithm for the GHEP as presented in [2].

    Randomized algorithm for Generalized Hermitian eigenvalue problems, i.e. 
    solving Au = lambda Bu
    Returns k largest eigenvalues computed using the randomized algorithm
    
    
    Parameters:
    -----------
    Aop : {Callable} tn x n
          Hermitian matrix operator whose eigenvalues need to be estimated
          y = Aop(w_hat) is the action of A in the direction w_hat 
          
    B  : {Callable} n x n
          Hermetian RHS matrix operator. The eigenvectors u should live
          in the range space of Aop
    Binv: {Callable} n x n
          Inverse of B 
    n : int,
           number of row/columns of the operator A
        
    k :  int, 
        number of eigenvalues/vectors to be estimated
    p :  int, optional
        oversampling parameter which can improve accuracy of resulting solution
        Default: 20
            
    Returns:
    --------
    
    w : ndarray, (k,)
        eigenvalues arranged in descending order
    u : ndarray, (n,k)
        eigenvectors arranged according to eigenvalues such that U^T B U = I_k
    
    References:
    -----------
    [2] Arvind K. Saibaba, Jonghyun Lee, Peter K. Kitanidis, Randomized algorithms for Generalized Hermitian Eigenvalue Problems with application to computing Karhunen-Loeve expansion, Numerical Linear Algebra with Applications, 2010
    Examples:
    ---------
    >>> import numpy as np
    >>> n = 100
    >>> A = np.diag(0.95**np.arange(n))
    >>> Aop = lambda w_hat: np.dot(A,w_hat)
    >>> k = 10
    >>> p = 5
    >>> lmbda, U = randomized_single_pass_generalized_eigensolver(Aop, n, k, p)
    """
    raise Exception("Need to reimplement this function")
    n, m  = Omega.shape
    
    assert(n >= m >= k )
    
    Ybar = Aop(Omega)
        
    Y = Binvop(Ybar)

    # CholQR with W-inner products
    Z,_ = np.linalg.qr(Y)
    BZ = Bop(Z)
        
    R = np.linalg.cholesky( np.dot(Z.T,BZ )) 
    Q = np.linalg.solve(R, Z.T).T #Q = Y*R^-1
    BQ = np.linalg.solve(R, BZ.T).T #WQ = Z * R^-1
    
        
    Xt = np.dot(Omega.T, BQ)
    Wt = np.dot(Ybar.T, Q)
    Tt = np.linalg.solve(Xt,Wt)
                
    T = .5*Tt + .5*Tt.T
        
    d, V = np.linalg.eigh(T)
    
    d_abs = np.abs(d) #sort by absolute value (we want the k largest eigenvalues regardless of sign)
    sort_perm = d_abs.argsort()
        
    sort_perm = sort_perm[::-1]
    
    d = d[sort_perm[0:k]]
    V = V[:, sort_perm[0:k]] 
        
    U = Q @ V
    return d, U


def randomized_double_pass_generalized_eigensolver(Aop, Bop, Binvop, Y, k): 
    """
    The double pass algorithm for the GHEP as presented in [2].

    Randomized algorithm for Generalized Hermitian eigenvalue problems, i.e. 
    solving Au = lambda Bu
    Returns k largest eigenvalues computed using the randomized algorithm
    
    #Algorithm 8 in the Arvind paper
    Parameters:
    -----------
    - :code:`Aop`: the operator for which we need to estimate the dominant generalized eigenpairs.
    - :code:`Bop`: the right-hand side operator.
    - :code:`Binvop`: the inverse of the right-hand side operator.
    - :code:`Omega`: a random gassian matrix with :math:`m \\geq k` columns.
    - :code:`k`: the number of eigenpairs to extract.
    - :code:`s`: the number of power iterations for selecting the subspace.
    
    Aop : {Callable} n x n
          Hermitian matrix operator whose eigenvalues need to be estimated
          y = Aop(w_hat) is the action of A in the direction w_hat 
    Y = AOmega : m x n Array of (presumably) sampled Gaussian or l-percent sparse random vectors (rows)
    B  : {Callable} n x n
          Hermetian RHS matrix operator. The eigenvectors u should live
          in the range space of Aop
    Binv : {Callable} n x n
          Hermetian inverse of RHS matrix operator.
    k :  int, 
        number of eigenvalues/vectors to be estimate
           
    Returns:
    --------
    
    - :code:`lmbda`: the estimate of the :math:`k` dominant eigenvalues of :math:`A`.
    - :code:`U`: the estimate of the :math:`k` dominant eigenvectors of :math:`A,\\, U^T B U = I_k`.
    
    Examples:
    ---------
    >>> import numpy as np
    >>> n = 100
    >>> A = np.diag(0.95**np.arange(n))
    >>> B = np.diag(0.25**np.arange(n))
    >>> Binv = np.diag(1/0.25**np.arange(n))
    >>> Aop = lambda w_hat: np.dot(A,w_hat)
    >>> Bop = lambda w_hat: np.dot(B,w_hat)
    >>> Binvop = lambda w_hat: np.dot(Binv,w_hat)
    >>> Omega = np.random.normal((n,n))
    >>> k = 10
    >>> p = 5
    >>> lmbda, U = randomized_double_pass_generalized_eigensolver(Aop, Bop, Binvop, Omega, k)
    """
    m, n  = Y.shape # m = k+p < N
    assert(n >= m >= k )

    #ALGO 4:
    # CholQR with B-inner products, Y = QR, Q.T B Q = I
    Zt, _ = qr(Y.T, mode='economic') #Y = QR, Z = Q
    BZ = Bop(Zt.T).T
    R = cholesky( np.dot(Zt.T, BZ), lower=True) #R = chol(Y.T B Y)
    Q = solve_triangular(R, Zt.T,lower=True) #Q = Y*R^-1 
    # T = np.zeros((m,m))
    T =  (Aop(Q)@ Q.T) #m x m small matrix, #m forward problems
    # T = .5*T + .5*T.T
    #Eigen subproblem
    lmbda, V = eigh(T, turbo=True, overwrite_a=True,check_finite=False)
    inds = np.abs(lmbda).argsort()[::-1]
    lmbda = lmbda[inds[0:k]]
    V = V[:, inds[0:k]] #S in the original paper m x m
    #Compute eigenvectors
    U = V.T @  Q
    return lmbda, U