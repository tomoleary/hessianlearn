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
# Author: Nick Alger

import numpy as np

np.random.seed(0)


def variance_based_nystrom(apply_AA, num_cols_A, oversampling_parameter=5, block_size=10, 
                           std_tol=0.5, max_bad_vectors=5, max_vectors=100, verbose=True):

    op = oversampling_parameter
    n = num_cols_A
    m = len(apply_AA)
    
    Q = np.zeros((n,0))
    Theta = np.zeros((0,0,m))
    num_bad_vectors = 0
    while num_bad_vectors < max_bad_vectors:
        Q1 = Q
        Theta11 = Theta

        Y = get_random_range_vectors(apply_AA, n, block_size)
        Y_perp = Y - np.dot(Q,np.dot(Q.T, Y))
        Q2,_ = np.linalg.qr(Y_perp)
        Q2 = Q2.reshape((n,-1)) # Reshape to guard against case block_size==1
        Q = np.hstack([Q1, Q2])

        Theta = compute_or_update_Theta(Q1, Q2, Theta11, apply_AA)
        dd, U, V = finish_computing_eigenvalue_decomposition(Q, Theta)
        _, all_std = compute_rayleigh_statistics(Theta, V)
        
        bad_inds = (all_std[:-op] / np.abs(dd[:-op])) > std_tol
        num_bad_vectors = np.sum(bad_inds)

        current_num_vectors = Q.shape[1]
        current_rank = current_num_vectors - op - num_bad_vectors
        if verbose:
            print('current_rank=', current_rank, ', num_bad_vectors=', num_bad_vectors)
            
        if current_num_vectors > max_vectors:
            break

    good_inds = np.logical_not(bad_inds)
    dd_good = dd[:-op][good_inds]
    U_good = U[:,:-op][:,good_inds]
    all_std_good = all_std[:-op][good_inds]
    return [dd_good, U_good, all_std_good],[dd[:-op],U[:,:-op],all_std[:-op]]
    

def get_random_range_vectors(apply_AA, num_cols_A, block_size_r,seed = 0):
    """
    Computes n x r matrix
        Y = A * Omega
    where A is an n x n matrix of the form
        A = (A1 + A2 + ... + Am)/m,
    matvecs with the matrices Ak may be computed via the function
        apply_AA[k](x) = Ak * x,
    and Omega is a random n x r matrix.
    """
    n = num_cols_A
    r = block_size_r
    m = len(apply_AA)
    
    Omega = np.random.randn(n, r)
    Y = np.zeros((n, r))
    # In Tensorflow:
    #     z = g^T Omega
    #     q = unstack(z)
    #     Y = (1/m) * restack(dq_i / dw)
    for j in range(r): # These loops can be trivially parallelized
        for k in range(m):
            Y[:,j] = Y[:,j] + (1./m)*apply_AA[k](Omega[:,j])
    return Y
    
    
def compute_Theta(orthonormal_range_basis_Q, apply_AA):
    """
    Computes r x r x m 3-tensor Theta with entries
        Theta_ijk = qi^T Ak qj.
    Theta has frontal slices
        Theta_::k = Q^T Ak Q.
    """
    Q = orthonormal_range_basis_Q
    m = len(apply_AA)
    r = Q.shape[1]
    
    Theta = np.zeros((r, r, m))
    for j in range(r): # These loops can be trivially parallelized
        for k in range(m):
            Theta[:,j,k] = np.dot(Q.T, apply_AA[k](Q[:,j]))
    return Theta
    
    
def finish_computing_eigenvalue_decomposition(orthonormal_range_basis_Q, Theta):
    """
    Finishes computing eigenvalue decomposition
        A = U diag(dd) U^T,
    and smaller auxiliary eigenvalue decomposition
        Q^T A Q = V diag(dd) V^T
    where Q is an orthonormal basis for the range of 
        A = (A1+A2+...+Am)/m, 
    and Theta is the matrix with frontal slices
        Theta_::k = Q^T Ak Q.
    """
    Q = orthonormal_range_basis_Q
    m = Theta.shape[-1]
    
    B = (1. / m) * np.sum(Theta, axis=-1)
    dd, V = np.linalg.eigh(B)
    idx = np.argsort(np.abs(dd))[::-1]
    dd = dd[idx]
    V = V[:,idx]
        
    U = np.dot(Q, V)
    return dd, U, V
    
    
def compute_rayleigh_statistics(Theta, small_eigenvectors_V):
    """
    Computes sample mean and standard deviation of Rayleigh quotients
        all_mu[i] = mean(ui^T Ak ui)
        all_std[i] = std(ui^T Ak ui)
    where Ak is randomly chosen, and ui is the i'th eigenvector of 
        A = (A1 + A2 + ... + Am)/m.
    Theta is the r x r x m 3-tensor with frontal slices
        Theta_::k = Q^T Ak Q,
    for orthonormal basis Q such that
        A =approx= Q * Q^T * A
    The columns, vi, of V are the eigenvectors of the matrix Q^T A Q, i.e.,
        Q^T A Q = V D V^T
    where D is the diagonal matrix of eigenvalues, which we do not need here.
    (Note that ui = Q * vi).
    """
    V = small_eigenvectors_V
    r = Theta.shape[0]
    
    C = np.sum(V.reshape((r,r,-1)) * np.einsum('jki,kl->jli', Theta, V), axis=0)
    all_mu = np.mean(C, axis=1)
    all_std = np.std(C, axis=1)
    return all_mu, all_std
    
    
def update_Theta(Q1, Q2, Theta11, apply_AA):
    """
    Computes updated r x r x m 3-tensor Theta with frontal slices
        Theta_::k = Q^T Ak Q
    based on old Theta1 with frontal slices
        Theta11_::k = Q1^T Ak Q1.
    Here Q1 and Q2 are orthonormal matrices, and
        Q = [Q1, Q2]
    is also an orthonormal matrix. 
    Q1 was the old range approximation for A.
    Q2 columns are more vectors to improve the range approximation.
    Q is the new range approximation.
    """
    m = len(apply_AA)
    r1 = Q1.shape[1]
    r2 = Q2.shape[1]
    r = r1 + r2
    Theta12 = np.zeros((r1, r2, m))
    Theta22 = np.zeros((r2, r2, m))
    for i in range(r2): # These loops can be trivially parallelized
        for k in range(m):
            Ak_qi = apply_AA[k](Q2[:,i])
            Theta12[:,i,k] = np.dot(Q1.T, Ak_qi)
            Theta22[:,i,k] = np.dot(Q2.T, Ak_qi)
            
    Theta = np.zeros((r, r, m))
    Theta[:r1, :r1, :] = Theta11
    Theta[:r1, r1:, :] = Theta12
    Theta[r1:, :r1, :] = Theta12.swapaxes(0,1)
    Theta[r1:, r1:, :] = Theta22
    return Theta
    
    
def compute_or_update_Theta(Q1, Q2, Theta11, apply_AA):
    if Theta11.size == 0:
        return compute_Theta(Q2, apply_AA)
    else:
        return update_Theta(Q1, Q2, Theta11, apply_AA)
        

