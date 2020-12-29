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
# Authors: Nick Alger, Tom O'Leary-Roseberry
# Contact: tom.olearyroseberry@utexas.edu
from __future__ import absolute_import, division, print_function

import unittest 
import numpy as np
import sys

sys.path.append('../../')
from hessianlearn import (variance_based_nystrom)
sys.path.append('../algorithms')
from varianceBasedNystrom import *

def make_random_symmetric_matrix(n,p):
	U, _ = np.linalg.qr(np.random.randn(n,n))
	ss = np.random.randn(n)**p
	A = np.dot(U, np.dot(np.diag(ss), U.T))
	return A


def compute_Theta_slow(Q, apply_AA):
    r = Q.shape[1]
    m = len(apply_AA)
    Theta_true = np.zeros((r, r, m))
    for i in range(r):
        for j in range(r):
            for k in range(m):
                Theta_true[i,j,k] = np.dot(Q[:,i], apply_AA[k](Q[:,j]))
    return Theta_true

def compute_rayleigh_statistics_slow(U, apply_AA):
    m = len(apply_AA)
    r = U.shape[1]
    C = np.zeros((r, m))
    for k in range(m):
        for i in range(r):
            C[i,k] = np.dot(U[:,i], apply_AA[k](U[:,i]))

    all_mu = np.mean(C, axis=1)
    all_std = np.std(C, axis=1)
    return all_mu, all_std


class TestVarianceBasedNystrom(unittest.TestCase):

	def setUp(self):
		self.n = 500
		m = 50
		p = 7
		self.batch_r = 10
		randomness_factor = 0.1

		A0 = make_random_symmetric_matrix(self.n,p)
		AA = [A0 + randomness_factor * make_random_symmetric_matrix(self.n,p) for _ in range(m)]

		self.apply_AA = [lambda x, Ak=Ak: np.dot(Ak,x) for Ak in AA]

		self.A = np.sum(AA, axis=0)/m

		



	def test_all(self):
		Y = get_random_range_vectors(self.apply_AA, self.n, self.batch_r)
		Q,_ = np.linalg.qr(Y)
		Theta = compute_Theta(Q, self.apply_AA)
		Theta_true = compute_Theta_slow(Q, self.apply_AA)
		err_Theta = np.linalg.norm(Theta - Theta_true)/np.linalg.norm(Theta_true)
		print('err_Theta=', err_Theta)
		assert err_Theta < 1e-10

		dd, U, V = finish_computing_eigenvalue_decomposition(Q, Theta)

		A_approx = np.dot(U, np.dot(np.diag(dd), U.T))
		err_A_1 = np.linalg.norm(self.A - A_approx)/np.linalg.norm(self.A)
		print('err_A_1=', err_A_1)
		assert err_A_1 < 1.0

		# Errors in computing statistics
		all_mu, all_std = compute_rayleigh_statistics(Theta, V)

		all_mu_true, all_std_true = compute_rayleigh_statistics_slow(U,self.apply_AA)

		err_mu = np.linalg.norm(all_mu - all_mu_true)/np.linalg.norm(all_mu_true)
		err_std = np.linalg.norm(all_std - all_std_true)/np.linalg.norm(all_std_true)

		print('err_mu=', err_mu)
		print('err_std=', err_std)
		assert err_mu < 1e-10
		assert err_std < 1e-10

		# Redo computations with better range approximation
		Y2 = get_random_range_vectors(self.apply_AA, self.n, self.batch_r)
		Y2_perp = Y2 - np.dot(Q,np.dot(Q.T, Y2))
		Q2,_ = np.linalg.qr(Y2_perp)
		Q_new = np.hstack([Q, Q2])
		err_Q_orth = np.linalg.norm(np.dot(Q_new.T, Q_new) - np.eye(Q_new.shape[1]))
		print('err_Q_orth=', err_Q_orth)
		assert err_Q_orth < 1e-10
		
		Theta_new = update_Theta(Q, Q2, Theta, self.apply_AA)

		Theta_true_new = compute_Theta_slow(Q_new, self.apply_AA)

		err_Theta_new = np.linalg.norm(Theta_new - Theta_true_new)/np.linalg.norm(Theta_true_new)
		print('err_Theta_new=', err_Theta_new)

		assert err_Theta_new < 1e-10

		dd_new, U_new, V_new = finish_computing_eigenvalue_decomposition(Q_new, Theta_new)
		A_approx_new = np.dot(U_new, np.dot(np.diag(dd_new), U_new.T))
		err_A_new = np.linalg.norm(self.A - A_approx_new)/np.linalg.norm(self.A)
		print('err_A_new=', err_A_new)

		# The approximation error should decrease monotonically as we increase the range
		assert err_A_new < err_A_1

		# Run the complete method from scratch

		[dd_good, U_good, all_std_good], [dd_all,U_all,all_std] = variance_based_nystrom(self.apply_AA, self.n)

		A_good_approx = np.dot(U_good, np.dot(np.diag(dd_good), U_good.T))
		err_A_good = np.linalg.norm(A_good_approx - self.A)/np.linalg.norm(self.A)
		print('err_A_good=', err_A_good)
		assert err_A_good < 0.1



if __name__ == '__main__':
    unittest.main()