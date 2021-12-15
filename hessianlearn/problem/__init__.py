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

from .problem import Problem, ClassificationProblem, KerasModelProblem, RegressionProblem, H1RegressionProblem,\
					AutoencoderProblem,VariationalAutoencoderProblem, GenerativeAdversarialNetworkProblem

from .hessian import Hessian, HessianWrapper

from .preconditioner import Preconditioner, IdentityPreconditioner

from .regularization import Regularization, L2Regularization