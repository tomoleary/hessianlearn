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

from .randomizedEigensolver import low_rank_hessian, randomized_eigensolver, eigensolver_from_range

from .rangeFinders import block_range_finder, noise_aware_adaptive_range_finder

from .varianceBasedNystrom import variance_based_nystrom

from .optimizer import Optimizer, ParametersOptimizer

from .cgSolver import CGSolver, ParametersCGSolver

from .gmresSolver import GMRESSolver, ParametersGMRESSolver

from .minresSolver import MINRESSolver, ParametersMINRESSolver

from .adam import Adam, ParametersAdam

from .gradientDescent import GradientDescent, ParametersGradientDescent

from .inexactNewtonCG import InexactNewtonCG, ParametersInexactNewtonCG

from .inexactNewtonGMRES import InexactNewtonGMRES, ParametersInexactNewtonGMRES

from .inexactNewtonMINRES import InexactNewtonMINRES, ParametersInexactNewtonMINRES

from .lowRankSaddleFreeNewton import LowRankSaddleFreeNewton, ParametersLowRankSaddleFreeNewton
