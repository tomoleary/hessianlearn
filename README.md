<!-- # hessianlearn -->



	      ___          ___          ___          ___                     ___          ___     
	     /__/\        /  /\        /  /\        /  /\       ___         /  /\        /__/\    
	     \  \:\      /  /:/_      /  /:/_      /  /:/_     /  /\       /  /::\       \  \:\   
	      \__\:\    /  /:/ /\    /  /:/ /\    /  /:/ /\   /  /:/      /  /:/\:\       \  \:\  
	  ___ /  /::\  /  /:/ /:/_  /  /:/ /::\  /  /:/ /::\ /__/::\     /  /:/~/::\  _____\__\:\ 
	 /__/\  /:/\:\/__/:/ /:/ /\/__/:/ /:/\:\/__/:/ /:/\:\\__\/\:\__ /__/:/ /:/\:\/__/::::::::\
	 \  \:\/:/__\/\  \:\/:/ /:/\  \:\/:/~/:/\  \:\/:/~/:/   \  \:\/\\  \:\/:/__\/\  \:\~~\~~\/
	  \  \::/      \  \::/ /:/  \  \::/ /:/  \  \::/ /:/     \__\::/ \  \::/      \  \:\  ~~~ 
	   \  \:\       \  \:\/:/    \__\/ /:/    \__\/ /:/      /__/:/   \  \:\       \  \:\     
	    \  \:\       \  \::/       /__/:/       /__/:/       \__\/     \  \:\       \  \:\    
	     \__\/        \__\/        \__\/        \__\/                   \__\/        \__\/    


			                   ___          ___          ___          ___     
			                  /  /\        /  /\        /  /\        /__/\    
			                 /  /:/_      /  /::\      /  /::\       \  \:\   
			  ___     ___   /  /:/ /\    /  /:/\:\    /  /:/\:\       \  \:\  
			 /__/\   /  /\ /  /:/ /:/_  /  /:/~/::\  /  /:/~/:/   _____\__\:\ 
			 \  \:\ /  /://__/:/ /:/ /\/__/:/ /:/\:\/__/:/ /:/___/__/::::::::\
			  \  \:\  /:/ \  \:\/:/ /:/\  \:\/:/__\/\  \:\/:::::/\  \:\~~\~~\/
			   \  \:\/:/   \  \::/ /:/  \  \::/      \  \::/~~~~  \  \:\  ~~~ 
			    \  \::/     \  \:\/:/    \  \:\       \  \:\       \  \:\     
			     \__\/       \  \::/      \  \:\       \  \:\       \  \:\    
			                  \__\/        \__\/        \__\/        \__\/    





[![Build Status](https://travis-ci.com/tomoleary/hessianlearn.svg?branch=master)](https://travis-ci.com/tomoleary/hessianlearn)
[![DOI](https://zenodo.org/badge/184635062.svg)](https://zenodo.org/badge/latestdoi/184635062)
[![License](https://img.shields.io/github/license/tomoleary/hessianlearn)](./LICENSE.md)
[![Top language](https://img.shields.io/github/languages/top/tomoleary/hessianlearn)](https://www.python.org)
![Code size](https://img.shields.io/github/languages/code-size/tomoleary/hessianlearn)
[![Issues](https://img.shields.io/github/issues/tomoleary/hessianlearn)](https://github.com/tomoleary/hessianlearn/issues)
[![Latest commit](https://img.shields.io/github/last-commit/tomoleary/hessianlearn)](https://github.com/tomoleary/hessianlearn/commits/master)

# Hessian-based stochastic optimization in TensorFlow and keras

This code implements Hessian-based stochastic optimization in TensorFlow and keras by exposing the matrix-free Hessian to users. The code is meant to allow for rapid-prototyping of Hessian-based algorithms via the matrix-free Hessian action, which allows users to inspect Hessian based information for stochastic nonconvex (neural network training) optimization problems. 

The Hessian action is exposed via matrix-vector products:
<p align="center">
	<img src="https://latex.codecogs.com/gif.latex?H\widehat{w}=\frac{d}{dw}(g^T\widehat{w})" /> 
</p>

and matrix-matrix products:
<p align="center">
	<img src="https://latex.codecogs.com/gif.latex?H\widehat{W}=\frac{d}{dw}(g^T\widehat{W})" /> 
</p>

## Compatibility

The code is compatible with Tensorflow v1 and v2, but certain features of v2 are disabled (like eager execution). This is because the Hessian matrix products in hessianlearn are implemented using `placeholders` which have been deprecated in v2. For this reason hessianlearn cannot work with data generators and things like this that require eager execution. If any compatibility issues are found, please open an [issue](https://github.com/tomoleary/hessianlearn/issues).

## Usage
Set `HESSIANLEARN_PATH` environmental variable

Train a keras model

```python
import os,sys
import tensorflow as tf
sys.path.append( os.environ.get('HESSIANLEARN_PATH'))
from hessianlearn import *

# Define keras neural network model
neural_network = tf.keras.models.Model(...)
# Define loss function and compile model
neural_network.compile(loss = ...)

```

hessianlearn implements various training [`problem`](https://github.com/tomoleary/hessianlearn/blob/master/hessianlearn/problem/problem.py) constructs (regression, classification, autoencoders, variational autoencoders, generative adversarial networks). Instantiate a `problem`, a `data` object (which takes a dictionary with keys that correspond to the corresponding `placeholders` in `problem`) and `regularization`

```python
# Instantiate the problem (this handles the loss function,
# construction of hessian and gradient etc.)
# KerasModelProblem extracts loss function and metrics from
# a compiled keras model
problem = KerasModelProblem(neural_network)
# Instantiate the data object, this handles the train / validation split
# as well as iterating during training
data = Data({problem.x:x_data,problem.y_true:y_data},train_batch_size,\
	validation_data_size = validation_data_size)
# Instantiate the regularization: L2Regularization is Tikhonov,
# gamma = 0 is no regularization
regularization = L2Regularization(problem,gamma = 0)
```

Pass these objects into the `HessianlearnModel` which handles the training

```python
HLModel = HessianlearnModel(problem,regularization,data)
HLModel.fit()
```

### Alternative Usage (More like Keras Interface)
The example above was the original way the optimizer interface was implemented in hessianlearn, however to better mimic the keras interface and allow for more end-user rapid prototyping of the optimizer that is used to fit data, as of December 2021, the following way has been created

```python
import os,sys
import tensorflow as tf
sys.path.append( os.environ.get('HESSIANLEARN_PATH'))
from hessianlearn import *

# Define keras neural network model
neural_network = tf.keras.models.Model(...)
# Define loss function and compile model
neural_network.compile(loss = ...)
# Instance keras model wrapper which deals with the 
# construction of the `problem` which handles the construction
# of Hessian computational graph and variables
HLModel = KerasModelWrapper(neural_network)
# Then the end user can pass in an optimizer 
# (e.g. custom end-user optimizer)
optimizer = LowRankSaddleFreeNewton # The class constructor, not an instance
opt_parameters = LowRankSaddleFreeNewtonParameters()
opt_parameters['hessian_low_rank'] = 40
HLModel.set_optimizer(optimizer,optimizer_parameters = opt_parameters)
# The data object still needs to key on to the specific computational
# graph variables that data will be passed in for.
# Note that data can naturally handle multiple input and output data,
# in which case problem.x, problem.y_true are lists corresponding to
# neural_network.inputs, neural_network.outputs
problem = HLModel.problem
data = Data({problem.x:x_data,problem.y_true:y_data},train_batch_size,\
	validation_data_size = validation_data_size)
# And finally one can call fit!
HLModel.fit(data)
```

## Examples

[Tutorial 0: MNIST Autoencoder](https://github.com/tomoleary/hessianlearn/blob/master/tutorial/Tutorial%200%20MNIST%20Autoencoder.ipynb)


# References

These manuscripts motivate and use the hessianlearn library for stochastic nonconvex optimization

- \[1\] O'Leary-Roseberry, T., Alger, N., Ghattas O.,
[**Inexact Newton Methods for Stochastic Nonconvex Optimization with Applications to Neural Network Training**](https://arxiv.org/abs/1905.06738).
arXiv:1905.06738.
([Download](https://arxiv.org/pdf/1905.06738.pdf))<details><summary>BibTeX</summary><pre>
@article{OLearyRoseberryAlgerGhattas2019,
  title={Inexact Newton methods for stochastic nonconvex optimization with applications to neural network training},
  author={O'Leary-Roseberry, Thomas and Alger, Nick and Ghattas, Omar},
  journal={arXiv preprint arXiv:1905.06738},
  year={2019}
}
}</pre></details>

- \[2\] O'Leary-Roseberry, T., Alger, N., Ghattas O.,
[**Low Rank Saddle Free Newton: A Scalable Method for Stochastic Nonconvex Optimization**](https://arxiv.org/abs/2002.02881).
arXiv:2002.02881.
([Download](https://arxiv.org/pdf/2002.02881.pdf))<details><summary>BibTeX</summary><pre>
@article{OLearyRoseberryAlgerGhattas2020,
  title={Low Rank Saddle Free Newton: Algorithm and Analysis},
  author={O'Leary-Roseberry, Thomas and Alger, Nick and Ghattas, Omar},
  journal={arXiv preprint arXiv:2002.02881},
  year={2020}
}
}</pre></details>


- \[3\] O'Leary-Roseberry, T., Villa, U., Chen P., Ghattas O.,
[**Derivative-Informed Projected Neural Networks for High-Dimensional Parametric Maps Governed by PDEs**](https://www.sciencedirect.com/science/article/pii/S0045782521005302).
Computer Methods in Applied Mechanics and Engineering. Volume 388, 1 January 2022, 114199.
([Download](https://arxiv.org/pdf/2011.15110.pdf))<details><summary>BibTeX</summary><pre>
@article{OLearyRoseberryVillaChenEtAl2022,
  title={Derivative-informed projected neural networks for high-dimensional parametric maps governed by {PDE}s},
  author={O’Leary-Roseberry, Thomas and Villa, Umberto and Chen, Peng and Ghattas, Omar},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={388},
  pages={114199},
  year={2022},
  publisher={Elsevier}
}
}</pre></details>


- \[4\] Thomas O'Leary-Roseberry, Xiaosong Du, Anirban Chaudhuri, Joaquim R. R. A. Martins, Karen Willcox, Omar Ghattas,
[**Adaptive Projected Residual Networks for Learning Parametric Maps from Sparse Data**](https://arxiv.org/abs/2112.07096).
arXiv:2112.07096.
([Download](https://arxiv.org/pdf/2112.07096.pdf))<details><summary>BibTeX</summary><pre>
@article{OLearyRoseberryDuChaudhuriEtAl2021,
  title={Adaptive Projected Residual Networks for Learning Parametric Maps from Sparse Data},
  author={O'Leary-Roseberry, Thomas and Du, Xiaosong, and Chaudhuri, Anirban, and Martins Joaqium R. R. A., and Willcox, Karen, and Ghattas, Omar},
  journal={arXiv preprint arXiv:2112.07096},
  year={2021}
}
}</pre></details>





