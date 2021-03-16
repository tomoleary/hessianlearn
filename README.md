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
[![License](https://https://img.shields.io/github/license/tomoleary/hessianlearn)](./LICENSE.md)
[![Top language](https://img.shields.io/github/languages/top/tomoleary/hessianlearn)](https://www.python.org)
![Code size](https://img.shields.io/github/languages/code-size/tomoleary/hessianlearn)
[![Issues](https://img.shields.io/github/issues/tomoleary/hessianlearn)](https://github.com/tomoleary/hessianlearn/issues)
[![Latest commit](https://img.shields.io/github/last-commit/tomoleary/hessianlearn)](https://github.com/tomoleary/hessianlearn/commits/master)

# Hessian based stochastic optimization in TensorFlow and keras

This code implements Hessian-based stochastic optimization in TensorFlow and keras by exposing the matrix-free Hessian to users. The code is meant to allow for rapid-prototyping of Hessian-based algorithms via the matrix-free Hessian action, which allows users to inspect Hessian based information for stochastic nonconvex (neural network training) optimization problems. 

The Hessian action is exposed via matrix-vector or matrix-matrix products:

<img src="https://latex.codecogs.com/gif.latex?H\widehat{w}=\frac{d}{dw}(g^T\widehat{w})" /> 

## Compatibility

The code is compatible with Tensorflow v1 and v2, but certain features of v2 are disabled (like eager execution). This is because the Hessian matrix products in hessianlearn are implemented using `placeholders` which have been deprecated in v2. For this reason hessianlearn cannot work with data generators and things like this that require eager execution. If any compatibility issues are found, please open an [issue](https://github.com/tomoleary/hessianlearn/issues).

## Usage

Blah blah blah 


# References

These publications motivate and use the hessianlearn library for stochastic nonconvex optimization

- \[1\] O'Leary-Roseberry, T., Alger, N., Ghattas O.,
[**Inexact Newton Methods for Stochastic Nonconvex Optimization with Applications to Neural Network Training**](https://arxiv.org/abs/1905.06738).
arXiv:1905.06738.
([Download](https://arxiv.org/pdf/1905.06738.pdf))<details><summary>BibTeX</summary><pre>
@article{o2019inexact,
  title={Inexact Newton methods for stochastic nonconvex optimization with applications to neural network training},
  author={O'Leary-Roseberry, Thomas and Alger, Nick and Ghattas, Omar},
  journal={arXiv preprint arXiv:1905.06738},
  year={2019}
}
}</pre></details>

- \[2\] O'Leary-Roseberry, T., Alger, N., Ghattas O.,
[**Low Rank Saddle Free Newton**](https://arxiv.org/abs/2002.02881).
arXiv:2002.02881.
([Download](https://arxiv.org/pdf/2002.02881.pdf))<details><summary>BibTeX</summary><pre>
@article{o2020low,
  title={Low Rank Saddle Free Newton: Algorithm and Analysis},
  author={O'Leary-Roseberry, Thomas and Alger, Nick and Ghattas, Omar},
  journal={arXiv preprint arXiv:2002.02881},
  year={2020}
}
}</pre></details>


- \[3\] O'Leary-Roseberry, T., Villa, U., Chen P., Ghattas O.,
[**Derivative-Informed Projected Neural Networks for High-Dimensional Parametric Maps Governed by PDEs**](https://arxiv.org/abs/2011.15110).
arXiv:2011.15110.
([Download](https://arxiv.org/pdf/2011.15110.pdf))<details><summary>BibTeX</summary><pre>
@article{o2020derivative,
  title={Derivative-Informed Projected Neural Networks for High-Dimensional Parametric Maps Governed by PDEs},
  author={O'Leary-Roseberry, Thomas and Villa, Umberto and Chen, Peng and Ghattas, Omar},
  journal={arXiv preprint arXiv:2011.15110},
  year={2020}
}
}</pre></details>




