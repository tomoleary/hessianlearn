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



# Transfer Learning

* Examples of CIFAR10, CIFAR100 classification from pre-trained Imagenet ResNet50 model in `transfer_learning/`

* Pre-trained model serves as well conditioned initial guess for transfer learning. In this setting Newton methods perform well due to their excellent properties in local convergence. Low Rank Saddle Free Newton is able to zero in on highly generalizable local minimizers bypassing indefinite regions. Below are validation accuracies of best choices of fixed step-length for Adam, SGD and LRSFN with fixed rank of 40.

<p align="center">
	<img src="https://github.com/tomoleary/images/blob/main/hessianlearn/cifar100transfer.png" width="75%" /> 
</p>

* For more information see the following manuscript

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





