#!/bin/bash

optimizers = ['incg']
for optimizer in optimizers:
	python applications/mnist/mnist_run.py -max_sweeps 1 -optimizer 'incg'



