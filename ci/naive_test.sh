#!/bin/bash

for optimizer in 'incg'
do
	python applications/mnist/mnist_run.py -max_sweeps 1 -optimizer optimizer
done


