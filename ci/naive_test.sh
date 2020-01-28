#!/bin/bash

for optimizer in 'incg' 'lrsfn' 'ingmres' 'gd'
do
	python applications/mnist/mnist_run.py -max_sweeps 1 -optimizer $optimizer
done


