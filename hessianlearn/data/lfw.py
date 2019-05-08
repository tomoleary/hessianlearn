from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys


import numpy as np
from scipy import signal
import random
from ..data.data import *

import math
import time

# from statsmodels import robust

def dir_check(dir):
	try:
		os.stat(dir)
	except:
		os.mkdir(dir)

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()



def load_lfw():
	try: 
		# read from file
		images = np.load('lfw_all_images.npy')
		labels = np.load('lfw_all_labels.npy')
		print('Loaded successfully locally')
		return [images, labels]

	except:
		# write to file
		print(80*'#')
		print('Did not load locally.')
		print(80*'#')
		try:
			os.stat("lfw.tgz")
		except:
			print('Downloading from source, and saving to disk.')
			print(80*'#')
			import urllib.request
			urllib.request.urlretrieve("http://vis-www.cs.umass.edu/lfw/lfw.tgz", "lfw.tgz",reporthook)
		folder_name = 'lfw/'
		try:
			os.stat(folder_name)
		except:
			try:
				import subprocess
				subprocess.run(['tar','zxvf',"lfw.tgz"])
			except:
				pass
		import shutil
		folder_names = os.listdir(folder_name)
		n_folders = len(folder_names)
		print(n_folders,' many folder names')
		if not os.path.isdir('lfw_all_images'):
		    os.mkdir('lfw_all_images')
		    print('Making directory lfw_all_images/')
		for folder in os.listdir('lfw/'):
		    for file in os.listdir('lfw/'+folder):
		    	if not os.path.isfile('lfw_all_images/'+file):
		        	shutil.move('lfw/'+folder+'/'+file,'lfw_all_images/')
		        	print('Moving ',file,'to lfw_all_images')

		file_names = os.listdir('lfw_all_images')
		n_files = len(file_names)

		images = np.empty(shape = (n_files,250,250,3))
		
		from keras.preprocessing import image
		for file,counter in zip(file_names,range(n_files)):
			img = image.load_img('lfw_all_images/'+file)
			images[counter,:,:,:] = image.img_to_array(img)
		labels = np.array(file_names)
		assert(labels.shape[0]==images.shape[0])
		print(labels.shape)
		images = np.array(images)
		np.save('lfw_all_images.npy',images)
		np.save('lfw_all_labels.npy',labels)
		print('Saved locally')
		return [images,labels]


	# def view_random_pair(self):
	# 	try:
	# 	    labelkey = ['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
	# 	    i = np.random.choice(range(60000))
	# 	    index = self.all_data[1][i]
	# 	    label = labelkey[index]
	# 	    import matplotlib.pyplot as plt
	# 	    fig, ax = plt.subplots(figsize = (3,3))
	# 	    ax.set_title(str(label))
	# 	    data = self.all_data[0][i,:,:,:].astype(np.uint8)
	# 	    ax.imshow(data)
	# 	    plt.show()
	# 	except:
	# 		pass
	

