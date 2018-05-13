import glob
from scipy.misc import imread 
from scipy.misc import imsave
from scipy.misc import toimage
from scipy.misc import imresize
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import os
datasetA ='/home/labuser/Documents/CycleGAN/grey/*.jpg'
datasetB  = '/home/labuser/Documents/CycleGAN/nir/*.jpg'
destination = 'datasets/NIRtoVIS/'
dataA = glob.glob(datasetA)
dataA = sorted(dataA)
dataB = glob.glob(datasetB)
dataB = sorted(dataB)
count = 0
for i,j in zip(dataA,dataB):
	count+=1
	img_A = imread(j)
	img_B = imread(i)
	line = img_A
	out = np.zeros((np.shape(img_B)[0],np.shape(img_B)[1]*2,3))
	out[:,0:np.shape(img_B)[1],0] = img_B
	out[:,0:np.shape(img_B)[1],1] = img_B
	out[:,0:np.shape(img_B)[1],2] = img_B
	out[:,np.shape(img_B)[1]:2*np.shape(img_B)[1],0] = line
	out[:,np.shape(img_B)[1]:2*np.shape(img_B)[1],1] = line
	out[:,np.shape(img_B)[1]:2*np.shape(img_B)[1],2] = line
	toimage(out, cmin=0, cmax=255).save(destination+str(count)+'.jpg')

