# CC214530 Dominique Kidd Applied AI Coursework Task 3
# Code herein taken from FaceRec.py on NOW, and the tutorial at
# http://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html
# -*- coding: utf-8 -*-
from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt
import scipy

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from scipy.stats import sem
from skimage import data
from skimage.color import rgb2gray, gray2rgb
import numpy as np
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

    
count = 0

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


###############################################################################
# Download the data, if not already on disk and load it as numpy arrays

lfw_people = fetch_lfw_people(min_faces_per_person=70, 
                               slice_ = (slice(50,140),slice(61,189)),
                             resize=1.5)

faces = lfw_people.images

#for all in lfw_people 
for face in faces:
 
    # Dee the below is just a quick fix, just do 10, not the whole lot for 
    # because that produced a lot of images to count. So for simplicity this
    # version stops after 10 images. 
    if count < 10: 
       
        image = face
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))
        ax1.imshow(image, cmap=plt.cm.gray)
        #edge detection
        #dee version
        edges = canny(image, sigma=0.1, low_threshold=10, high_threshold=90)
        # Detect two radii
        #edges = canny(image, sigma=1)
        ax2.imshow(edges, cmap=plt.cm.gray)
        #circle detection
        hough_radii = np.arange(5, 30, 2)
        hough_res = hough_circle(edges, hough_radii)
        centers = []
        accums = []
        radii = []
        for radius, h in zip(hough_radii, hough_res):
            num_peaks = 5
            peaks = peak_local_max(h, num_peaks=num_peaks)
            centers.extend(peaks)
            accums.extend(h[peaks[:, 0], peaks[:, 1]])
            radii.extend([radius] * num_peaks)
        # Draw the most prominent 3 circles
        for idx in np.argsort(accums)[::-1][:3]:
            print(centers[idx], radii[idx], accums[idx])
            center_x, center_y = centers[idx]
            radius = radii[idx]
            cx, cy = circle_perimeter(center_y, center_x, radius, shape=image.shape)            
            #image[cy, cx] = (220, 20, 20) 
            #dealing with greyscale so just draw white circles, not colour. 
            image[cy, cx] = (220)
        ax3.imshow(image, cmap=plt.cm.gray)
        plt.show()

        count+=1
#==============================================================================

