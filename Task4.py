# CC214530 Dominique Kidd Applied AI Coursework Task 4
# Code herein taken from FaceRec.py on NOW, and the k-means clustering based 
# example from http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
print(__doc__)

from time import time
import numpy as np


from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_lfw_people
from sklearn import preprocessing
from skimage.feature import canny


np.random.seed(42)


faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)


# Dee here. We want to perform canny edge detection on the faces. We use the 
# flattened images from the data array of LFW to perforn the klustering but 
# canny requires the data in 2d. So the simplest thing to do is;
# for each data in data reshape to 2d. 
# pass through canny
# reflatten resultant to 1d
# our data is now cannyfied
# pass through kmeans and see what the results are

# we need to know the height and width of the face images so we grab that here
n_samples, h, w = faces.images.shape
idx=0

# used in debugging to compare the non cannied and cannied dataset
#dataprecanny = preprocessing.normalize(faces.data, norm='max', axis=1, copy=True, return_norm=False)

# we enter a for loop and work through all 1288 (in this case) entries in data
# for the faces data we have extracted from LFW. We reshape each one, pass
# it to canny and then reflatten the resultant back into data
for data in faces.data:
    #grab the 1d data and make it 2d
    Img1D = np.copy(faces.data[idx])
    Img2D = np.reshape(Img1D, (h, w))
    #dp the canny edge detection on our face image
    cannyfiedImg = canny(Img2D, sigma=0.1, low_threshold=10, high_threshold=90)
    #reflatten the canny image so it can go back into for 1d form of data
    reFlat = np.ravel(cannyfiedImg)
    faces.data[idx] = reFlat[:]
    idx = idx + 1

# now we have cannyfied faces we can carry on with the k-means clustering as normal. 
data = preprocessing.normalize(faces.data, norm='max', axis=1, copy=True, return_norm=False)


n_samples, n_features = data.shape
n_faces = len(np.unique(faces.target))
labels = faces.target

sample_size = 300

print("\n", 48 * '_ ', "\n")
print("n_faces: %d, \t n_samples %d, \t n_features %d"
      % (n_faces, n_samples, n_features))


print(95 * '_')
print('% 9s' % '    init'
      '       time     inertia        homo    compl   v-meas   ARI     AMI      NMI    silhouette')

# -------------------------------------------------------
def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)

    print("\n", '% 9s     %.2fs    %.3f      %.3f   %.3f   %.3f   %.3f   %.3f    %.3f   %.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.normalized_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))
# -------------------------------------------------------
#setting n_init higher only takes longer, doesn't improve, same with max iter
bench_k_means(KMeans(init='k-means++', n_clusters=n_faces, n_init=10, max_iter=300),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_faces, n_init=10, max_iter=300),
              name="random", data=data)


print(95 * '_')
