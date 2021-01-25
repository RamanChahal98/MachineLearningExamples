import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

# load data
digits = load_digits()
# scale all features down so they are in the range -1 and 1
# to save down on computation because digits in this dataset are very large (rbg numbers)
data = scale(digits.data)
# labels
y = digits.target

# set amount of clusters to look for (amount of centroids to make)
k = len(np.unique(y))
# get amount of instances  (amount of numbers we are going to classify and amount of features that go along with that data)
samples, features = data.shape

# scoring function from sklearn
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

# training the model
# create K-Means classifier
clf = KMeans(n_clusters=k, init="random", n_init=10)
# call function to score and train
bench_k_means(clf, "1", data)