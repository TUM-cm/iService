import numpy
from jenks import jenks
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from scipy.signal import argrelextrema
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.metrics import silhouette_score
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering

from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

def goodness_of_variance_fit(data, classes):
    
    def classify(value, breaks):
        for i in range(1, len(breaks)):
            if value < breaks[i]:
                return i
        return len(breaks) - 1
    
    # classification
    classified = numpy.array([classify(i, classes) for i in data])
    # nested list of range indices
    zone_indices = [[idx for idx, val in enumerate(classified) if zone+1 == val] for zone in range(max(classified))]
    # sorted polygon stats
    array_sort = [numpy.array([data[index] for index in zone]) for zone in zone_indices if len(zone) > 0]
    # sum of squared deviations of class means
    sdcm = sum([numpy.sum((classified - classified.mean()) ** 2) for classified in array_sort])
    # sum of squared deviations from array mean
    sdam = numpy.sum((data - data.mean()) ** 2)
    # goodness of variance fit
    return (sdam - sdcm) / sdam

def gaussian_mixture_bic(data, min_clusters, max_clusters):
    bic = list()
    num_cluster = list()
    for n_cluster in range(min_clusters, max_clusters+1):
        model = GaussianMixture(n_cluster).fit(data)
        num_cluster.append(n_cluster)
        bic.append(model.bic(data))
    return num_cluster[numpy.argmin(bic)]

def gaussian_mixture_aic(data, min_clusters, max_clusters):
    aic = list()
    num_cluster = list()
    for n_cluster in range(min_clusters, max_clusters+1):
        model = GaussianMixture(n_cluster).fit(data)
        num_cluster.append(n_cluster)
        aic.append(model.aic(data))
    return num_cluster[numpy.argmin(aic)]

def dbscan(data): # min_samples=2
    cluster_labels = DBSCAN().fit_predict(data)
    unique_clusters = numpy.unique(cluster_labels)
    return len(unique_clusters)

def birch(data):
    cluster_labels = Birch(n_clusters=None).fit_predict(data)
    unique_clusters = numpy.unique(cluster_labels)
    return len(unique_clusters)

def affinity_propagation(data):
    cluster_labels = AffinityPropagation().fit_predict(data)
    unique_clusters = numpy.unique(cluster_labels)
    return len(unique_clusters)

def mean_shift(data):
    if data.ndim > 1:
        return -1
    X = numpy.asarray(list(zip(data, numpy.zeros(len(data)))))
    mean_shift = MeanShift().fit(X)
    labels_unique = numpy.unique(mean_shift.labels_)
    return len(labels_unique)

# local minima in density are be good places to split the data into clusters
def kernel_density_estimation(data, length_density=50):
    # https://stackoverflow.com/questions/35094454/how-would-one-use-kernel-density-estimation-as-a-1d-clustering-method-in-scikit
    if data.ndim > 1:
        return -1
    kde = KernelDensity().fit(data)
    s = numpy.linspace(0, length_density)
    e = kde.score_samples(s.reshape(-1, 1))
    minimum = argrelextrema(e, numpy.less)[0]
    return len(minimum) + 1

# https://github.com/mthh/jenkspy
# https://stats.stackexchange.com/questions/143974/jenks-natural-breaks-in-python-how-to-find-the-optimum-number-of-breaks
def jenks_natural_breaks(data, min_clusters, max_clusters):
    if data.ndim > 1:
        return -1
    num_cluster = list()
    gvf = list()
    for n_cluster in range(min_clusters, max_clusters+1):
        classes = jenks(data, n_cluster)
        if classes:
            num_cluster.append(n_cluster)
            gvf.append(goodness_of_variance_fit(data, classes))
    return num_cluster[numpy.argmax(gvf)]

# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
def kmeans_silhouette(data, min_clusters, max_clusters):
    num_cluster = list()
    silhouette = list()
    for n_cluster in range(min_clusters, max_clusters+1):
        kmeans = KMeans(n_clusters=n_cluster)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        num_cluster.append(n_cluster)
        silhouette.append(silhouette_avg)
    return num_cluster[numpy.argmax(silhouette)]

# k-means + BIC to identify optimal number of clusters
# BIC integrated internally and not accessible from outside
def xmeans_clustering(data, min_clusters, max_clusters):
    initial_centers = kmeans_plusplus_initializer(data, min_clusters).initialize()
    xmeans_instance = xmeans(data, initial_centers, kmax=max_clusters)
    xmeans_instance.process()
    centers = xmeans_instance.get_centers()
    return len(centers)

def agglomerative_hierarchical(data, min_clusters, max_clusters):
    num_cluster = list()
    silhouette = list()
    for n_cluster in range(min_clusters, max_clusters+1):
        hc = AgglomerativeClustering(n_clusters=n_cluster)
        cluster_labels = hc.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        num_cluster.append(n_cluster)
        silhouette.append(silhouette_avg)
    return num_cluster[numpy.argmax(silhouette)]
