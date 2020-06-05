import numpy
from scipy import stats
from sklearn.linear_model import RANSACRegressor

def detect_by_ransac(X, y):
    ransac = RANSACRegressor()
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = numpy.logical_not(inlier_mask)
    return outlier_mask

def detect_by_cusum(data, threshold=0.15):
    min_val = numpy.min(data)
    max_val = numpy.max(data)
    likelihood = min_val + ((max_val - min_val)/2)
    #likelihood = numpy.median(data)
    #likelihood = numpy.mean(data)
    cusum = numpy.empty(data.shape[0])
    cusum_val = 0
    for i in range(data.shape[0]):
        # max: changes only positive direction, min: both directions
        cusum_val = max(0, cusum_val + data[i] - likelihood)
        cusum[i] = cusum_val
    #print(cusum)
    return numpy.where(cusum > threshold)

def detect_by_fft(signal, threshold_freq=.1, frequency_amplitude=.01):
    fft_of_signal = numpy.fft.fft(signal)
    f = numpy.fft.fftfreq(len(fft_of_signal), 0.001)
    outlier = numpy.max(signal) if abs(numpy.max(signal)) > abs(numpy.min(signal)) else numpy.min(signal)
    if numpy.any(numpy.abs(fft_of_signal[f>threshold_freq]) > frequency_amplitude):
        return numpy.where(signal == outlier)
    
def detect_by_median(signal, threshold=3):
    difference = numpy.abs(signal - numpy.median(signal))
    median_difference = numpy.median(difference)
    median_ratio = 0 if median_difference == 0 else difference / float(median_difference)
    return numpy.where(median_ratio > threshold)

# http://colingorrie.github.io/outlier-detection.html
def detect_by_modified_z_score(signal, threshold=3):
    median_y = numpy.median(signal)
    median_absolute_deviation_y = numpy.median([numpy.abs(y - median_y) for y in signal])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y for y in signal]
    return numpy.where(numpy.abs(modified_z_scores) > threshold)

def detect_by_z_score(ys, threshold=3):
    mean_y = numpy.mean(ys)
    stdev_y = numpy.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return numpy.where(numpy.abs(z_scores) > threshold)

def detect_by_z_score_scipy(signal, threshold=3):
    return numpy.where(stats.zscore(signal) > threshold)

def detect_by_iqr(signal, bottom=25, top=75, threshold=1.5):
    quartile_1, quartile_3 = numpy.percentile(signal, [bottom, top])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * threshold)
    upper_bound = quartile_3 + (iqr * threshold)
    return numpy.where((signal > upper_bound) | (signal < lower_bound))

def ransac_example():
    import numpy as np
    from matplotlib import pyplot as plt
    from sklearn import linear_model, datasets
    n_samples = 1000
    n_outliers = 50
    X, y, _ = datasets.make_regression(
        n_samples=n_samples, n_features=1, n_informative=1, noise=10, coef=True, random_state=0)
    # Add outlier data
    np.random.seed(0)
    X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
    y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)
    
    # Robustly fit linear model with RANSAC algorithm
    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    
    # Predict data of estimated models
    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)
    plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.', label='Inliers')
    plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.', label='Outliers')
    plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=2, label='RANSAC regressor')
    plt.legend(loc='lower right')
    plt.xlabel("Input")
    plt.ylabel("Response")
    plt.show()

def main():
    ransac_example()
    
if __name__ == "__main__":
    main()
