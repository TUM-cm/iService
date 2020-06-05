import numpy

def is_outlier_zscore(data, thresh=2.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(data.shape) == 1:
        data = data[:,None]
    median = numpy.median(data, axis=0)
    diff = numpy.sum((data - median)**2, axis=-1)
    diff = numpy.sqrt(diff)
    med_abs_deviation = numpy.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh

def is_outlier_std(data, multiple_std=2):    
    return abs(data - data.mean()) > multiple_std * data.std()

def main():
    data = numpy.array([1,0,1,10,3,2,1,0])
    print(is_outlier_zscore(data))
    print(is_outlier_std(data))

if __name__ == "__main__":
    main()
