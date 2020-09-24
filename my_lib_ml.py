from nipype.algorithms.icc import ICC_rep_anova
import numpy

def icc21(M):
    """ICC Intraclass correlation coefficient.
    Calculates ICC is for a two-way, fully crossed random efects model.
    This type of ICC is appropriate to describe the absolute agreement
    among shape measurements from a group of k raters, randomly selected
    from the population of all raters, made on a set of n items.
    Shrout and Fleiss: ICC(2,1)
    McGraw and Wong:   ICC(A,1)
    M is the array of measurements
        The dimensions of M are n x k, where
            n is the # of subjects / groups
            k is the # of raters """

    if not isinstance(M, np.ndarray):
        raise TypeError("Input must be a numpy array")

    n, k = M.shape

    u1 = np.mean(M, axis = 0)
    u2 = np.mean(M, axis = 1)
    u = np.mean(M[:])

    SS = np.sum((M - u) ** 2)
    MSR = k/(n-1) * np.sum((u2-u) ** 2)
    MSC = n/(k-1) * np.sum((u1-u) ** 2)
    MSE = (SS - (n-1)*MSR - (k-1)*MSC) / ((n-1)*(k-1))
    icc = (MSR - MSE) / (MSR + (k-1)*MSE + k/n*(MSC-MSE))
    # notice: this is ICC(2,1)
    return icc

def icc31(Y):
    icc31,_,_,_,_,_ = ICC_rep_anova(Y)
    return icc31