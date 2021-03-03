# -*- coding: utf-8 -*-
"""
@author: kristl
"""

"""
# EXAMPLES for PCA

import pandas as pd
import matplotlib.pyplot as plt
from PCA import pca_, nanpca_

X = np.array([[1, 2, 3, 4],[2, 1, 3, 3], [3, 5, 5, 1]], dtype='float64')
scores, loadings = pca_(X)

NIR = pd.read_csv('./data/gasoline_NIR.txt', header=None, sep='\s+')
plt.plot(NIR.values.T)
plt.show()

scores, loadings = pca_(NIR.values)
plt.plot(scores[:,0],scores[:,1],'o')
plt.show()
plt.plot(loadings[:,0])
plt.plot(loadings[:,1])
plt.show()

# Random NaNs
Z = NIR.values.copy()
scores_orig, loadings_orig = pca_(Z,10)

Z = NIR.values.copy()
Z_shape = Z.shape
nelem = np.prod(Z_shape)
proportion = 0.10
positions = np.random.permutation(list(range(nelem)))[:int(nelem*proportion)]
ind1, ind2 = np.unravel_index(positions,Z.shape)
for i in range(len(positions)):
    Z[ind1[i], ind2[i]] = np.nan
scores, loadings, (iters, err, imputed) = nanpca_(Z,10)
plt.plot(scores_orig[:,0],scores_orig[:,1],'o')
plt.plot(scores[:,0],scores[:,1],'o')
plt.legend(['Full data','{}% NaNs'.format(proportion*100)])
plt.show()
"""

#%% PCA
import numpy as np
import numpy.linalg as nplin
def pca_(X, ncomp='max', center=True, sing_vals=False):
    """
    Just PCA
    """
    if isinstance(center, bool) and center:
        Xc = X-np.mean(X, axis=0)
    u, s, vh = nplin.svd(Xc, full_matrices=False, compute_uv=True)
    if ncomp == 'max':
        ncomp = np.shape(u)[1]
    scores = u[:,:ncomp] * s[:ncomp]
    loadings = vh[:ncomp,:].T
    if sing_vals:
        return (scores, loadings, s[:ncomp])
    else:
        return (scores, loadings)

def nanpca_(X, ncomp='max', center=True, tol=10e-12, max_iter=200, sing_vals=False):
    """
    Just PCA with imputation
    """
    # Location of NaNs and initial guess using column means
    the_nans = np.isnan(X)
    X_means = np.nanmean(X,axis=0)
    imputations = np.zeros(np.shape(X), dtype='float64')
    imputations[the_nans] = 1
    imputations *= X_means
    Z = X.copy()
    Z[the_nans] = imputations[the_nans]
    err = np.inf
    
    # Imputation loop
    iters = 0
    while (err > tol) and (iters < max_iter):
        iters += 1
        Z_means = np.mean(Z, axis=0)
        scores, loadings = pca_(Z, ncomp=ncomp)
        Z_pred = scores@loadings.T + Z_means
        err = sum((Z[the_nans]-Z_pred[the_nans])**2)
        Z[the_nans] = Z_pred[the_nans]
        
    if sing_vals:
        scores, loadings, singulars = pca_(Z, ncomp=ncomp, sing_vals=True)
        return (scores, loadings, singulars, (iters, err, Z))
    else:
        return (scores, loadings, (iters, err, Z))
