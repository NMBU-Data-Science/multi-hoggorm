# -*- coding: utf-8 -*-
"""
@author: kristl
"""

"""
# EXAMPLE PCR

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import PCR

NIR = pd.read_csv('./data/gasoline_NIR.txt', header=None, sep='\s+')
octane = pd.read_csv('./data/gasoline_octane.txt', header=None, sep='\s+')
pcr = PCR.PCR(ncomp = 10)
pcr.fit(NIR.values,octane.values)
Y_pred = pcr.predict(NIR.values)

Z = NIR.values.copy()
Z_shape = Z.shape
nelem = np.prod(Z_shape)
proportion = 0.10
positions = np.random.permutation(list(range(nelem)))[:int(nelem*proportion)]
ind1, ind2 = np.unravel_index(positions,Z.shape)
for i in range(len(positions)):
    Z[ind1[i], ind2[i]] = np.nan
nanpcr = PCR.PCR(ncomp = 10)
nanpcr.fit(Z,octane.values)
Y_pred_nan = nanpcr.predict(Z)

plt.plot(octane.values, Y_pred, 'o')
plt.plot(octane.values, Y_pred_nan, 'o')
plt.legend(['Full data','{}% NaNs'.format(proportion*100)])
plt.xlabel('Reference')
plt.ylabel('Prediction')
plt.plot([83.5,89.5],[83.5,89.5],'--')
plt.show()
"""

#%% PCR à la scikit-learn
from sklearn.base import BaseEstimator
from PCA import pca_, nanpca_
import numpy as np
import numpy.linalg as nplin
#from sklearn.utils.validation import check_array #check_is_fitted, check_X_y, , 

class PCR(BaseEstimator):
    """ A template estimator to be used as a reference implementation.
    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, ncomp = 'max',tol=10e-12, max_iter=200):
        self.ncomp = ncomp
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, Y):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : list of {array-like, sparse matrices}, shape (n_samples, n_features)
            The training input samples.
        Y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        self.has_missing = np.any(np.isnan(X))
        if not self.has_missing:
            self.scores, self.loadings, self.singulars = pca_(X, ncomp=self.ncomp, sing_vals=True)
            self.X_means = np.mean(X, axis=0)
        else:
            self.scores, self.loadings, self.singulars, (iters, err, self.imputed) = nanpca_(X, ncomp=self.ncomp, sing_vals=True, max_iter=self.max_iter, tol=self.tol)
            self.X_means = np.mean(self.imputed, axis=0)
        
        self.X = X
        self.Y_means = np.mean(Y, axis=0)
        self.Y = Y-self.Y_means
        self.beta = self.loadings @ nplin.inv(np.diag(self.singulars**2)) @ self.scores.T @ self.Y
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
#        X = check_array(X, accept_sparse=True)
        assert self.is_fitted_, 'Run fit before predict or use fit_predict'
        
        if np.any(np.isnan(X)):
            scores, loadings, (iters, err, X) = nanpca_(X, self.ncomp, max_iter = self.max_iter, tol = self.tol)
        
        Y_pred = (X-self.X_means) @ self.beta + self.Y_means

        return Y_pred



