# -*- coding: utf-8 -*-
"""
@author: kristl
"""

"""
# Example for PLSR

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import PLSR

NIR = pd.read_csv('./data/gasoline_NIR.txt', header=None, sep='\s+')
octane = pd.read_csv('./data/gasoline_octane.txt', header=None, sep='\s+')
plsr = PLSR.PLSR(ncomp = 10)
plsr.fit(NIR.values, octane.values)
Y_pred = plsr.predict(NIR.values)

Z = NIR.values.copy()
Z_shape = Z.shape
nelem = np.prod(Z_shape)
proportion = 0.10
positions = np.random.permutation(list(range(nelem)))[:int(nelem*proportion)]
ind1, ind2 = np.unravel_index(positions,Z.shape)
for i in range(len(positions)):
    Z[ind1[i], ind2[i]] = np.nan
nanplsr = PLSR.PLSR(ncomp = 10)
nanplsr.fit(Z,octane.values)
Y_pred_nan = nanplsr.predict(Z)

plt.plot(octane.values, Y_pred, 'o')
plt.plot(octane.values, Y_pred_nan, 'o')
plt.legend(['Full data','{}% NaNs'.format(proportion*100)])
plt.xlabel('Reference')
plt.ylabel('Prediction')
plt.plot([83.5,89.5],[83.5,89.5],'--')
plt.show()
"""

#%% Utility functions
def issymmetric(X, rtol=1e-05, atol=1e-08):
    if not np.diff(X.shape)[0] == 0:
        return False
    else:
        return np.allclose(X, X.T, rtol=rtol, atol=atol)

#%% PLSR
import numpy as np
import numpy.linalg as nplin
def plsr_(X, Y, ncomp='max', algorithm='auto'):
    """
    Just PLSR
    """
    E = X - np.mean(X, axis=0)
    F = Y - np.mean(Y, axis=0)
    n_obj, p = X.shape
    n_resp = Y.shape[1]
    
    if algorithm == 'auto':
        if n_obj > p:
            algorithm = 'NIPALS'
        else:
            algorithm = 'PKPLS'
    
    if ncomp == 'max':
        ncomp = min(n_obj-1, p)

    if algorithm == 'NIPALS':
        T = np.zeros([n_obj, ncomp])
        W = np.zeros([p, ncomp])
        P = W.copy()
        Q = np.zeros([n_resp, ncomp])
        
        for i in range(ncomp):
            w, _, _ = nplin.svd(E.T @ F, full_matrices=False); w = w[:,0:1]
            w = w / np.sqrt(np.sum(w**2))
            t = E @ w
            p = (E.T @ t) / np.sum(t**2)
            q = (F.T @ t) / np.sum(t**2)
            E = E - t @ p.T
            F = F - t @ q.T
            
            W[:,i] = w[:,0]
            T[:,i] = t[:,0]
            P[:,i] = p[:,0]
            Q[:,i] = q[:,0]

    if algorithm == 'PKPLS':
        if issymmetric(X):
            C = X
        else:
            C = E @ E.T
        Ry = np.zeros([n_obj, ncomp])
        T  = np.zeros([n_obj, ncomp])
        Q  = np.zeros([n_resp, ncomp])
        for i in range(ncomp):
            if n_resp == 1:
                t = C @ F
            else:
                tt = C @ F
                _, _, a = nplin.svd(F.T @ tt, full_matrices=False)
                t = tt @ a[:,:1]
            if i>0: # Orthogonalize on previous
                t = t - T[:,:i] @ (T[:,:i].T @ t)
            t = t / np.sqrt(np.sum(t**2))
            T [:,i:i+1] = t
            q = t.T @ F
            Q[:,i] = q
            if n_resp == 1:
                Ry[:,i:i+1] = F
            else:
                Ry[:,i:i+1] = F @ a[:,:1]
            F = F - t @ q
        W = X.T @ Ry
        W_norms = np.sqrt(np.sum(W**2, axis=0))
        for i in range(ncomp):
            W[:,i] = W[:,i]/W_norms[i]
        P = X.T @ T
    
    return W, P, T, Q

def nanplsr_(X, Y, ncomp='max', algorithm='auto', tol=10e-12, max_iter=200):
    """
    Just PLSR
    """
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
        W, P, T, Q = plsr_(Z, Y, ncomp=ncomp, algorithm=algorithm)
        Z_pred = T@P.T + Z_means
        err = sum((Z[the_nans]-Z_pred[the_nans])**2)
        Z[the_nans] = Z_pred[the_nans]
        
    return (W, P, T, Q, (iters, err, Z))


#%% PLSR à la scikit-learn
from sklearn.base import BaseEstimator
from PCA import nanpca_
#from sklearn.utils.validation import check_array #check_is_fitted, check_X_y, , 

class PLSR(BaseEstimator):
    """ A template estimator to be used as a reference implementation.
    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, ncomp = 'max',tol=10e-12, max_iter=200, algorithm='auto'):
        self.ncomp = ncomp
        self.tol = tol
        self.max_iter = max_iter
        self.algorithm = algorithm

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
            self.loadingWeights, self.loadings, self.scores, self.Yloadings = plsr_(X, Y, ncomp=self.ncomp, algorithm=self.algorithm)
            self.X_means = np.mean(X, axis=0)
        else:
            self.loadingWeights, self.loadings, self.scores, self.Yloadings, (iters, err, self.imputed) = nanplsr_(X, Y, ncomp=self.ncomp, algorithm=self.algorithm, max_iter=self.max_iter, tol=self.tol)
            self.X_means = np.mean(self.imputed, axis=0)
        
        self.X = X
        self.Y_means = np.mean(Y, axis=0)
        self.Y = Y-self.Y_means
        self.beta = self.loadingWeights @ nplin.inv(self.loadings.T @ self.loadingWeights) @ self.Yloadings.T
        self.is_fitted_ = True
        return self

    def predict(self, X, Y=None):
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
            if Y == None:
                scores, loadings, (iters, err, X) = nanpca_(X, self.ncomp, max_iter = self.max_iter, tol = self.tol)
            else:
                loadingWeights, scores, loadings, Yloadings, (iters, err, X) = plsr_(X, Y, ncomp=self.ncomp, algorithm=self.algorithm, max_iter=self.max_iter, tol=self.tol)
        
        Y_pred = (X-self.X_means) @ self.beta + self.Y_means

        return Y_pred



#%% Compare PLSR with hoggorm
# import hoggorm as ho
# pls1 = ho.nipalsPLS1(NIR.values, octane.values, numComp=10)


#%% Convenience functions
def X_PRESSE_indVar(X, T, P):
    Z = X-np.mean(X,axis=0)
    presse = np.zeros([T.shape[1]+1, X.shape[1]])
    presse[0,:] = np.sum(Z**2, axis=0)
    for i in range(T.shape[1]):
        presse[i+1,:] = np.sum((Z - T[:,:i+1]@P[:,:i+1].T)**2, axis=0)
    return presse
def X_PRESSE(X, T, P):
    Z = X-np.mean(X,axis=0)
    presse = np.zeros([T.shape[1]+1])
    presse[0] = np.sum(Z**2)
    for i in range(T.shape[1]):
        presse[i+1] = np.sum((Z - T[:,:i+1]@P[:,:i+1].T)**2)
    return presse
def X_MSEE_indVar(X, T, P):
    return X_PRESSE_indVar(X, T, P)/X.shape[0]
def X_MSEE(X, T, P):
    return X_PRESSE(X, T, P)/X.shape[0]
def X_RMSEE_indVar(X, T, P):
    return np.sqrt(X_PRESSE_indVar(X, T, P)/X.shape[0])
def X_RMSEE(X, T, P):
    return np.sqrt(X_PRESSE(X, T, P)/X.shape[0])

#%% Strategies
"""
Standardisation : 
- Exact : explicit standardization in the CV-loop
- Approximate : pre-standardization outside the CV-loop (huge speed gains for PKPLS)
Strip down basic calculations to a bare minimum:
- Leave most calculation as post-calculations.
    - Automatic feeding back into the class object for quick later retrival.
"""
