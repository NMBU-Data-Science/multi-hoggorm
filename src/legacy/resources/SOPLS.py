# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 20:00:16 2019

@author: kristl
"""

"""
# Example for SOPLS
    
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import SOPLS

Y_df = pd.read_table('./data/D.txt', index_col=0)
Y = Y_df.values
X1_df = pd.read_table('./data/A.txt', index_col=0)
X1 = X1_df.values
X2_df = pd.read_table('./data/B.txt', index_col=0)
X2 = X2_df.values
X3_df = pd.read_table('./data/C.txt', index_col=0)
X3 = X3_df.values
X = np.hstack([X1, X2, X3])
blocks = np.hstack([np.ones(X1.shape[1]),np.ones(X2.shape[1])*2,np.ones(X3.shape[1])*3])

mlf = make_pipeline(SOPLS.SOPLS(blocks=blocks, ncomp=[5,3,7], max_comp=10, wide_data=True))
mlf.fit(X,Y)
mlf.predict(X)
mlf2 = make_pipeline(StandardScaler(),SOPLS.SOPLS(blocks=blocks, ncomp=[5,3,7], max_comp=10, wide_data=True))
mlf2.fit(X,Y)
mlf2.predict(X)

"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted #check_X_y, check_array, 


class SOPLS(BaseEstimator):
    """ A template estimator to be used as a reference implementation.
    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, blocks=None, ncomp = 'max', max_comp = 20, wide_data = 'auto'):
        self.ncomp = ncomp
        self.max_comp = max_comp
        self.wide_data = wide_data
        assert len(blocks)>1, "Please, specify blocks as an integer vector of length equal to the number of variables"
        self.blocks = blocks

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
        # Store data shapes
        unique_blocks = np.unique(self.blocks)
        self.nblock = len(unique_blocks)
        self.n = Y.shape[0]
        
        # Split X into blocks
        Xsplit = []
        for i in range(self.nblock):
            Xsplit.append(X[:,self.blocks == unique_blocks[i]])
        X = Xsplit
        del Xsplit
        
        p = []
        for i in range(self.nblock):
            p.append(X[i].shape[1])
        if self.ncomp == 'max':
            ncomp = []
            for i in range(self.nblock):
                ncomp.append(min(self.n-1, p[i]))
            self.ncomp = ncomp

        # Check if data are wide or tall
        if self.wide_data == 'auto':
            if sum(p) > np.sqrt(Y.shape[0]):
                self.wide_data = True
            else:
                self.wide_data = False
    
        # Main fitting
        fit = SOPLS_fit(X, Y, self.ncomp, self.max_comp, self.wide_data)
        self.decomp = {'Q' : fit[0][0], 'T' : fit[0][1], 'Ry' : fit[0][2]}
        self.comp_order = {'comp_list' : fit[0][3], 'change_block' : fit[0][4]}
        if self.wide_data == True:
            self.C = fit[1][0]
        
        self.X = X
        self.Y = Y
        
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X, comps = []):
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
        # Store data shapes
        unique_blocks = np.unique(self.blocks)
        Xsplit = []
        for i in range(self.nblock):
            Xsplit.append(X[:,self.blocks == unique_blocks[i]])
        X = Xsplit
        del Xsplit
        
#        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        if self.wide_data:
            if len(comps) == 0:
                comps = self.comp_order['comp_list'][-1]
            Y_pred = SOPLS_predict_wide(self, X, comps)
        
        return Y_pred


#################################################################
# SO-PLS fit function (branching out to wide and tall algorithms)
def SOPLS_fit(X, Y, ncomp, max_comp, wide_data='auto'):
    nblock = len(X)

    # Calculate means and centre input matrices, store dimensions
    X_mean = []
    for i in range(nblock):
        X_mean.append(X[i].mean(axis=0))
        X[i] = X[i] - X_mean[i]
    Y_mean = Y.mean(axis=0)
    Y = Y - Y_mean

    # Select algorithm (wide or tall)
    if wide_data:
        # Liland's kernel PLS
        C = []
        for i in range(nblock):
            C.append(X[i]@X[i].T)
            
        SO = SOPLS_wide(C,Y, ncomp, max_comp)
        return (SO, C)
    
    else:
        # NIPALS based computations
        print('something')        
    

#################################
# SO-PLS workhorse for wide data
def SOPLS_wide(C,Y, ncomp, max_comp, Cval = None):
    nblock = len(C)
    n, nresp = Y.shape

    # Prepare for low redundancy computations
    comp_list, change_block, block_usage = component_combos(nblock, ncomp, max_comp)
    block_combo = block_usage[0]
    block_index = block_usage[1]
    n_combos    = max(block_index)
    n_comps     = np.sum(comp_list, axis=1)
    tot_comps   = len(change_block)

    # All combinations of block usage
    sumC = []
    for i in range(n_combos):
        sumC.append(0)
        for j in range(nblock):
            if block_combo[i,j]:
                sumC[i] = sumC[i] + C[j]

    # Check for prediction
    pred = False
    if not (Cval == None):
        pred = True
        sumCval = []
        sumCval.append(0)
        for j in range(nblock):
            if block_combo[i,j]:
                sumCval[i] = sumCval[i] + Cval[j]
        Y_mean = np.mean(Y, axis=0)
        nval  = Y.shape[0]
        Crval_currB = []
        for b in range(nblock):
            Crval_currB[b] = np.zeros([nval, max_comp])
    
    # Prepare storage
    Ry   = np.zeros([n, tot_comps])
    T    = Ry.copy()
    Q    = np.zeros([nresp, tot_comps])
    Ry_curr = np.zeros([n, max_comp])
    T_curr  = Ry_curr.copy()
    Q_curr  = np.zeros([nresp, max_comp])
    Cr_currB = []
    Y_currB  = []
    for b in range(nblock):
        Y_currB.append(Y.copy())
        Cr_currB.append(np.zeros([n, max_comp]))
    Y_curr = Y.copy()
    if pred:
        Y_pred = np.zeros([nval, nresp, tot_comps])
  
    # --------- Component extraction loop -------------
    for comp in range(1,tot_comps):
        cb = change_block[comp]
        comp_curr = n_comps[comp]
        Y_curr    = Y_currB[cb]
        
        t = C[cb] @ Y_curr
        if nresp > 1: # Multi-response
            usv = np.linalg.svd(Y_curr.T @ t)
            w = usv[0][:,0]
            t = t @ w
        else:
            t = t.flatten()
        if comp_curr > 1: # Orthogonalize on previous
            t = t - T_curr[:,:comp_curr-1] @ (T_curr[:,:comp_curr-1].T @ t)
        t = t/np.sqrt(sum(t*t))
        if nresp > 1:
            ry = Y_curr @ w
        else:
            ry = Y_curr.copy().flatten()
        q = t.T @ Y_curr
        Y_curr = Y_curr - t[:,np.newaxis] @ q[:,np.newaxis].T # Deflation
        for b in range(cb,nblock):
            Y_currB[b] = Y_curr.copy()
       
        # Store t, q, ry
        T_curr[:, comp_curr-1] = t
        Q_curr[:, comp_curr-1] = q
        Ry_curr[:,comp_curr-1] = ry
        if not pred:
            T[:, comp] = t
            Q[:, comp] = q
            Ry[:,comp] = ry
        if pred:
            # Update "X_val*W" ~= C*Ry with current component
            Cr_currB[cb][:,comp_curr-1]    = C[cb] @ ry
            Crval_currB[cb][:,comp_curr-1] = Cval[cb] @ ry
            if cb < nblock:
                for b in range(cb+1, nblock):
                    Cr_currB[b]    = Cr_currB[cb].copy()
                    Crval_currB[b] = Crval_currB[cb].copy()
            
    
        # Perform prediction at the end of each "curr"-series
        if pred and (comp == tot_comps or change_block[comp+1] < nblock):
            comp_last_block = comp_list[comp, nblock-1] # Length of current series
            if comp_curr - comp_last_block == 0:        # Compensate for first series starting at 0
                comp_last_block = comp_last_block - 1

            # XW(P'W)^-1, ie. WB without Q
            no_Q = Crval_currB[cb][:,:comp_curr] @ \
                np.linalg.inv(T_curr[:,:comp_curr].T @ Cr_currB[cb][:,:comp_curr])
            # Prediction per response
            for r in range(nresp):
                Yp_long = np.cumsum(no_Q * np.repeat(Q_curr[r,:comp_curr], nval, 0))
                Y_pred[:, r, comp-comp_last_block:comp] = Yp_long[:, comp_curr-comp_last_block:comp_curr]
    # -------------- End component extraction loop ---------------
  
    # If prediction, return predicted values
    if pred:
        Y_pred = Y_pred + Y_mean
        return (Y_pred, comp_list, change_block)
    else:
        # Otherwise, return decomposition
        return (Q, T, Ry, comp_list, change_block)


#####################
# SO-PLS prediction #
#####################
def SOPLS_predict_wide(SO, Xval, comps):
    X = SO.X
    Y = SO.Y
    nval = Xval[0].shape[0]
    nblock = len(X)
    nresp  = Y.shape[1]
    path, hits = pathComps(comps, SO.comp_order['comp_list'])
    tot_comp   = len(hits)
    
    Y_pred = np.zeros([nval, nresp, tot_comp])
    Cr = 0; Crval = 0
    for i in range(nblock):
        if comps[i] > 0:
            Xval[i] = Xval[i] - np.mean(X[i], axis=0)
            X[i]    = X[i]    - np.mean(X[i], axis=0)
            Cr      = Cr    + (X[i] @ X[i].T) @ SO.decomp['Ry'][:, hits]
            Crval   = Crval + (Xval[i] @ X[i].T) @ SO.decomp['Ry'][:, hits]
    
    # XW(P'W)^-1, ie. WB without Q
    no_Q = Crval @ np.linalg.inv(SO.decomp['T'][:,hits].T @ Cr)
    # Prediction per response
    for r in range(nresp):
        Yp_long = np.cumsum(no_Q * SO.decomp['Q'][r,hits], axis=1)
        Y_pred[:,r,:] = Yp_long
    Y_pred = Y_pred + np.mean(Y,axis=0)[np.newaxis,:,np.newaxis]
    return Y_pred


#############################################
# Create no-redundance sequence of component
def component_combos(nblock, ncomp, max_comps):
    # Determine block order
    unfiltered = np.array(list(range(ncomp[0]+1)))[:,np.newaxis]
    for i in range(1,nblock):
        unfiltered = np.hstack([np.repeat(list(range(ncomp[i]+1)), unfiltered.shape[0])[:,np.newaxis],
                    np.tile(unfiltered, (ncomp[i]+1, 1))])

#  names(unfiltered) <- paste0('block ', 1:nblock)
    comp_list = unfiltered[np.sum(unfiltered,1) <= max_comps]
    first_great = lambda a: np.argmax(a>0)
    change_block = np.hstack([nblock-1, np.apply_along_axis(first_great,1,np.diff(comp_list,axis=0))])
  
    # Determine involved blocks
    block_usage = np.unique(comp_list!=0,axis=0, return_inverse=True)
    return (comp_list, change_block, block_usage)


########################
# Path through compList
def pathComps(comps, comp_list):
    nblock = len(comps)
    mat = np.zeros([0, nblock])
    for b in range(nblock):
        base = np.arange(1,comps[b]+1)[np.newaxis,:]
        base = [list(range(1,comps[b]+1))]
        if b > 0:
            for c in range(b-1,-1,-1):
                base.insert(0,[comps[c]]*comps[b])
        if b < nblock:
            for c in range(b+1,nblock):
                base.append([0]*comps[b])
        mat = np.append(mat, np.array(base).T, axis=0)
    hits = []
    for i in range(mat.shape[0]):
        hits.append(np.where(np.sum(mat[i]==comp_list, axis=1)==nblock)[0][0])
    return (mat, hits)

