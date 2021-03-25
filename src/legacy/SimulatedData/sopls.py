# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 10:02:38 2011

@author: oliver.tomic

@info:
Implementation of SO-PLS algorithm.

FIXME: provide refernce to paper 


Changes from last version: 10.10.2012
-------------------------------------
- changed X_cumCalExplVars_indVar to X_cumCalExplVar_indVar
- changed X_cumCalExplVars to X_cumCalExplVar_indVar

- fixed computation of cumulative validated explained variance for Y
-- changed <<for indX, Xblock in enumerate(Xblocks_train_procList):>>
   to <<for indX in range(len(comb_nonZeroIndexArr)):>>
   around line 1543

Changes from last version: 10.01.2013
-------------------------------------
- fixed reconstruction of X blocks when standarised. Used always X_std from
  very last X block instead of X_std from each particluar X block that is to
  be reconstructed
- Extended plotting for more than 5 Y variables up to 15.
- Cleaned up and removed most print commands

"""

# Import necessary modules
#import matplotlib
#matplotlib.use('Agg')
import numpy as np
import numpy.linalg as nl
import plsr
import cross_val as cv
import statTools as st
import matplotlib.pyplot as plt
import pca



class SOPLSCV:
    """
    GENERAL INFO
    ------------
    This class carries out cross validation for Sequential Orthogonlised 
    Partial Least Squares regression (SO-PLS).
    """
    
    def __init__(self, Y, XblocksList, **kargs):
        """
        Do cross validation for SO-PLS. Compute RMSECV for a number of 
        components provided by user. RMSECV is then used in Måge plot to 
        decide on optimal number of components in each X block.
        """
        
#==============================================================================
#         Check what is provided by user for SO-PLS cross validation
#============================================================================== 
        
        # First of all, centre Y and all X blocks, such that orthogonalisation
        # procedure is simplified.        
        self.arrY_input = Y
        self.XblocksList_input = XblocksList[:]
        
        # Check whether number of PC's that are to be computed is provided.
        # If NOT, then number of PC's is set to either number of objects or
        # variables of X, whichever is smaller (maxComp). If number of  
        # PC's IS provided, then number is checked against maxPC and set to
        # maxPC if provided number too large.
        self.XcompsList = []
        
        # If number of components for X blocks are not provided        
        if 'Xcomps' not in kargs.keys():
            for Xblock in self.XblocksList_input:
                maxComp = min(np.shape(Xblock))
                
                # Set maximum number of components to 15
                if maxComp > 20: maxComp = 20 
                self.XcompsList.append(maxComp)
           
        # If number of components are provided, check whether that many
        # components can be computed. If given number of components is too
        # high, set number to max possible.
        else:
            self.XcompsList_input = kargs['Xcomps']
            for ind, comp in enumerate(kargs['Xcomps']):
                givenComp = kargs['Xcomps'][ind]
                
                # Set maximum number of components to 15
                maxComp = min(np.shape(self.XblocksList_input[ind]))
                if maxComp > 20: maxComp = 20       
                
                if givenComp > maxComp:
                    self.XcompsList.append(maxComp)
                else:
                    self.XcompsList.append(givenComp)
        
        
        # Check whether cvType is provided. If NOT, then use "loo" by default.
        if 'cvType' not in kargs.keys():
            self.cvType = ["loo"]
        else:
            self.cvType = kargs['cvType']
        
        # Check whether standardisation of X and Y are requested by user. If 
        # NOT, then all X blocks and Y are only centred by default. 
        if 'Xstand' not in kargs.keys():
            self.XstandList = []
            for Xblock in self.XblocksList_input:
                self.XstandList.append(False)
        else:
            self.XstandList = kargs['Xstand']
        
        if 'Ystand' not in kargs.keys():
            self.Ystand = False
        else:
            self.Ystand = kargs['Ystand']
        
        
        # Check dimensionality of Y and choose between PLS1 or PLS2 thereafter
        # --------------------------------------------------------------------
        numYvars = np.shape(self.arrY_input)[1]
        if numYvars == 1:
            PLS = plsr.nipalsPLS1
        else:
            PLS = plsr.nipalsPLS2
            
        
#==============================================================================
#         Here the main loop starts: looping over all possible combinations
#         of PC's in X blocks
#==============================================================================
        

        # Find all possible combinations of provided X components
        # -------------------------------------------------------        
        # Construct a list that holds all possible combinations of components
        # for each X block. Must use a dummy list that holds for construction 
        # [compsX1+1, compsX2+1] when [compsX1, compsX2] is provided. This is 
        # because of Python indexing starting at zero. 
        dummyXcomps = []
        for comp in self.XcompsList:
            comp = comp + 1
            dummyXcomps.append(comp)
                
        self.XcompComb = []
        self.sopls_RMSECVarr = np.zeros(dummyXcomps)
        for ind, item in np.ndenumerate(self.sopls_RMSECVarr):
            self.XcompComb.append(ind)
        
#        # Temporary code for computing RMSECV for Y1 and Y2
#        self.sopls_RMSECVarr_Y1 = self.sopls_RMSECVarr.copy()
#        self.sopls_RMSECVarr_Y2 = self.sopls_RMSECVarr.copy()
       
        # ----- HERE STARTS THE LOOP OVER ALL PC COMBINATIONS -----        
        # Run SO-PLS for all possible combinations of X components given
        # for each X block
        self.resultsAllCombs = {}
        self.PRESS = {}
        self.Ypred = {}
        
        allComputations = len(self.XcompComb)
        
        for ind, comb in enumerate(self.XcompComb):
            print; print;
            print(comb, ind, round(float(ind+1)/allComputations*100,1))
            
            resultsCurrentComb = {}
            
            
            # Solutions for special cases where cross-validation of SO-PLS
            # will not be run (e.g. zero components in all blocks, or zero in
            # all except one X block. In these cases we use RMSECV directly
            # from simple one X block PLSR model. 
            
            # When all ALL X blocks have zero components do nothing and
            # continue with next components combination
            # =========================================================
            if sum(comb) == 0:                
                continue
            
            # Check which X block is the first that has one or more components
            # to be computed (first non-zero). 
            comb_nonZeroIndexArr = np.flatnonzero(np.array(comb))
            position_firstNonZero = comb_nonZeroIndexArr[0] 
            actualComps = comb[position_firstNonZero]
            
            
            # Do ordinary PLSR when only one X block has non-zero components.
            # ===============================================================
            if len(comb_nonZeroIndexArr) == 1:
                model = PLS(self.XblocksList_input[position_firstNonZero], \
                            self.arrY_input, \
                            numPC=actualComps, \
                            Xstand = self.XstandList[position_firstNonZero], \
                            Ystand = self.Ystand, \
                            cvType = ["loo"])
                self.sopls_RMSECVarr[comb] = model.Y_RMSECV()[-1]
                

                # Here is where RMSCV for each individual variable is extracted                
#                self.sopls_RMSECVarr_Y1[comb] = model.RMSECV_indVar_arr()[:,0][-1]
#                self.sopls_RMSECVarr_Y2[comb] = model.RMSECV_indVar_arr()[:,1][-1]
#                print '***', self.sopls_RMSECVarr_Y1[comb], comb
#                print '***', self.sopls_RMSECVarr_Y2[comb], comb
                
                # Insert RMSECV for zero X components into SO-PLS RMSCEV array
                self.sopls_RMSECVarr[self.XcompComb[0]] =  model.Y_RMSECV()[0]
                
#                self.sopls_RMSECVarr_Y1[self.XcompComb[0]] = model.RMSECV_indVar_arr()[:,0][0]
#                self.sopls_RMSECVarr_Y2[self.XcompComb[0]] = model.RMSECV_indVar_arr()[:,1][0]
#                
#                print '***', self.sopls_RMSECVarr[self.XcompComb[0]], self.XcompComb[0]
                print('Single PLS this time')
                continue
                
            
            # FOR ALL OTHER CASES with two or more X blocks
            #==============================================        
            else:
                # First devide into combinations of training and test sets
                numObj = np.shape(self.arrY_input)[0]
                
                if self.cvType[0] == "loo":
                    print("loo")
                    cvComb = cv.LeaveOneOut(numObj)
                elif self.cvType[0] == "lpo":
                    print("lpo")
                    cvComb = cv.LeavePOut(numObj, self.cvType[1])
                elif self.cvType[0] == "lolo":
                    print("lolo")
                    cvComb = cv.LeaveOneLabelOut(self.cvType[1])
                else:
                    print('Requested form of cross validation is not available')
                    pass
                
                # Collect train and test set in dictionaries for each PC and put
                # them in this list.
                segCount = 0
                
                # Generate a dictionary that holds Ypred after each X block. This
                # will used later to compute mean Ypred across X blocks, which 
                # then is used to compute RMSEP
                YpredConstrDict = {}
                YpredList_cv = []
                
                for constrInd, constrItem in enumerate(self.XblocksList_input):
                    YpredConstrDict[constrInd] = []
                
                # ----- HERE STARTS THE CROSS VALIDATION LOOP -----
                # First devide into combinations of training and test sets
                for train_index, test_index in cvComb:
                    
                    # Define training and test set for Y
                    Y_train, Y_test = cv.split(train_index, test_index, \
                            self.arrY_input)
                    
                    # -------------------------------------------------------------                    
                    # Center or standardise Y according to users choice                    
                    if self.Ystand == True:
                        # Standardise training set Y using mean and STD
                        Y_train_mean = np.average(Y_train, axis=0).reshape(1,-1)
                        Y_train_std = np.std(Y_train, axis=0, ddof=1).reshape(1,-1)
                        Y_train_proc = (Y_train - Y_train_mean) / Y_train_std
                        
                        # Standardise test set Y using mean and STD from 
                        # training set
                        Y_test_proc = (Y_test - Y_train_mean) / Y_train_std
                    
                    else:
                        # Centre training set Y using mean
                        Y_train_mean = np.average(Y_train, axis=0).reshape(1,-1)
                        Y_train_proc = Y_train - Y_train_mean
                        Y_train_std = None
                        
                        # Centre test set Y using mean and STD from 
                        # training set
                        Y_test_proc = Y_test - Y_train_mean
                    # -------------------------------------------------------------
                    
                    # Now do the same for the X blocks. Do this by iterating 
                    # through self.XblocksList_input and take out test and
                    # training data
                    Xblocks_trainDict = {}
                    Xblocks_testDict = {}
                    
                    Xblocks_trainList = []
                    Xblocks_testList = []
                    
                    Xblocks_train_meanList = []
                    Xblocks_train_stdList = []
                    Xblocks_train_procList = []
                    
                    Xblocks_test_procList = []
                    
                    segCount = segCount + 1
                    
                    
                    # For each X block split up into training and test set for 
                    # the cross validation segment we are in right now
                    for ind, item in enumerate(self.XblocksList_input):
                        
                        X_train, X_test = cv.split(train_index, test_index, item)
                        Xblocks_trainDict[ind] = X_train
                        Xblocks_testDict[ind] = X_test
                        
                        Xblocks_trainList.append(X_train)
                        Xblocks_testList.append(X_test)
                        
                        
                        # ---------------------------------------------------------
                        # Center or standardise X blocks according to users choice
                        if self.XstandList[ind] == True:
                            # Compute standardised X blocks using mean and STD
                            X_train_mean = np.average(X_train, axis=0).reshape(1,-1)
                            X_train_std = np.std(X_train, axis=0, ddof=1).reshape(1,-1)
                            X_train_proc = (X_train - X_train_mean) / X_train_std
                            
                            # Append each standardised X block to X blocks training
                            # list
                            Xblocks_train_meanList.append(X_train_mean)
                            Xblocks_train_stdList.append(X_train_std)
                            Xblocks_train_procList.append(X_train_proc)
                            
                            # Standardise test set of each X block using mean and 
                            # and STD from training X blocks and append to X blocks 
                            # test list
                            X_test_proc = (X_test - X_train_mean) / X_train_std                        
                            Xblocks_test_procList.append(X_test_proc)
     
                        else:
                            # Compute centred X blocks using mean
                            X_train_mean = np.average(X_train, axis=0).reshape(1,-1)
                            X_train_proc = X_train - X_train_mean
                            
                            # Append each centred X block to X blocks training list
                            Xblocks_train_meanList.append(X_train_mean.reshape(1,-1))
                            Xblocks_train_stdList.append(None)
                            Xblocks_train_procList.append(X_train_proc)
                            
                            # Centre test set of each X block using mean from 
                            # X block training set and append to X blocks test list
                            X_test_proc = X_test - X_train_mean                        
                            Xblocks_test_procList.append(X_test_proc)
                        # ---------------------------------------------------------
                        
                    
                    # Put all training and test data for Y and X blocks in a 
                    # dictionary. 
                    segDict = {}
                    segDict['Y train'] = Y_train
                    segDict['Y test'] = Y_test
                    segDict['X train'] = Xblocks_trainDict
                    segDict['X test'] = Xblocks_testDict
                    
                    segDict['proc Y train'] = Y_train_proc
                    segDict['Y train mean'] = Y_train_mean.reshape(1,-1)
                    #segDict['Y train std'] = Y_train_std
                    
                    segDict['proc X train'] = Xblocks_train_procList
                    segDict['X train mean'] = Xblocks_train_meanList
                    #segDict['X train std'] = Xblocks_train_stdList
                    
                    segDict['proc X test'] = Xblocks_test_procList
                    segDict['proc Y test'] = Y_test_proc
                    
                    
                    # Now start modelling sequential PLSR over all X blocks.
                    # First X block with non-zero X components will be modelled
                    # with ordinary PLSR. For all following X blocks with non-
                    # zero components, the X block must be orthogonalised with
                    # regard to the X scores of the prior X block. Only then the
                    # orthogonalised X block can be modelled against Y.
                    scoresList = []
                    cv_scoresList = []
                    Blist = []
                    orthoXblockList = []                
                    tCQlist = []                
                    Wlist = []
                    XmeanList = []
                    YmeanList = []
                    Qlist = []
                    Clist = []
                    
                    
                    for indMod, Xblock in enumerate(Xblocks_train_procList):
                        
                        if indMod not in comb_nonZeroIndexArr:
                            continue
                        
                        if indMod == comb_nonZeroIndexArr[0]:
                            
                            # Do ordinary PLSR prior to PLRS with orth. Xblocks
                            model = PLS(Xblock, Y_train_proc, numPC=comb[indMod])
                            
                            # Get X scores and store them in a scores list. The
                            # scores are needed for the orthogonlisation step of 
                            # the next X block.                        
                            scoresList.append(model.X_scores())
                            
                            # Here prediction part starts.
                            # First estimate X scores for test set of X blocks
                            arrW = model.X_loadingsWeights()
                            arrP = model.X_loadings()
                            
                            projScoresList = []
                            X_test_deflationList = [Xblocks_test_procList[indMod]]
                            
                            for predInd in range(comb[indMod]):
    
                                projScores = np.dot(X_test_deflationList[-1], \
                                        arrW[:,predInd])
                                projScoresList.append(projScores)
                                
                                deflated_Xblock_test = X_test_deflationList[-1] - \
                                        np.dot(projScores.reshape(1,-1), \
                                        np.transpose(arrP[:,predInd]).reshape(1,-1))
                                X_test_deflationList.append(deflated_Xblock_test)
                            
                            
                            # Construct array which holds projected X test scores.
                            T_test = np.transpose(np.array(projScoresList))
                            cv_scoresList.append(T_test)
                            
                            # Get Y loadings and scores regression coefficients
                            # for computation Y pred.
                            arrQ = model.Y_loadings()
                            arrC = model.scoresRegressionCoeffs()
                            
                            # Now compute Ypred for the test set of the acutal
                            # X block.
                            tCQ = np.dot(T_test, np.dot(arrC, np.transpose(arrQ)))                        
                            
                            tCQlist.append(tCQ)                        
                            XmeanList.append(model.X_means())
                            YmeanList.append(model.Y_means())
                            
                            Wlist.append(arrW)
                            Qlist.append(arrQ)
                            Clist.append(arrC)
                        
                        else:
                            # Orthogonalise and run PLS and model Y. Could also 
                            # use residuals from previous model, which would give 
                            # the same result. 
                            
                            # Stack X scores horizontally from previous PLS models
                            # and orthogonalise next X block with regard to the
                            # stacked X scores. If there is only one set of scores
                            # then stacking is not necessary
                            
                            if len(scoresList) == 1:
                                T = scoresList[0]
                                cv_T = cv_scoresList[0]
                            
                            else:
                                T = np.hstack(scoresList)
                                cv_T = np.hstack(cv_scoresList)
                            
                            
                            # Orthogonalisation process
                            # X_orth = X - TB
                            B = np.dot(np.dot(nl.inv(np.dot(np.transpose(T), T)), \
                                    np.transpose(T)), Xblock)
                            orth_Xblock = Xblock - np.dot(T, B)
                            Blist.append(B)
                            orthoXblockList.append(orth_Xblock)
                            
                            # Run PLSR on orthogonalised X block. 
                            model = PLS(orth_Xblock, Y_train_proc, \
                                    numPC=comb[indMod])
                            scoresList.append(model.X_scores())
                            
                            # Orthogonalisation of test set of X block
                            orth_Xblock_test = Xblocks_test_procList[indMod] - \
                                    np.dot(cv_T, B)
                            
                            # Here the prediction part starts.
                            # First estimate X scores for test set of X blocks
                            arrW = model.X_loadingsWeights()
                            arrP = model.X_loadings()
                            
                            projScoresList = []
                            X_test_deflationList = [orth_Xblock_test]
                            
                            for predInd in range(comb[indMod]):
                                
                                projScores = np.dot(X_test_deflationList[-1], \
                                        arrW[:,predInd])
                                projScoresList.append(projScores)
                                
                                deflated_Xblock_test = X_test_deflationList[-1] - \
                                        np.dot(projScores.reshape(1,-1), \
                                        np.transpose(arrP[:,predInd]).reshape(1,-1))
                                X_test_deflationList.append(deflated_Xblock_test)
                            
                            
                            # Construct array which holds projected X test scores.
                            T_test = np.transpose(np.array(projScoresList))
                            cv_scoresList.append(T_test)
                            
                            # Get Y loadings and scores regression coefficients
                            # for computation Y pred.
                            arrQ = model.Y_loadings()
                            arrC = model.scoresRegressionCoeffs()
                            
                            # Now compute Ypred for the test set of the acutal
                            # X block.
                            tCQ = np.dot(T_test, np.dot(arrC, np.transpose(arrQ)))
                            
                            tCQlist.append(tCQ)                        
                            XmeanList.append(model.X_means())                        
                            Wlist.append(arrW)
                            Qlist.append(arrQ)
                            Clist.append(arrC)
                
                    # Here the Ypreds for one segment are added across X blocks.
                    # The Ypreds are stored in a list that is later converted into
                    # a YpredFull that has the same dimensions (number of rows)
                    # as Y.
                    tCQsum = np.sum(tCQlist, axis=0)
    
                    if self.Ystand == True:
                        Ypred_cv = (tCQsum * Y_train_std) + Y_train_mean
                    
                    else:
                        Ypred_cv = tCQsum + Y_train_mean
                    
                    YpredList_cv.append(Ypred_cv)
                    
                # Now construct Ypred from cross validated Ypred of each segment
                YpredFull = np.vstack(YpredList_cv)
                            
                #resultsCurrentComb['cv data'] = segDict
                resultsCurrentComb['x scores'] = scoresList
                resultsCurrentComb['B list'] = Blist
                resultsCurrentComb['orth X block'] = orthoXblockList
                resultsCurrentComb['new X scores'] = cv_scoresList
                resultsCurrentComb['W list'] = Wlist
                resultsCurrentComb['C list'] = Clist
                resultsCurrentComb['Q list'] = Qlist
                resultsCurrentComb['tCQ list'] = tCQlist
                resultsCurrentComb['tCQ sum'] = tCQsum
                resultsCurrentComb['Ypred_cv'] = Ypred_cv
                resultsCurrentComb['YpredList_cv'] = YpredList_cv
                resultsCurrentComb['YpredFull'] = YpredFull
                
                
                #resultsCurrentComb['X means'] = XmeanList
                #resultsCurrentComb['Y means'] = YmeanList
                
                    
                self.scoresList = scoresList
                self.Blist = Blist
                self.cv_scoresList = cv_scoresList

                comb_PRESS = np.sum(np.square(self.arrY_input - YpredFull))
                comb_MSEP = comb_PRESS / np.size(self.arrY_input)
                comb_RMSEP = np.sqrt(comb_MSEP)
                self.sopls_RMSECVarr[comb] = comb_RMSEP
                self.PRESS[comb] = comb_PRESS
                self.resultsAllCombs[comb] = resultsCurrentComb
            
#                # Temporary code:
#                # Computing RMSECV for Y1 and Y2
#                Y1_comb_PRESS = np.sum(np.square(self.arrY_input[:,0] - sumYpred[:,0]))
#                Y1_comb_MSEP = Y1_comb_PRESS / np.shape(self.arrY_input)[0]
#                Y1_comb_RMSEP = np.sqrt(Y1_comb_MSEP)
#                self.sopls_RMSECVarr_Y1[comb] = Y1_comb_RMSEP
#                
#                Y2_comb_PRESS = np.sum(np.square(self.arrY_input[:,1] - sumYpred[:,1]))
#                Y2_comb_MSEP = Y2_comb_PRESS / np.shape(self.arrY_input)[0]
#                Y2_comb_RMSEP = np.sqrt(Y2_comb_MSEP)
#                self.sopls_RMSECVarr_Y2[comb] = Y2_comb_RMSEP
            
            
#                PRESS_segment = np.sum(np.square(Y_test - sumYpred))
#                comb_PRESSlist.append(PRESS_segment)           
#                comb_PRESS = np.sum(comb_PRESSlist)
#                comb_MSEP = comb_PRESS / np.shape(self.arrY_input)[0]
#                comb_RMSEP = np.sqrt(comb_MSEP)
#                self.sopls_RMSECVarr[comb] = comb_RMSEP
#                self.PRESS[comb] = comb_PRESS
            
            
            
            
            
    def results(self):
        resDict = {}
        resDict['combRes'] = self.resultsAllCombs            
        
        return resDict
    
    
    def modelSettings(self):
        """
        Returns a dictionary holding most important model settings.
        """
        settings = {}
        settings['Y'] = self.arrY_input
        settings['X blocks'] = self.XblocksList_input
        settings['Xstand'] = self.XstandList
        settings['Ystand'] = self.Ystand
        #settings['analysed X blocks'] = self.XblocksList
        #settings['analysed Y'] = self.arrY
        settings['X comps'] = self.XcompsList_input
        settings['used X comps'] = self.XcompsList
        
        return settings
    

    def RMSECV(self):
        """
        Returns an array holding RMSECV of SO-PLS 
        """
        return self.sopls_RMSECVarr
        #return [self.sopls_RMSECVarr, self.sopls_RMSECVarr_Y1,self.sopls_RMSECVarr_Y2]
    


def plotRMSEP(RMSEParr, *args):
    
    """
    Input:
    ======
    
    RMSEParr: type <array>: Array holding the RMSEP values returned by function
              lsplsCV.
    
    
    Output:
    =======
    
    A scatter plot showing RMSEP for each possible combination of components
    chosen.
    
    """
    
    # Check whether further arguments are provided. If not, then go for
    # maximum number of components, which is the dimension of the RMSEP array.
    if len(args) == 0:
        print('no further parameters given')
        dummyArr = np.zeros(np.shape(RMSEParr))
    
    # Build dummyArr according to provided components and use it for iterating
    # through the desired combinations. May be a multi-dimensional array.
    # Example: [3,2,2] will result in plotting [2,1,1] components because of 
    # Python syntax counting from zero. Need therefore to add 1 to the 
    # submitted component numbers.
    else:
        newArgs = []
        
        for item in args[0]:
            newArgs.append(item + 1)
        dummyArr = np.zeros(newArgs)
    
    
    # This is plotting code. Plot RMSEP values in a scatter plot.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Now plot the RMSEP values in a scatter plot. Save the the minimum RMSEP
    # for each total number of components in a dictionary named minima. 
    minima = {}
    for ind, x in np.ndenumerate(dummyArr):
        #print('++++++++++', ind, '-', x, '--', sum(ind))
        x = sum(ind)
        
        # Use RMSEP with any components as maxRMSEP. If RMSEP is larger than 
        # maxRMSEP for any combination of components it will not be plotted.
        if x == 0:
            maxRMSEP = RMSEParr[ind]
#            print('MAX RMSEP:', maxRMSEP)
#            print('MIN RMSEP:', np.min(RMSEParr))
        
        if RMSEParr[ind] > maxRMSEP:
#            print('............ MAX OUT at', ind, '-', x, '--', sum(ind))
            continue
        
        else:        
            # Make a text lable out of the tuple holding the number of components.
            # that is, from (3,0,1) make a text lable '301'.
            text = ''
            for i in ind:
                #text += str(i)
                text = text + '_' + str(i)
            
            # Plot cirles for each RMSEP value. This needs to be done, as text
            # alone will not be printed/shown in matplotlib.
            ax.scatter(x, RMSEParr[ind], s=10, c='w', marker='o', edgecolor='grey')
            ax.text(x, RMSEParr[ind], text, fontsize=13)
            
#            # TEMPORARY CODE:
#            if RMSEParr[ind] < 2.15:
#                
#                if x >= 10 and x <= 16:
#            
#                    # Plot cirles for each RMSEP value. This needs to be done, as text
#                    # alone will not be printed/shown in matplotlib.
#                    ax.scatter(x, RMSEParr[ind], s=10, c='w', marker='o', edgecolor='grey')
#                    ax.text(x, RMSEParr[ind], text, fontsize=13)
            
            # Store minimum RMSEP in dictionary named minima.
            #print('++++++++++', ind, '-', x, '--', sum(ind))
            if x in minima.keys():
                
                if RMSEParr[ind] < minima[x]:
                    minima[x] = RMSEParr[ind]
                    
            else:
                minima[x] = RMSEParr[ind]
            
        
    
    
    # Find maximum total number of components for iteration. Need to make a
    # list holding all minimum RMSEP.
    maxComp = max(minima.keys())
    minimaList = []
    xPosList = []
    
    for comps in range(maxComp + 1):
        try:
            minimaList.append(minima[comps])
            xPosList.append(comps)
        except KeyError:
            continue
    
    
    # Plot line for lowest RMSEP for each total number of components.
    ax.plot(xPosList, minimaList, 'r--')
    
    ax.set_xlabel('# components')
    ax.set_ylabel('RMSEP')
    
    
#    # TEMPORARY CODE    
#    ax.set_xlim(9.5, 18.8)
#    ax.set_ylim(1.6, 2.2)
    
    plt.show()



class SOPLS:
    """
    GENERAL INFO
    ------------
    This class carries out Sequential Orthogonlised  Partial Least Squares 
    regression (SO-PLS) for predetermined number of components in each 
    X block.
    """
    
    def __init__(self, Y, XblocksList, Xcomps, **kargs):
        """
        Do SO-PLS for a given number of components. Compute all important
        features as Xscores, Xloadings, etc.
        """
    
#==============================================================================
#         Check what is provided by user for SO-PLS
#============================================================================== 
        
        # Acess Y block and X blocks       
        self.arrY_input = Y.copy()
        self.XblocksList_input = XblocksList[:]
        
        # Get number of components for each X block
        self.XcompsList_input = Xcomps[:]
           
        # If number of components are provided, check whether that many
        # components can be computed. If given number of components is too
        # high, set number to max possible.
        self.XcompsList = []
        for ind, comp in enumerate(self.XcompsList_input):
            givenComp = self.XcompsList_input[ind]
            
            # Check each given number of X components against maximum possible
            # number of X components
            maxComp = min(np.shape(self.XblocksList_input[ind]))
            if maxComp > 20: maxComp = 20       
            
            if givenComp > maxComp:
                self.XcompsList.append(maxComp)
            else:
                self.XcompsList.append(givenComp)
        
        # Check whether cvType is provided. If NOT, then use "loo" by default.
        if 'cvType' not in kargs.keys():
            self.cvType = ["loo"]
        else:
            self.cvType = kargs['cvType']
        
        # Check whether standardisation of X and Y are requested by user. If 
        # NOT, then all X blocks and Y are only centred by default. 
        if 'Xstand' not in kargs.keys():
            self.XstandList = []
            for Xblock in self.XblocksList_input:
                self.XstandList.append(False)
        else:
            self.XstandList = kargs['Xstand']
        
        if 'Ystand' not in kargs.keys():
            self.Ystand = False
        else:
            self.Ystand = kargs['Ystand']
        
        # Check dimensionality of Y and choose between PLS1 or PLS2 thereafter
        numYvars = np.shape(self.arrY_input)[1]
        if numYvars == 1:
            PLS = plsr.nipalsPLS1
        else:
            PLS = plsr.nipalsPLS2

#==============================================================================
#         From here SO-PLS algorithm starts (CALIBRATION)
#==============================================================================
        
        # Collect relevant computation results in lists
        self.Xblocks_meanList = []
        self.XscoresList = []
        self.XloadingsList = []
        self.XcorrLoadingsList = []
        
        self.XcumCalExplVars_indVar = []
        self.XcumCalExplVars = []
        self.XpredCalList = []
        
        self.XcumValExplVars_indVar = []
        self.XcumValExplVars = []
        self.XpredValList = []
        
        
        self.YscoresList = []
        self.YloadingsList = []
        self.YcorrLoadingsList = []
        
        self.YcumPredCalList = []
        self.YcumCalExplVar = []
        
        self.YcumPredValList = []
        self.YcumValExplVar = []
        
        # Check which X block is the first that has one or more components
        # to be computed (first non-zero). 
        comb_nonZeroIndexArr = np.flatnonzero(np.array(self.XcompsList))
        position_firstNonZero = comb_nonZeroIndexArr[0] 
        actualComps = self.XcompsList[position_firstNonZero]
        print('Positions of non-zero:', position_firstNonZero)
        print('Actual # of comps of first non-zero:', actualComps)      
                
        # When all ALL X blocks have zero components do nothing and
        # continue with next components combination
        # =========================================================
        if sum(self.XcompsList) == 0:
            print('nothing to do when no components are given')                
        
        # Do ordinary PLSR when only one X block has non-zero components. 
        # ===============================================================
        elif len(comb_nonZeroIndexArr) == 1:

            model = PLS(self.XblocksList_input[position_firstNonZero], \
                        self.arrY_input, \
                        numPC=actualComps, \
                        Xstand = self.XstandList[position_firstNonZero], \
                        Ystand = self.Ystand, \
                        cvType = self.cvType)
            
            # Collect results for X blocks
            # ----------------------------
            self.Xblocks_meanList.append(model.X_means())
            self.XscoresList.append(model.X_scores())
            self.XloadingsList.append(model.X_loadings())
            self.XcorrLoadingsList.append(model.X_corrLoadings())
            
            self.XcumCalExplVars_indVar.append(model.X_cumCalExplVar_indVar())
            self.XcumCalExplVars.append(model.X_cumCalExplVar())
            self.XpredCalList.append(model.X_predCal()\
                    [self.XcompsList_input[position_firstNonZero]])
            
            self.XcumValExplVars_indVar.append(model.X_cumValExplVar_indVar())
            self.XcumValExplVars.append(model.X_cumValExplVar())
            self.XpredValList.append(model.X_predVal()\
                    [self.XcompsList_input[position_firstNonZero]])
            
            
            # Collect results for Y block
            # ---------------------------
            self.Y_mean = model.Y_means()
            self.YscoresList.append(model.Y_scores())
            self.YloadingsList.append(model.Y_loadings())
            self.YcorrLoadingsList.append(model.Y_corrLoadings())
            
            self.YcumPredCalList.append(model.Y_predCal()\
                    [self.XcompsList_input[position_firstNonZero]])            
            
            Y_cal_first = model.Y_cumCalExplVar_indVar()[0,:]
            Y_cal_last = model.Y_cumCalExplVar_indVar()[-1,:]
            self.YcumCalExplVar_indVar = np.vstack((Y_cal_first, Y_cal_last))
            #self.YcumCalExplVar_indVar = model.Y_cumCalExplVar_indVar()
            
            self.YcumCalExplVar.append(0)            
            self.YcumCalExplVar.append(model.Y_cumCalExplVar()\
                    [self.XcompsList_input[position_firstNonZero]]) 
            
            
            self.YcumPredValList.append(model.Y_predVal()\
                    [self.XcompsList_input[position_firstNonZero]])
                    
            Y_val_first = model.Y_cumValExplVar_indVar()[0,:]
            Y_val_last = model.Y_cumValExplVar_indVar()[-1,:]
            self.YcumValExplVar_indVar = np.vstack((Y_val_first, Y_val_last))
            #self.YcumValExplVar_indVar = model.Y_cumValExplVar_indVar()
            
            self.YcumValExplVar.append(0)
            self.YcumValExplVar.append(model.Y_cumValExplVar()\
                    [self.XcompsList_input[position_firstNonZero]]) 
            
            
            # Collect other general results
            # -----------------------------
            self.orthoXblockList = []
            self.Y_proc = model.modelSettings()['analysed Y']
            self.Xblocks_procList = model.modelSettings()['analysed X']
            
        
        # If X components for more than one X block are given, then run 
        # SO-PLS computations.
        # =============================================================
        else:
            
            self.Xblocks_stdList = []
            self.Xblocks_procList = []
            
            # ----------------------------------------------------------
            # Center or standardise X blocks according to user's choice
            for ind, item in enumerate(self.XblocksList_input):
 
                if self.XstandList[ind] == True:
                    # Compute standardised X blocks using mean and STD
                    X_mean = np.average(item, axis=0).reshape(1,-1)
                    X_std = np.std(item, axis=0, ddof=1).reshape(1,-1)
                    X_proc = (item - X_mean) / X_std

                    
                    # Append each standardised X block to X blocks training
                    # list
                    self.Xblocks_meanList.append(X_mean)
                    self.Xblocks_stdList.append(X_std)
                    self.Xblocks_procList.append(X_proc)
     
                else:
                    # Compute centred X blocks using mean
                    X_mean = np.average(item, axis=0).reshape(1,-1)
                    X_proc = item - X_mean
                    
                    # Append each centred X block to X blocks training list
                    self.Xblocks_meanList.append(X_mean.reshape(1,-1))
                    self.Xblocks_stdList.append(None)
                    self.Xblocks_procList.append(X_proc)
            # ----------------------------------------------------------
                            
            # ----------------------------------------------------------                    
            # Center or standardise Y according to user's choice                    
            if self.Ystand == True:
                # Standardise training set Y using mean and STD
                self.Y_mean = np.average(self.arrY_input, axis=0).reshape(1,-1)
                self.Y_std = np.std(self.arrY_input, axis=0, ddof=1).reshape(1,-1)
                self.Y_proc = (self.arrY_input - self.Y_mean) / self.Y_std
            
            else:
                # Centre training set Y using mean
                self.Y_mean = np.average(self.arrY_input, axis=0).reshape(1,-1)
                self.Y_proc = self.arrY_input - self.Y_mean
                self.Y_std = None
            # ---------------------------------------------------------- 
                        
            # Now start modelling sequential PLSR over all X blocks.
            # First X block with non-zero X components will be modelled
            # with ordinary PLSR. For all following X blocks with non-
            # zero components, the X block must be orthogonalised with
            # regard to the X scores of the prior X block. Only then the
            # orthogonalised X block can be modelled against Y.
            self.Blist = []
            self.orthoXblockList = []
            self.Wlist = []
            self.Qlist = []
            self.Clist = []
            
            YprocPredList = []
            
            for indMod, Xblock in enumerate(self.Xblocks_procList):
                
                if indMod not in comb_nonZeroIndexArr:
                    print('NO COMPS for this BLOCK')
                    continue
                
                # Do PLSR on first X block (not orthogonalised)                
                if indMod == comb_nonZeroIndexArr[0]:
                    
                    # Do ordinary PLSR prior to PLRS with orth. Xblocks
                    model = PLS(Xblock, self.Y_proc, \
                            numPC=self.XcompsList_input[indMod], \
                            cvType=self.cvType)
                    
                    # Get X scores and store them in a scores list. The
                    # scores are needed for the orthogonlisation step of 
                    # the next X block.                        
                    self.XscoresList.append(model.X_scores())
                    
                    # Collect X loadings and X correlation loadings in a list 
                    self.XloadingsList.append(model.X_loadings())
                    self.XcorrLoadingsList.append(model.X_corrLoadings())
                    
                    # Get calibrated explained variance in first X block
                    self.XcumCalExplVars_indVar.append(model.X_cumCalExplVar_indVar())
                    self.XcumCalExplVars.append(model.X_cumCalExplVar())
                    
                    # Get X pred for calibration for chosen number of 
                    # components for first X block                    
                    XpredCal_proc = model.X_predCal()[self.XcompsList_input[indMod]]
                    if self.XstandList[indMod] == True:
#                        XpredCal = (XpredCal_proc * X_std) + \
#                                self.Xblocks_meanList[indMod]
                        XpredCal = (XpredCal_proc * self.Xblocks_stdList[indMod]) + \
                                self.Xblocks_meanList[indMod]
                    else:
                        XpredCal = XpredCal_proc + self.Xblocks_meanList[indMod]
                    self.XpredCalList.append(XpredCal)
                    
                    # Get X pred for validation for chosen number of 
                    # components for first X block                    
                    XpredVal_proc = model.X_predVal()[self.XcompsList_input[indMod]]
                    if self.XstandList[indMod] == True:
#                        XpredVal = (XpredVal_proc * X_std) + \
#                                self.Xblocks_meanList[indMod]
                        XpredVal = (XpredVal_proc * self.Xblocks_stdList[indMod]) + \
                                self.Xblocks_meanList[indMod]
                    else:
                        XpredVal = XpredVal_proc + self.Xblocks_meanList[indMod]
                    self.XpredValList.append(XpredVal)
                    
                    # Get validated explained variance in first X block
                    self.XcumValExplVars_indVar.append(model.X_cumValExplVar_indVar())
                    self.XcumValExplVars.append(model.X_cumValExplVar())
                    
                    # Get Y scores, Y loadings and Y correlation loadings for 
                    # the chosen number of components in this X block 
                    self.YscoresList.append(model.Y_scores())
                    self.YloadingsList.append(model.Y_loadings())
                    self.YcorrLoadingsList.append(model.Y_corrLoadings())  
                    
                    # Get Y pred from calibration. This Y is processed and
                    # needs to be 'un-processed' (un-center, un-standardise)
                    # before being put into list.
                    YpredCal_proc = model.Y_predCal()[self.XcompsList_input[indMod]]
                    YprocPredList.append(YpredCal_proc)
                    
                    if self.Ystand == True:
                        YpredCal = (YpredCal_proc * self.Y_std) + \
                                self.Y_mean
                    else:
                        YpredCal = YpredCal_proc + self.Y_mean
                    self.YcumPredCalList.append(YpredCal)    
                                                                                                                                                     
                # Do PLSR on all other X blocks (orthogonalised)                          
                else:
                    print('second or later X block')
                    # Orthogonalise next X block and run PLSR to model Y. Could 
                    # also use residuals from previous model, which would give 
                    # the same result. 
                    
                    # Stack X scores horizontally from previous PLS models
                    # and orthogonalise next X block with regard to the
                    # stacked X scores. If there is only one set of scores
                    # then stacking is not necessary
                    if len(self.XscoresList) == 1:
                        T = self.XscoresList[0]
                    
                    else:
                        T = np.hstack(self.XscoresList)
                                        
                    # Orthogonalisation process
                    # X_orth = X - TB
                    B = np.dot(np.dot(nl.inv(np.dot(np.transpose(T), T)), \
                            np.transpose(T)), Xblock)
                    orth_Xblock = Xblock - np.dot(T, B)
                    self.Blist.append(B)
                    self.orthoXblockList.append(orth_Xblock)
                    
                    # Run PLSR on orthogonalised X block. 
                    model = PLS(orth_Xblock, self.Y_proc, \
                            numPC=self.XcompsList[indMod], \
                            cvType = self.cvType)
                    
                    # Get X scores for PLSR on orthogonlised X block
                    self.XscoresList.append(model.X_scores())
                    
                    # Get X loadings and X correlation loadings on 
                    # orthogonalised X block
                    self.XloadingsList.append(model.X_loadings())
                    self.XcorrLoadingsList.append(model.X_corrLoadings())
                    
                    # Get X pred for calibration for chosen number of 
                    # components for first X block. Need to reverse 
                    # orthogonalisation first before un-processing (that is
                    # un-center or un-standardise)
                    XpredCal_orth = model.X_predCal()[self.XcompsList_input[indMod]]
                    if self.XstandList[indMod] == True:
                        XpredCal_proc = XpredCal_orth + np.dot(T, B)
#                        XpredCal = (XpredCal_proc * X_std) + \
#                                self.Xblocks_meanList[indMod]
                        XpredCal = (XpredCal_proc * self.Xblocks_stdList[indMod]) + \
                                self.Xblocks_meanList[indMod]
                    else:
                        XpredCal_proc = XpredCal_orth + np.dot(T, B)
                        XpredCal = XpredCal_proc + self.Xblocks_meanList[indMod]
                    self.XpredCalList.append(XpredCal)
                    
                    # Get X pred for validation for chosen number of 
                    # components for first X block. Need to reverse 
                    # orthogonalisation first before un-processing (that is
                    # un-center or un-standardise)
                    XpredVal_orth = model.X_predVal()[self.XcompsList_input[indMod]]
                    if self.XstandList[indMod] == True:
                        XpredVal_proc = XpredVal_orth + np.dot(T, B)
#                        XpredVal = (XpredVal_proc * X_std) + \
#                                self.Xblocks_meanList[indMod]
                        XpredVal = (XpredVal_proc * self.Xblocks_stdList[indMod]) + \
                                self.Xblocks_meanList[indMod]
                    else:
                        XpredVal_proc = XpredVal_orth + np.dot(T, B)
                        XpredVal = XpredVal_proc + self.Xblocks_meanList[indMod]
                    self.XpredValList.append(XpredVal)
                    
                    # Get explained variance in X block
                    self.XcumCalExplVars_indVar.append(model.X_cumCalExplVar_indVar())
                    self.XcumCalExplVars.append(model.X_cumCalExplVar())
                    
                    # Get validated explained variance in first X block
                    self.XcumValExplVars_indVar.append(model.X_cumValExplVar_indVar())
                    self.XcumValExplVars.append(model.X_cumValExplVar())
                    
                    # Get Y loadings and scores regression coefficients
                    # for computation Y pred.
                    arrQ = model.Y_loadings()
                    arrC = model.scoresRegressionCoeffs()
                    
                    # Get Y socres, Y loadings and Y correlation loadings for 
                    # the chosen number of components in this orthogonalised 
                    # X block
                    self.YscoresList.append(model.Y_scores())
                    self.YloadingsList.append(model.Y_loadings())
                    self.YcorrLoadingsList.append(model.Y_corrLoadings())
                    
                    # Get Y pred from calibration. This Y is processed and
                    # needs to be 'un-processed' (un-center, un-standardise).
                    # But prior to that the cumulative Y needs to be computed
                    YpredCal_proc = model.Y_predCal()[self.XcompsList_input[indMod]]
                    YprocPredList.append(YpredCal_proc)
                    
                    sumYpredCal_proc = np.sum(YprocPredList, axis=0)
                    
                    if self.Ystand == True:
                        YpredCal = (sumYpredCal_proc * self.Y_std) + \
                                self.Y_mean
                    else:
                        YpredCal = sumYpredCal_proc + self.Y_mean
                    self.YcumPredCalList.append(YpredCal)
                    

#==============================================================================
# From here computations of CALBIRATED explained variance in Y
#==============================================================================            
            
            # Now compute global explained variance in Y after each block
            # Construct a list that holds cumulative calibrated explained 
            # variance in Y. This is done for individual variables in Y as 
            # well as the whole Y block across all individual variable at
            # once.
                                    
            # First compute global PRSEE for zero components
            Y_cent = self.arrY_input - np.average(self.arrY_input, axis=0)
            
            Y_PRESSE_indVar_list = []
            Y_PRESSE_0_indVar = np.sum(np.square(Y_cent), axis=0)
            Y_PRESSE_indVar_list.append(Y_PRESSE_0_indVar)
            
            for ind, Yhat in enumerate(self.YcumPredCalList):
                diffY = st.centre(self.arrY_input) - st.centre(Yhat)
                Y_PRESSE_indVar = np.sum(np.square(diffY), axis=0)
                Y_PRESSE_indVar_list.append(Y_PRESSE_indVar)
            
            self.Y_PRESSEarr_indVar = np.vstack(Y_PRESSE_indVar_list)
                        
            self.Y_MSEEarr_indVar = self.Y_PRESSEarr_indVar / \
                np.shape(self.arrY_input)[0]
            self.Y_RMSEEarr_indVar = np.sqrt(self.Y_MSEEarr_indVar)
            # -----------------------------------------------------------------
        
            # -----------------------------------------------------------------
            # Compute explained global explained variance for each variable
            # in Y using MSEE for each variable. Also collect PRESSE, MSEE, 
            # RMSEE in their respective dictionaries for each variable. Keys 
            # represent now variables and NOT components as above with 
            # self.PRESSEdict_indVar
            self.YcumCalExplVar_indVar = np.zeros(np.shape(self.Y_MSEEarr_indVar))
            Y_MSEE_0_indVar = self.Y_MSEEarr_indVar[0,:]
            
            for ind, Y_MSEE_indVar in enumerate(self.Y_MSEEarr_indVar):
                explVar = (Y_MSEE_0_indVar - Y_MSEE_indVar) / \
                        Y_MSEE_0_indVar * 100
                self.YcumCalExplVar_indVar[ind] = explVar
            # -----------------------------------------------------------------
            
            # -----------------------------------------------------------------
            # Collect total PRESSE across all variables in a dictionary. Also,
            # compute total calibrated explained variance in Y.
            self.Y_PRESSEarr = np.sum(self.Y_PRESSEarr_indVar, axis=1)
            self.Y_MSEEarr = self.Y_PRESSEarr / np.shape(self.arrY_input)[1]
            self.Y_RMSEEarr = np.sqrt(self.Y_MSEEarr)
            
            self.YcumCalExplVar = []
            Y_MSEE_0 = self.Y_MSEEarr[0]
            
            for ind, Y_MSEE in enumerate(self.Y_MSEEarr):
                explVar = (Y_MSEE_0 - Y_MSEE) / Y_MSEE_0 * 100
                self.YcumCalExplVar.append(explVar)
            # ----------------------------------------------------------------- 

#==============================================================================
#         From here SO-PLS algorithm starts (VALIDATION)
#==============================================================================

            
            # For all other cases where more than one X block has components 
            # provided.
            # --------------------------------------------------------------         
        
            # First devide into combinations of training and test sets
            numObj = np.shape(self.arrY_input)[0]
            
            if self.cvType[0] == "loo":
                print("loo")
                cvComb = cv.LeaveOneOut(numObj)
            elif self.cvType[0] == "lpo":
                print("lpo")
                cvComb = cv.LeavePOut(numObj, self.cvType[1])
            elif self.cvType[0] == "lolo":
                print("lolo")
                cvComb = cv.LeaveOneLabelOut(self.cvType[1])
            else:
                print('Requested form of cross validation is not available')
                pass
                        
            # Collect all predicted Y from cross validation in a list. These
            # predicted Y are processed and need to be un-processed.
            tCQlistsAll = []
            self.Y_train_mean_segList = []
            self.Y_train_std_segList = []
            
            # ----- HERE STARTS THE CROSS VALIDATION LOOP -----
            # First devide into combinations of training and test sets
            for train_index, test_index in cvComb:
                
                # Define training and test set for Y
                Y_train, Y_test = cv.split(train_index, test_index, \
                        self.arrY_input)
                
                # -----------------------------------------------------------------                    
                # Center or standardise Y according to users choice      
                if self.Ystand == True:
                    # Standardise training set Y using mean and STD
                    Y_train_mean = np.average(Y_train, axis=0).reshape(1,-1)
                    Y_train_std = np.std(Y_train, axis=0, ddof=1).reshape(1,-1)
                    Y_train_proc = (Y_train - Y_train_mean) / Y_train_std
                    
                    self.Y_train_mean_segList.append(Y_train_mean)
                    self.Y_train_std_segList.append(Y_train_std)
                    
                    # Standardise test set Y using mean and STD from 
                    # training set
                    Y_test_proc = (Y_test - Y_train_mean) / Y_train_std
                
                else:
                    # Centre training set Y using mean
                    Y_train_mean = np.average(Y_train, axis=0).reshape(1,-1)
                    Y_train_proc = Y_train - Y_train_mean
                    Y_train_std = None
                    
                    self.Y_train_mean_segList.append(Y_train_mean)
                    self.Y_train_std_segList.append(Y_train_std)
                    
                    # Centre test set Y using mean and STD from 
                    # training set
                    Y_test_proc = Y_test - Y_train_mean
                # -----------------------------------------------------------------   
                
                # Now do the same for the X blocks. Do this by iterating 
                # through self.XblocksList_input and take out test and 
                # training data
                Xblocks_trainDict = {}
                Xblocks_testDict = {}
                
                Xblocks_trainList = []
                Xblocks_testList = []
                
                Xblocks_train_meanList = []
                Xblocks_train_stdList = []
                Xblocks_train_procList = []
                
                Xblocks_test_procList = []
                                                
                # For each X block split up into training and test set for 
                # the cross validation segment we are in right now
                for ind, item in enumerate(self.XblocksList_input):
                    
                    X_train, X_test = cv.split(train_index, test_index, item)
                    Xblocks_trainDict[ind] = X_train
                    Xblocks_testDict[ind] = X_test
                    
                    Xblocks_trainList.append(X_train)
                    Xblocks_testList.append(X_test)
                    
                    
                    # -------------------------------------------------------------
                    # Center or standardise X blocks according to users choice
                    if self.XstandList[ind] == True:
                        # Compute standardised X blocks using mean and STD
                        X_train_mean = np.average(X_train, axis=0).reshape(1,-1)
                        X_train_std = np.std(X_train, axis=0, ddof=1).reshape(1,-1)
                        X_train_proc = (X_train - X_train_mean) / X_train_std
                        
                        # Append each standardised X block to X blocks training
                        # list
                        Xblocks_train_meanList.append(X_train_mean)
                        Xblocks_train_stdList.append(X_train_std)
                        Xblocks_train_procList.append(X_train_proc)
                        
                        # Standardise test set of each X block using mean and 
                        # and STD from training X blocks and append to X blocks 
                        # test list
                        X_test_proc = (X_test - X_train_mean) / X_train_std                        
                        Xblocks_test_procList.append(X_test_proc)
     
                    else:
                        # Compute centred X blocks using mean
                        X_train_mean = np.average(X_train, axis=0).reshape(1,-1)
                        X_train_proc = X_train - X_train_mean
                        
                        # Append each centred X block to X blocks training list
                        Xblocks_train_meanList.append(X_train_mean.reshape(1,-1))
                        Xblocks_train_stdList.append(None)
                        Xblocks_train_procList.append(X_train_proc)
                        
                        # Centre test set of each X block using mean from 
                        # X block training set and append to X blocks test list
                        X_test_proc = X_test - X_train_mean                        
                        Xblocks_test_procList.append(X_test_proc)
                    # -------------------------------------------------------------
                    
                                    
                # Put all training and test data for Y and X blocks in a 
                # dictionary. 
                segDict = {}
                segDict['Y train'] = Y_train
                segDict['Y test'] = Y_test
                segDict['X train'] = Xblocks_trainDict
                segDict['X test'] = Xblocks_testDict
                
                segDict['proc Y train'] = Y_train_proc
                segDict['Y train mean'] = Y_train_mean.reshape(1,-1)
                #segDict['Y train std'] = Y_train_std
                
                segDict['proc X train'] = Xblocks_train_procList
                segDict['X train mean'] = Xblocks_train_meanList
                
                segDict['proc X test'] = Xblocks_test_procList
                segDict['proc Y test'] = Y_test_proc
                
                
                # Now start modelling sequential PLSR over all X blocks.
                # First X block with non-zero X components will be modelled
                # with ordinary PLSR. For all following X blocks with non-
                # zero components, the X block must be orthogonalised with
                # regard to the X scores of the prior X block. Only then the
                # orthogonalised X block can be modelled against Y.
                T_train_list = []
                T_test_list = []
                
                Blist = []
                orthoXblockList = []
                Wlist = []
                
                YmeanList = []
                
                Qlist = []
                Clist = []
                tCQlist = []
                                                
                for indMod, Xblock in enumerate(Xblocks_train_procList):
                    
                    if indMod not in comb_nonZeroIndexArr:
                        continue
                    
                    
                    if indMod == comb_nonZeroIndexArr[0]:
                        
                        # Do ordinary PLSR prior to PLRS with orth. Xblocks
                        model = PLS(Xblock, Y_train_proc, \
                                numPC=self.XcompsList_input[indMod])
                        
                        # Get X scores and store them in a scores list. The
                        # scores are needed for the orthogonlisation step of 
                        # the next X block. 
                        T_train_list.append(model.X_scores())
                        
                        # Here prediction part starts.
                        # First estimate X scores for test set of X blocks
                        arrW = model.X_loadingsWeights()
                        arrP = model.X_loadings()
                        
                        projTlist = []
                        X_test_deflationList = [Xblocks_test_procList[indMod]]
                        
                        for predInd in range(self.XcompsList_input[indMod]):
    
                            projT = np.dot(X_test_deflationList[-1], \
                                    arrW[:,predInd])
                            projTlist.append(projT)
                            
                            deflated_Xblock_test = X_test_deflationList[-1] - \
                                    np.dot(projT.reshape(1,-1), \
                                    np.transpose(arrP[:,predInd]).reshape(1,-1))
                            X_test_deflationList.append(deflated_Xblock_test)                    
                        
                        # Construct array which holds projected X test scores.
                        T_test = np.transpose(np.array(projTlist))
                        
                        T_test_list.append(T_test)
                        
                        # Get Y loadings and scores regression coefficients
                        # for computation Y pred.
                        arrQ = model.Y_loadings()
                        arrC = model.scoresRegressionCoeffs()
                        
                        # Now compute Ypred for the test set of the acutal
                        # X block. The mean This Ypred is processed and will
                        # be un-processed later after summarising across all
                        # Y pred from each X block.
                        tCQ = np.dot(T_test, np.dot(arrC, np.transpose(arrQ)))
                        tCQlist.append(tCQ)
                        
                        YmeanList.append(model.Y_means())
                        
                        Wlist.append(arrW)
                        Qlist.append(arrQ)
                        Clist.append(arrC)
                    
                    else:
                        # Orthogonalise and run PLSE and model Y. Could also 
                        # use residuals from previous model, which would give 
                        # the same result. 
                        
                        # Stack X scores horizontally from previous PLS models
                        # and orthogonalise next X block with regard to the
                        # stacked X scores. If there is only one set of scores
                        # then stacking is not necessary
                        
                        if len(T_train_list) == 1:
                            T = T_train_list[0]
                            cv_T = T_test_list[0]
                        
                        else:
                            T = np.hstack(T_train_list)
                            cv_T = np.hstack(T_test_list)
                        
                        
                        # Orthogonalisation process
                        # X_orth = X - TB
                        B = np.dot(np.dot(nl.inv(np.dot(np.transpose(T), T)), \
                                np.transpose(T)), Xblock)
                        orth_Xblock = Xblock - np.dot(T, B)
                        Blist.append(B)
                        orthoXblockList.append(orth_Xblock)
                        
                        # Run PLSR on orthogonalised X block. 
                        model = PLS(orth_Xblock, Y_train_proc, \
                                numPC=self.XcompsList_input[indMod])
                        T_train_list.append(model.X_scores())
                        
                        # Orthogonalisation of test set of X block
                        orth_Xblock_test = Xblocks_test_procList[indMod] - \
                                np.dot(cv_T, B)
                        
                        # Here the prediction part starts.
                        # First estimate X scores for test set of X blocks
                        arrW = model.X_loadingsWeights()
                        arrP = model.X_loadings()
                        
                        projTlist = []
                        X_test_deflationList = [orth_Xblock_test]
                        
                        for predInd in range(self.XcompsList_input[indMod]):
                            
                            projT = np.dot(X_test_deflationList[-1], \
                                    arrW[:,predInd])
                            projTlist.append(projT)
                            
                            deflated_Xblock_test = X_test_deflationList[-1] - \
                                    np.dot(projT.reshape(1,-1), \
                                    np.transpose(arrP[:,predInd]).reshape(1,-1))
                            X_test_deflationList.append(deflated_Xblock_test)
                        
                        
                        # Construct array which holds projected X test scores.
                        T_test = np.transpose(np.array(projTlist))
                        T_test_list.append(T_test)
                        
                        # Get Y loadings and scores regression coefficients
                        # for computation Y pred.
                        arrQ = model.Y_loadings()
                        arrC = model.scoresRegressionCoeffs()
                        
                        # Now compute Ypred for the test set of the acutal
                        # X block. The mean This Ypred is processed and will
                        # be un-processed later after summarising across all
                        # Y pred from each X block.
                        tCQ = np.dot(T_test, np.dot(arrC, np.transpose(arrQ)))
                        tCQlist.append(tCQ)
                        
                        Wlist.append(arrW)
                        Qlist.append(arrQ)
                        Clist.append(arrC)
                        
                
                # Collect all predicted Y test after each X block from cross  
                # validation for each cross validation segment. These Y pred 
                # are processed (need to un-centered/un-standardised).
                tCQlistsAll.append(tCQlist)
                                
            
            # Now stack predicted Y tests from cross validation and from all
            # segments and un-process.
            #for indX, Xblock in enumerate(Xblocks_train_procList):
            for indX in range(len(comb_nonZeroIndexArr)):
                
                # Summarise Y pred test across X blocks.
                YpredVal_test_list_proc = []
                for objInd, objList in enumerate(tCQlistsAll):
                    
                    sumYpredVal_test_proc = np.sum(objList[0:indX+1], axis=0)                    
                    YpredVal_test_list_proc.append(sumYpredVal_test_proc)
                    
                YpredVal_test_proc = np.vstack(YpredVal_test_list_proc)                    
                if self.Ystand == True:
                    YpredVal_train = (YpredVal_test_proc * Y_train_std) + \
                            Y_train_mean
            
                else:
                    YpredVal_train = YpredVal_test_proc + Y_train_mean
                
                self.YcumPredValList.append(YpredVal_train)

#==============================================================================
# From here computations of VALIDATED explained variance in Y
#==============================================================================            
            
            # Now compute global explained variance in Y after each block
            # Construct a list that holds cumulative calibrated explained 
            # variance in Y. This is done for individual variables in Y as 
            # well as the whole Y block across all individual variable at
            # once.
            
            # First compute global PRSECV for zero components
            Y_cent = self.arrY_input - np.average(self.arrY_input, axis=0)
            
            Y_PRESSCV_indVar_list = []
            Y_PRESSCV_0_indVar = np.sum(np.square(Y_cent), axis=0)
            Y_PRESSCV_indVar_list.append(Y_PRESSCV_0_indVar)
            
            for ind, Yhat in enumerate(self.YcumPredValList):
                diffY = st.centre(self.arrY_input) - st.centre(Yhat)
                Y_PRESSCV_indVar = np.sum(np.square(diffY), axis=0)
                Y_PRESSCV_indVar_list.append(Y_PRESSCV_indVar)
            
            self.Y_PRESSCVarr_indVar = np.vstack(Y_PRESSCV_indVar_list)            
            
            self.Y_MSECVarr_indVar = self.Y_PRESSCVarr_indVar / \
                np.shape(self.arrY_input)[0]
            self.Y_RMSECVarr_indVar = np.sqrt(self.Y_MSECVarr_indVar)
            # -----------------------------------------------------------------
        
            # -----------------------------------------------------------------
            # Compute explained global explained variance for each variable
            # in Y using MSECV for each variable. Also collect PRESSE, MSEE, 
            # RMSEE in their respective dictionaries for each variable. Keys 
            # represent now variables and NOT components as above with 
            # self.PRESSEdict_indVar
            self.YcumValExplVar_indVar = np.zeros(np.shape(self.Y_MSECVarr_indVar))
            Y_MSECV_0_indVar = self.Y_MSECVarr_indVar[0,:]
            
            for ind, Y_MSECV_indVar in enumerate(self.Y_MSECVarr_indVar):
                explVar = (Y_MSECV_0_indVar - Y_MSECV_indVar) / \
                        Y_MSECV_0_indVar * 100
                self.YcumValExplVar_indVar[ind] = explVar
            # -----------------------------------------------------------------
            
            # -----------------------------------------------------------------
            # Collect total PRESSCV across all variables in a dictionary. Also,
            # compute total calibrated explained variance in Y.
            self.Y_PRESSCVarr = np.sum(self.Y_PRESSCVarr_indVar, axis=1)
            self.Y_MSECVarr = self.Y_PRESSCVarr / np.shape(self.arrY_input)[1]
            self.Y_RMSECVarr = np.sqrt(self.Y_MSECVarr)
            
            self.YcumValExplVar = []
            Y_MSECV_0 = self.Y_MSECVarr[0]
            
            for ind, Y_MSECV in enumerate(self.Y_MSECVarr):
                explVar = (Y_MSECV_0 - Y_MSECV) / Y_MSECV_0 * 100
                self.YcumValExplVar.append(explVar)
            # -----------------------------------------------------------------
            
            
#==============================================================================
# From here PCP procedure starts
# Compute PCA on Yhat (after last X block) and compute projected scores from
# previous X blocks
#==============================================================================
        
        # First get Yhat based on predicitons of all X blocks           
        self.Yhat = self.YcumPredCalList[-1]
        
        if len(comb_nonZeroIndexArr) == 1:
            self.X = model.modelSettings()['analysed X']
        
        else:
            self.X = np.hstack(self.Xblocks_procList)
        
        # Centre Yhat before PCP. User gets no choice here        
        self.Yhat_mean = np.average(self.Yhat, axis=0).reshape(1,-1)
        self.Yhat_proc = self.Yhat - self.Yhat_mean
        
        # Run PCA on Yhat after last X block in model. 
        pcaMod = pca.nipalsPCA(self.Yhat, stand=False)
        self.PCP_C = pcaMod.X_loadings()
        
        # If number of variables in Y is larger than sum of variables across
        # all X blocks, then extract as many columns in 
        # PCP scores as equals total number of components chosen across
        # all X blocks.
        if np.shape(self.Yhat)[1] > np.sum(self.XcompsList): 
            self.PCP_T = pcaMod.X_scores()[:,0:np.sum(self.XcompsList)]
        else:
            self.PCP_T = pcaMod.X_scores()
        
        self.PCP_P = np.dot(np.transpose(self.X), self.PCP_T)
        self.PCP_calExplVar = pcaMod.X_calExplVar()
        
        
        # PCP X loadings hold loadings from all X blocks (concatenated). 
        # This needs to bes plit up into PCP X loadings from separate 
        # X blocks
        # First construct a list that hold number of variables in each
        # X block.
        XblockVariableDimList = []
        for blo in XblocksList:
            XblockVariableDimList.append(np.shape(blo)[1])
        
        cumXblockVariableDimList = np.cumsum(XblockVariableDimList)
        
        # Now split loadings according to specified positions
        self.PCP_splitP = np.split(self.PCP_P, cumXblockVariableDimList)
        self.PCP_splitP.pop(-1)
                                            
        # Now project Yhat after each X block (except last one) into PCA
        # space.
        self.YpredProjScoresList = []
        tempYcumPredList = self.YcumPredCalList[:]
        tempYcumPredList.pop(-1)
        
        for arr in tempYcumPredList:            
            cent_arr = st.centre(arr)
            projYpredScores = np.dot(cent_arr, self.PCP_C)
            self.YpredProjScoresList.append(projYpredScores)
                
        # Finally, append scores from PCA on Yhat to list
        self.YpredProjScoresList.append(self.PCP_T)

    
    
    def X_means(self):
        """
        Returns a list containing column means of each X block. The order of 
        columns means is the same as the order in which the X blocks were 
        submitted to the SO-PLS class.
        """
        return self.Xblocks_meanList
    
    
    def X_scores(self):
        """
        Returns a list with arrays holding X scores of each X block. The order  
        of X scores corresponds to the order in which the X blocks were 
        submitted to the SO-PLS class.
        """
        return self.XscoresList
    
    
    def X_loadings(self):
        """
        Returns a list with arrays holding X loadings of each X block. The  
        order of X loadings corresponds to the order in which the X blocks were 
        submitted to the SO-PLS class.
        """
        return self.XloadingsList
    
    
    def X_corrLoadings(self):
        """
        Returns a list with arrays holding X correlation loadings of each X 
        block. The order of X loadings corresponds to the order in which the 
        X blocks were submitted to the SO-PLS class.
        """
        return self.XcorrLoadingsList
    
    
    def X_cumCalExplVar_indVar(self):
        """
        Returns a list with arrays holding total cumulative explained varinace 
        in each X block.
        """
        return self.XcumCalExplVars_indVar
    
    
    def X_cumCalExplVar(self):
        """
        Returns a list with arrays holding total cumulative explained varinace 
        in each X block.
        """
        return self.XcumCalExplVars
    
    
    def X_predCals(self):
        """
        Returns a list holding predicted X from each modelled X bock.
        """
        return self.XpredCalList
    
    
    def X_cumValExplVar_indVar(self):
        """
        Returns a list with arrays holding total cumulative explained varinace 
        in each X block.
        """
        return self.XcumValExplVars_indVar
    
    
    def X_cumValExplVar(self):
        """
        Returns a list with arrays holding total cumulative explained varinace 
        in each X block.
        """
        return self.XcumValExplVars
    
    
    def X_predVals(self):
        """
        Returns a list holding predicted X from each modelled X bock.
        """
        return self.XpredValList
            
    
    def Y_means(self):
        """
        Returns array holidng column means of Y block.
        """
        return self.Y_mean
    
    
    def Y_scores(self):
        """
        Returns a list with arrays holding Y scores after modelling each X
        block. The order of Y scores corresponds with the order in which the
        X blocks were submitted to the SO.PLS class.
        """
        return self.YscoresList
    
    
    def Y_loadings(self):
        """
        Returns a list with arrays holding Y loadings after modelling each X
        block. The order of Y loadings corresponds with the order in which the
        X blocks were submitted to the SO.PLS class.
        """
        return self.YloadingsList
    
    
    def Y_corrLoadings(self):
        """
        Returns a list with arrays holding Y correlation loadings after 
        modelling each X block. The order of Y loadings corresponds with the 
        order in which the X blocks were submitted to the SO.PLS class.
        """
        return self.YcorrLoadingsList
    
    
    def Y_cumPredCal(self):
        """
        Returns a list holding accumulated predicted Y across predicted Y from
        each modelled X block.
        """
        return self.YcumPredCalList
    
    
    def Y_cumCalExplVar_indVar(self):
        """
        Returns a list holding total cumulative explained variance in Y after
        modelling each X block.
        """
        return self.YcumCalExplVar_indVar
    
    
    def Y_cumCalExplVar(self):
        """
        Returns a list holding total cumulative explained variance in Y after
        modelling each X block.
        """
        return self.YcumCalExplVar
    
    
    def Y_cumPredVal(self):
        """
        Returns a list holding accumulated predicted Y across predicted Y from
        each modelled X block.
        """
        return self.YcumPredValList
        #return self.cumYpredValList
    
    
    def Y_cumValExplVar_indVar(self):
        """
        Returns a list holding total cumulative explained variance in Y after
        modelling each X block.
        """
        return self.YcumValExplVar_indVar
    
    
    def Y_cumValExplVar(self):
        """
        Returns a list holding total cumulative explained variance in Y after
        modelling each X block.
        """
        return self.YcumValExplVar
    
    
    def PCPscores(self):
        """
        Returns array holding scores from PCP of Yhat.
        """
        return self.PCP_T
    
    
    def PCPXloadings(self):
        """
        Returns array holding X loadings from PCP of Yhat.
        """
        return self.PCP_splitP
    
    
    def PCPYloadings(self):
        """
        Return array holding Y loadings from PCP of Yhat.
        """
        return self.PCP_C
    
    
    def PCPexplVar(self):
        """
        Returns array holding explained variance in Y from PCP of Yhat.
        """
        return self.PCP_calExplVar
    
    
    def projectedYpredScores(self):
        """
        Returns PCA scores on Ypred after last X block (last entry in the 
        list). In addition, returns projected PCA scores from Ypred after 
        previous X blocks in SP-PLS model (chronological order).
        """
        return self.YpredProjScoresList
    
    
    def orthoXblocks(self):
        """
        Returns a list with orthogonalised X blocks in cronological order. 
        Does not include very first X block in order of X blocks, since first
        X block is not orthogonalised.
        """
        return self.orthoXblockList
    
    
    def modelSettings(self):
        """
        Returns a dictionary holding the most important settings under which
        the SO-PLS algorithm was run.
        """
        res = {}
        
        res['Y block input'] = self.arrY_input
        res['proc Y'] = self.Y_proc
        res['Y stand'] = self.Ystand
        
        res['X blocks input'] = self.XblocksList_input
        res['proc X'] = self.Xblocks_procList
        res['X comp list'] = self.XcompsList
        res['X stand list'] = self.XstandList
        
        return res
    
    


def plotSOPLS(model, objNames, YvarNames, XblockVarNames, XblockNames):
    
#==============================================================================
# Plot PCP scores T
#==============================================================================
    
    PCPscores = model.PCPscores()
    PCPexplVar = model.PCPexplVar()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    
    # Loop through all coordinates (PC1,PC2) and names to plot scores.
    for ind, objName in enumerate(objNames):
        
        ax.scatter(PCPscores[ind,0], PCPscores[ind,1], s=10, c='w', \
            marker='o', edgecolor='grey')
        ax.text(PCPscores[ind,0], PCPscores[ind,1], objName, fontsize=10)
    
    
    # Find maximum and minimum scores along PC1 and PC2
    xMax = max(PCPscores[:,0])
    xMin = min(PCPscores[:,0])
    
    yMax = max(PCPscores[:,1])
    yMin = min(PCPscores[:,1])
    
    
    # Set limits for lines representing the axes.
    # x-axis
    if abs(xMax) >= abs(xMin):
        extraX = xMax * .4
        limX = xMax * .3
    
    elif abs(xMax) < abs(xMin):
        extraX = abs(xMin) * .4
        limX = abs(xMin) * .3
    
    # y-axis
    if abs(yMax) >= abs(yMin):
        extraY = yMax * .4
        limY = yMax * .3
    
    elif abs(yMax) < abs(yMin):
        extraY = abs(yMin) * .4
        limY = abs(yMin) * .3
    
    
    xMaxLine = xMax + extraX
    xMinLine = xMin - extraX
    
    yMaxLine = yMax + extraY
    yMinLine = yMin - extraY
    
    
    ax.plot([0,0], [yMaxLine,yMinLine], color='0.4', linestyle='dashed', \
                    linewidth=1)
    ax.plot([xMinLine,xMaxLine], [0,0], color='0.4', linestyle='dashed', \
                    linewidth=1)
    
    
    # Set limits for plot regions.
    xMaxLim = xMax + limX
    xMinLim = xMin - limX
    
    yMaxLim = yMax + limY
    yMinLim = yMin - limY
    
    ax.set_xlim(xMinLim,xMaxLim)
    ax.set_ylim(yMinLim,yMaxLim)
    
    
    # Plot title, axis names. 
    ax.set_xlabel('PC1 ({0}%)'.format(str(round(PCPexplVar[0],1))))
    ax.set_ylabel('PC2 ({0}%)'.format(str(round(PCPexplVar[1],1))))
    
    ax.set_title('PCP scores plot')
    
    plt.show()
        
    
#==============================================================================
# Plot PCP Y loadings C
#==============================================================================
    
    PCPYloadings = model.PCPYloadings()    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Loop through all coordinates (PC1,PC2) and names to plot scores.
    for ind, varName in enumerate(YvarNames):
        
        ax.scatter(PCPYloadings[ind,0], PCPYloadings[ind,1], s=10, c='w', \
            marker='o', edgecolor='grey')
        
        ax.text(PCPYloadings[ind,0], PCPYloadings[ind,1], varName, fontsize=10)
    
    
    # Find maximum and minimum scores along PC1 and PC2
    xMax = max(PCPYloadings[:,0])
    xMin = min(PCPYloadings[:,0])
    
    yMax = max(PCPYloadings[:,1])
    yMin = min(PCPYloadings[:,1])
    
    
    # Set limits for lines representing the axes.
    # x-axis
    if abs(xMax) >= abs(xMin):
        extraX = xMax * .4
        limX = xMax * .3
    
    elif abs(xMax) < abs(xMin):
        extraX = abs(xMin) * .4
        limX = abs(xMin) * .3
    
    # y-axis
    if abs(yMax) >= abs(yMin):
        extraY = yMax * .4
        limY = yMax * .3
    
    elif abs(yMax) < abs(yMin):
        extraY = abs(yMin) * .4
        limY = abs(yMin) * .3
    
    
    xMaxLine = xMax + extraX
    xMinLine = xMin - extraX
    
    yMaxLine = yMax + extraY
    yMinLine = yMin - extraY
    
    
    ax.plot([0,0], [yMaxLine,yMinLine], color='0.4', linestyle='dashed', \
                    linewidth=1)
    ax.plot([xMinLine,xMaxLine], [0,0], color='0.4', linestyle='dashed', \
                    linewidth=1)
    
    
    # Set limits for plot regions.
    xMaxLim = xMax + limX
    xMinLim = xMin - limX
    
    yMaxLim = yMax + limY
    yMinLim = yMin - limY
    
    ax.set_xlim(xMinLim,xMaxLim)
    ax.set_ylim(yMinLim,yMaxLim)
    
    
    # Plot title, axis names. 
    ax.set_xlabel('PC1 ({0}%)'.format(str(round(PCPexplVar[0],1))))
    ax.set_ylabel('PC2 ({0}%)'.format(str(round(PCPexplVar[1],1))))
    
    ax.set_title('PCP Y-loadings plot')
    
    plt.show()
    
    
#==============================================================================
# Plot PCP X loadings P as line plots (mostly for spectra)
#==============================================================================
   
#    # Access PCP X loadings from SP-PLS model    
#    PCPXloadings = model.PCPXloadings()
#    Xdims = model.modelSettings()['X comp list']
#    nonZeroCompIndex = np.flatnonzero(np.array(Xdims))
#    print 'NON', nonZeroCompIndex
#    
#    colourList = ['r', 'k', 'b', 'g', 'm']
#    lineStyleList = ['-', '--', '-.', '-', ':']
#    
#    # Loop through all X block PCP loadings    
#    for ind, block in enumerate(PCPXloadings):
#        print np.shape(block), XblockNames[ind]
#        
#        if Xdims[ind] == 0:
#            continue
#        
#        else:       
#            fig = plt.figure()
#            ax = fig.add_subplot(111)
#            
#            # For each X block plot PCP X loadings for every computed component         
#            for load in range(np.shape(block)[1]):
#                lineLabel = 'PC' + str(load + 1)
#                ax.plot(block[:,load], color=colourList[load], \
#                        linestyle=lineStyleList[load], label=lineLabel)    
#            
#            titleName = 'PCP X-loadings plot: ' + XblockNames[ind]
#            ax.set_title(titleName)
#            
#            # Setting limits for x axis, which depends on numer of variables
#            xmax = np.shape(block)[0]
#            xmin = 0
#            
#            # Plot X loadings as lines        
#            ax.plot([xmax,xmin], [0,0], color='0.4', linestyle='dashed', \
#                        linewidth=1)
#            ax.set_xlim(xmin,xmax)
#            
#            # Legend
#            plt.legend(loc='lower left', shadow=True, labelspacing=.1)
#            ltext = plt.gca().get_legend().get_texts()
#            plt.setp(ltext[0], fontsize = 10, color = 'k')
#                    
#            plt.show()


#==============================================================================
# Plot PCP X loadings P as scatter plots (mostly for other than spectra)
#==============================================================================
    # Access PCP X loadings from SP-PLS model    
    PCPXloadings = model.PCPXloadings()
    Xdims = model.modelSettings()['X comp list']
    nonZeroCompIndex = np.flatnonzero(np.array(Xdims))
    #print('NON', nonZeroCompIndex)
    
#    dotStyleList = ['rs', 'go', 'kD', 'bp', 'mv']
    cList = ['r', 'g', 'm', 'b', 'y']
    markerList = ['s', 'o', 'D', 'p', 'v']
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Loop through all X block PCP loadings    
    for ind, block in enumerate(PCPXloadings):
        #print(np.shape(block), XblockNames[ind])
        
        if Xdims[ind] == 0:
            continue
        
        else:               
            # For each X block plot PCP X loadings for every computed component         
            for varNum in range(np.shape(block)[0]):
                
                dotLabel = XblockNames[ind]
                #print('*****', ind, varNum)
                if varNum == 0:
                    ax.scatter(PCPXloadings[ind][varNum,0], \
                            PCPXloadings[ind][varNum,1], s=20, \
                            c=cList[ind], marker=markerList[ind], 
                            edgecolor='grey', label=dotLabel)
                else:
                   ax.scatter(PCPXloadings[ind][varNum,0], \
                            PCPXloadings[ind][varNum,1], s=20, \
                            c=cList[ind], marker=markerList[ind], 
                            edgecolor='grey') 
                ax.text(PCPXloadings[ind][varNum,0], PCPXloadings[ind][varNum,1],\
                        XblockVarNames[ind][varNum], fontsize=10)
    
    # Stack horizontally PCP X loadings from each block and then look for
    # max and min values to find right scale for plot
    allPCPXloadings = np.vstack(PCPXloadings)
    
    # Find maximum and minimum scores along PC1 and PC2
    xMax = max(allPCPXloadings[:,0])
    xMin = min(allPCPXloadings[:,0])
    
    yMax = max(allPCPXloadings[:,1])
    yMin = min(allPCPXloadings[:,1])
    
    
    # Set limits for lines representing the axes.
    # x-axis
    if abs(xMax) >= abs(xMin):
        extraX = xMax * .4
        limX = xMax * .3
    
    elif abs(xMax) < abs(xMin):
        extraX = abs(xMin) * .4
        limX = abs(xMin) * .3
    
    # y-axis
    if abs(yMax) >= abs(yMin):
        extraY = yMax * .4
        limY = yMax * .3
    
    elif abs(yMax) < abs(yMin):
        extraY = abs(yMin) * .4
        limY = abs(yMin) * .3
    
    
    xMaxLine = xMax + extraX
    xMinLine = xMin - extraX
    
    yMaxLine = yMax + extraY
    yMinLine = yMin - extraY
    
    
    ax.plot([0,0], [yMaxLine,yMinLine], color='0.4', linestyle='dashed', \
                    linewidth=1)
    ax.plot([xMinLine,xMaxLine], [0,0], color='0.4', linestyle='dashed', \
                    linewidth=1)
    
    
    # Set limits for plot regions.
    xMaxLim = xMax + limX
    xMinLim = xMin - limX
    
    yMaxLim = yMax + limY
    yMinLim = yMin - limY
    
    ax.set_xlim(xMinLim,xMaxLim)
    ax.set_ylim(yMinLim,yMaxLim)
    
    
    # Plot title, axis names. 
    ax.set_xlabel('PC1 ({0}%)'.format(str(round(PCPexplVar[0],1))))
    ax.set_ylabel('PC2 ({0}%)'.format(str(round(PCPexplVar[1],1))))
    
    ax.set_title('PCP X-loadings plot')
    plt.legend(loc='lower left', shadow=True, labelspacing=.1)
    
    plt.show()
        
    
    
#==============================================================================
# Plot global explained variance in Y
#==============================================================================    
    
    calGlobExplVar = model.Y_cumCalExplVar()    
    valGlobExplVar = model.Y_cumValExplVar() 
    
    # Plot global explained variance
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Set axis limits for a pretty plot
    left = -0.2; right = len(XblockNames) + 0.5; top = 105; bottom = -5
    
    
    # Construct positions for ticks along x-axis.
    xPosCal = range(len(XblockNames)+1)
#    if len(nonZeroCompIndex) == 1:
#        xPosCal = range(2)
#    else:
#        xPosCal = range(len(XblockNames)+1)
    
    tickPos_XblockNames = XblockNames[:]
    tickPos_XblockNames.insert(0, 'none')
    
    print('xPosCal', len(xPosCal), xPosCal); print; print
    print('calibrated globExplVar', calGlobExplVar); print; print
    print('validated globExplVar', valGlobExplVar); print; print
    
    # Extend list of cumulative explained variances with zeros for each 
    # X block without components. This needs to be done for plotting
    # function
    xCompList = model.modelSettings()['X comp list']
    if 0 in xCompList:
        
        for index, value in enumerate(xCompList):
            #print(index, value)
            
            if value == 0:
                calPriorValue = calGlobExplVar[index]
                calGlobExplVar.insert(index+1, calPriorValue)
                
                valPriorValue = valGlobExplVar[index]
                valGlobExplVar.insert(index+1, valPriorValue)                

    
    # Do the plotting and set the ticks on x-axis with corresponding name.
    ax.plot(xPosCal, calGlobExplVar, color='b', linestyle='solid', \
            linewidth=1, label='calibrated')
    ax.plot(xPosCal, valGlobExplVar, color='r', linestyle='solid', \
            linewidth=1, label='validated')
    ax.set_xticks(xPosCal)
    
    # Set lables for x-ticks.
    #labels = ax.set_xticklabels(XblockNames, rotation=0, ha='center')
    ax.set_xticklabels(tickPos_XblockNames, rotation=0, ha='center')
    
    ax.set_ylabel('Explained variance [%]')
    
    ax.set_xlim(left,right)
    ax.set_ylim(bottom,top)
    ax.set_title('Global explained variance in Y')
    
    plt.legend(loc='lower right', shadow=True, labelspacing=.1)
    ltext = plt.gca().get_legend().get_texts()
    plt.setp(ltext[0], fontsize = 10, color = 'k')
    
    plt.show()
    
#    print 'xPosCal: ', xPosCal
#    print 'ind var exvar: ', globExplVar
#    print 'tick Xblock names:', tickPos_XblockNames
#    print; print '*****'; print



#==============================================================================
# Plot global explained variance for each variable in Y 
#==============================================================================

#    globIndVarExplVar = model.Y_cumValExplVar_indVar()
#    
#    # Plot global explained variance
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    
#    # Set axis limits for a pretty plot
#    left = -0.2; right = len(XblockNames) + 0.5; top = 105; bottom = -5
#    
#    # Construct positions for ticks along x-axis.
#    xPosCal = range(len(XblockNames)+1)
#    tickPos_XblockNames = XblockNames[:]
#    tickPos_XblockNames.insert(0, 'none')
#    
##    colourList = ['r', 'k', 'b', 'k', '#FF6600--']
##    lineStyleList = ['-', '--', '.', '-', ':']
#    
#    colourList = ['r', 'r', 'r', 'r' \
#            , 'k', 'k', 'k', 'k' \
#            , 'b', 'b', 'b', 'b'
#            , 'g', 'g', 'g', 'g']
#    lineStyleList = ['-', '--', '-.',  ':' \
#                    ,'-', '--', '-.', ':' \
#                    ,'-', '--', '-.', ':'
#                    ,'-', '--', '-.', ':']
#
#    
#    for varInd in range(np.shape(globIndVarExplVar)[1]):
#        explVarList = list(globIndVarExplVar[:,varInd])
#        ax.plot(xPosCal, explVarList, color=colourList[varInd], \
#                linestyle=lineStyleList[varInd], linewidth=1, \
#                label=YvarNames[varInd])
#
#    # Set lables for x-ticks.
#    ax.set_xticks(xPosCal)    
#    ax.set_xticklabels(tickPos_XblockNames, minor=False, rotation=0, ha='center')
#    
#    ax.set_ylabel('Explained variance [%]')
#     
#    ax.set_xlim(left,right)
#    ax.set_ylim(bottom,top)
#    ax.set_title('Global explained variance Y individual variables')
#    
#    plt.legend(loc='lower right', shadow=True, labelspacing=.1)
#    ltext = plt.gca().get_legend().get_texts()
#    plt.setp(ltext[0], fontsize = 10, color = 'k')
#    
#    plt.show()
#        


#==============================================================================
# Plot projected scores from Y pred after each X block
#==============================================================================

    # First access projected scores of Y pred
    projScoresList = model.projectedYpredScores()
    
    # Reverse list so that scores of Ypred (after last X block) are first
    # in line, etc.
    projScoresList.reverse()
    reversedXblockNames = XblockNames[:]
    reversedXblockNames.reverse()
    
    copyXblockNames = XblockNames[:]
    
    # Start plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Below are three list holding different colours and marker shapes for 
    # up to six X blocks. This list needs to be extended if more X blocks
    # are present in the model.    
    colourList_lines = ['r--', 'g--', 'b--', 'k--', '#FF6600--']
    colourList_dots = ['r', 'g', 'b', 'k', '#FF6600']
    markerStyleList = ['s', 'D', 'p', 'v']
    
    
    # Loop through all score arrays
    for arrInd, arr in enumerate(projScoresList):
        # These are the "real" scores from Ypred after last X block. They are
        # plottet in a scatter plot.         
        if arrInd == 0:
            lab = 'after ' + copyXblockNames[0]
            copyXblockNames.pop(0)
            for name in copyXblockNames:
                lab = lab + ' & ' + name
            
            ax.scatter(arr[arrInd,0], arr[arrInd,1], s=10, c='w', \
                    marker='o', edgecolor='grey', label=lab)
            
            for ind, objName in enumerate(objNames):
                
                ax.scatter(arr[ind,0], arr[ind,1], s=10, c='w', \
                        marker='o', edgecolor='grey')
                ax.text(arr[ind,0], arr[ind,1], objName, fontsize=10)
            
            # Find maximum and minimum scores along PC1 and PC2
            xMax = max(arr[:,0])
            xMin = min(arr[:,0])
            
            yMax = max(arr[:,1])
            yMin = min(arr[:,1])
        
        # For all other projected scores plot them in different colours and 
        # marker shapes and connect them with dashed lines. Lines are starting 
        # from "real" scores and go to next projected Ypred scores (from next)
        # to last X block, and so on until first Ypred.
        else:
            previousScores = projScoresList[arrInd-1]
            actualScores = projScoresList[arrInd]
            
            copyXblockNames = reversedXblockNames[arrInd:]

            revCopyXblockNames = copyXblockNames[:]
            revCopyXblockNames.reverse()
            
            lab = 'after ' + revCopyXblockNames[0]
            revCopyXblockNames.pop(0)
            if revCopyXblockNames == []:
                pass
            else:
                for name in revCopyXblockNames:
                    lab = lab + ' & ' + name
            
            ax.scatter(arr[ind,0], arr[ind,1], s=10, c='w', \
                    marker=markerStyleList[arrInd-1], \
                    edgecolor=colourList_dots[arrInd-1], label=lab)
            
            for ind, objName in enumerate(objNames):
                xcoord = [previousScores[ind][0], actualScores[ind][0]]
                ycoord = [previousScores[ind][1], actualScores[ind][1]]
                 
                ax.scatter(arr[ind,0], arr[ind,1], s=10, c='w', \
                        marker=markerStyleList[arrInd-1], \
                        edgecolor=colourList_dots[arrInd-1])             
                ax.plot(xcoord, ycoord, colourList_lines[arrInd-1])
            
            # Find maximum and minimum scores along PC1 and PC2
            xMax_temp = max(arr[:,0])
            xMin_temp = min(arr[:,0])
            
            yMax_temp = max(arr[:,1])
            yMin_temp = min(arr[:,1])
            
            if xMax_temp > xMax: xMax = xMax_temp
            if xMin_temp < xMin: xMin = xMin_temp
            
            if yMax_temp > yMax: yMax = yMax_temp
            if yMin_temp < yMin: yMin = yMin_temp
    
#    # Find maximum and minimum scores along PC1 and PC2
#    xMax = max(PCPscores[:,0])
#    xMin = min(PCPscores[:,0])
#    
#    yMax = max(PCPscores[:,1])
#    yMin = min(PCPscores[:,1])
    
    
    # Set limits for lines representing the axes.
    # x-axis
    if abs(xMax) >= abs(xMin):
        extraX = xMax * .4
        limX = xMax * .3
    
    elif abs(xMax) < abs(xMin):
        extraX = abs(xMin) * .4
        limX = abs(xMin) * .3
    
    # y-axis
    if abs(yMax) >= abs(yMin):
        extraY = yMax * .4
        limY = yMax * .3
    
    elif abs(yMax) < abs(yMin):
        extraY = abs(yMin) * .4
        limY = abs(yMin) * .3
    
    
    xMaxLine = xMax + extraX
    xMinLine = xMin - extraX
    
    yMaxLine = yMax + extraY
    yMinLine = yMin - extraY
    
    
    ax.plot([0,0], [yMaxLine,yMinLine], color='0.4', linestyle='dashed', \
                    linewidth=1)
    ax.plot([xMinLine,xMaxLine], [0,0], color='0.4', linestyle='dashed', \
                    linewidth=1)
    
    
    # Set limits for plot regions.
    xMaxLim = xMax + limX
    xMinLim = xMin - limX
    
    yMaxLim = yMax + limY
    yMinLim = yMin - limY
    
    ax.set_xlim(xMinLim,xMaxLim)
    ax.set_ylim(yMinLim,yMaxLim)
    
    
    # Plot title, axis names. 
    ax.set_xlabel('PC1 ({0}%)'.format(str(round(PCPexplVar[0],1))))
    ax.set_ylabel('PC2 ({0}%)'.format(str(round(PCPexplVar[1],1))))
    
    ax.set_title('PCP scores plot with projections')
    
    
    plt.legend(loc='lower left', shadow=True, labelspacing=.1)
    ltext = plt.gca().get_legend().get_texts()
    plt.setp(ltext[0], fontsize = 10, color = 'k')    
    
    plt.show()