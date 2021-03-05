# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 21:51:01 2019

@author: olive
"""



# =============================================================================
# Import modules
# =============================================================================
import pandas as pd
import sopls
import time
import numpy as np


# =============================================================================
# Load data, process data and prepare objects
# =============================================================================

# Load data into pandas dataframes
A_df = pd.read_excel("A.xlsx", index_col=0)
B_df = pd.read_excel("B.xlsx", index_col=0)
C_df = pd.read_excel("C.xlsx", index_col=0)
D_df = pd.read_excel("D.xlsx", index_col=0)


# Check for missing values
A_missing = A_df.isnull().values.any()
B_missing = B_df.isnull().values.any() 
C_missing = C_df.isnull().values.any()
D_missing = D_df.isnull().values.any()


# Count missing values for individual variables
A_missingCount_indVar = list(A_df.isnull().sum())
B_missingCount_indVar = list(B_df.isnull().sum())
C_missingCount_indVar = list(C_df.isnull().sum())
D_missingCount_indVar = list(D_df.isnull().sum())


# Count missing values in total for each block
A_missingCount_all = A_df.isnull().sum().sum()
B_missingCount_all = B_df.isnull().sum().sum()
C_missingCount_all = C_df.isnull().sum().sum()
D_missingCount_all = D_df.isnull().sum().sum()


# Find rows and columns that contain missing data
A_null = A_df[A_df.isnull().any(axis=1)]
B_null = B_df[B_df.isnull().any(axis=1)]
C_null = C_df[C_df.isnull().any(axis=1)]
D_null = D_df[D_df.isnull().any(axis=1)]
null_df_list = [A_null, B_null, C_null, D_null]


# Find rows across all blocks that contain missing data
null_indices_list = []
for null_df in null_df_list:
    null_indices_list.extend(list(null_df.index))
null_rowID_list = list(set(null_indices_list))


# Drop those rows that are missing across all blocks
A_noMissing = A_df.drop(null_rowID_list)
B_noMissing = B_df.drop(null_rowID_list)
C_noMissing = C_df.drop(null_rowID_list)
D_noMissing = D_df.drop(null_rowID_list)


# Get numpy arrays for computation
A = A_noMissing.values
B = B_noMissing.values
C = C_noMissing.values
D = D_noMissing.values


# Define name of blocks
A_name = 'A'
B_name = 'B'
C_name = 'C'
D_name = 'D'


# Extract variable names of each block
A_colNames = list(A_df.columns)
B_colNames = list(B_df.columns)
C_colNames = list(C_df.columns)
D_colNames = list(D_df.columns)


## =============================================================================
## Model CV: D <-- A + B + C
## =============================================================================
#
## Check how long it takes to run the whole script. This captures the starting
## time.
#tStart = time.time()
#
#
## Define in which order data arrays are fit in model and whether to use scaling
#case = 3
#
## CASE 1: D <-- A
#if case == 1:
#    
#    modCV = sopls.SOPLSCV(D, [A], Xcomps=[7], \
#            Xstand=[False], Ystand=False)
#
#
## CASE 2: D <-- A(fixed_comp) + B
#if case == 2:
#    
#    modCV = sopls.SOPLSCV(D, [A, B], Xcomps=[1, 7], \
#            Xstand=[False, False], Ystand=False)
#
#
## CASE 3: D <-- A(fixed_comp) + B(fixed_comp) + C
#if case == 3:
#    
#    modCV = sopls.SOPLSCV(D, [A, B, C], Xcomps=[1, 1, 5], \
#            Xstand=[False, False, False], Ystand=False)
#
#
#
## Access results from SO-PLS
#results = modCV.results()
#settings = modCV.modelSettings()
#rmsecv = modCV.RMSECV()
#
## Compute how much time was used to run the code
#tEnd = time.time()
#
#print
#print('TOTAL TIME: ', tEnd - tStart, 'seconds')
#print
#
#del tStart, tEnd
#
## Now plot the results in Maage plot
#sopls.plotRMSEP(rmsecv)


# =============================================================================
# Model: D <-- A + B + C
# =============================================================================

# Compute final model with selected components
mod = sopls.SOPLS(D, [A, B, C], Xcomps=[1, 1, 2], \
            Xstand=[False, False, False], Ystand=False)


# For running 'plotSOPLS' a few parameters need to be predefined
YvarNames = list(D_colNames)
XblocksVarNames = [A_colNames, B_colNames, C_colNames]
objNames = list(A_noMissing.index)
XblockNames = [A_name, B_name, C_name]


# Plot model results
sopls.plotSOPLS(mod, objNames, YvarNames, XblocksVarNames, XblockNames)


## Since the model has only one component it is not possible to plot the 
## PCP plot. Hence, results are printed out as below.
#PCP_scores = mod.PCPscores()
#
YcumCal = mod.Y_cumCalExplVar()
YcumVal = mod.Y_cumValExplVar()
print()
print('Cum cal expl var:', np.round(YcumCal, 2))
print('Cum val expl var:', np.round(YcumVal, 2))


YcumCal_ind = mod.Y_cumCalExplVar_indVar()
YcumVal_ind = mod.Y_cumValExplVar_indVar()
print('\nCum cal expl var ind:\n', np.round(YcumCal_ind, 1))
print('\nCum cal expl var ind:\n', np.round(YcumVal_ind, 1))





