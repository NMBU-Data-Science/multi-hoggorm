import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from resources import SOPLS

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
print(mlf2.predict(X))