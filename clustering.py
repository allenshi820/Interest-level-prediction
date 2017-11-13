import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


df = pd.read_pickle('Two_sigma')
df = df.reset_index()
#train
df1 = df[0:30000]
#predict
df2 = df[30000::]

k = AgglomerativeClustering(n_clusters=5)
dis = k.fit(np.array(df1[['latitude', 'longitude']]))
df1.loc[::, 'class']=k.labels_


qda = QuadraticDiscriminantAnalysis()

qda.fit(np.array(df1[['latitude', 'longitude']]), k.labels_,  store_covariances=None)
df2.loc[::, 'class']=qda.predict(np.array(df2[['latitude', 'longitude']]))

frames = [df1, df2]
result = pd.concat(frames)

result.to_pickle('New')
