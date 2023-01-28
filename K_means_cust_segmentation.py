import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("C:/Users/Mahdieh/Desktop/Machine Learning/clustring/Cust_Segmentation.csv")
df=df.drop('Address',axis=1) #axis=1 means columns, not rows ** for removing rows, we can write axis=0
# print(df.head(4))
# print(df.describe())
x=df.values[:,1:] # all rows with columns from 1 to the end
#print(df.dtypes)
x=np.nan_to_num(x)
clus_dataset=StandardScaler().fit_transform(x)
clust_Num=3
k_means=KMeans(init="k-means++",n_clusters=clust_Num, n_init=12)
k_means.fit(x)
print(k_means.labels_)
df["Clust Num"]=k_means.labels_
print(df.head(5))
print(df.groupby("Clust Num").mean())
area = np.pi * ( x[:, 1])**2 # education as area of each dot (education is the radius) 
plt.scatter(x[:, 0], x[:, 3], s=area, c=k_means.labels_.astype(np.float), alpha=0.5) #age vs. income *  for color the 3 labels are changed to color
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()
###### 3d analysis

from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
#plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')
ax.scatter(x[:, 1], x[:, 0], x[:, 3], c=k_means.labels_.astype(np.float))
plt.show()