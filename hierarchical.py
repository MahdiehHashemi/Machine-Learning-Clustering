import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
x1,y1=make_blobs(n_samples=50, centers=[[4,4],[-2,-1],[1,1],[10,4]],cluster_std=0.9)
#print(y1)
plt.scatter(x1[:,0], x1[:,1], marker='.')
plt.show()
agglom=AgglomerativeClustering(n_clusters=4, linkage='average')
agglom.fit(x1,y1)
#xmin,xmax=np.min(x1,axis=0),np.max(x1,axis=0)# finding min between rows (not columns): axis=0 is for it
#print(xmin,xmax) # it would be : [-3.5058992 -2.7591039] [10.94481388  5.80216054]
#X1=(x1-xmin)/(xmax-xmin) #all numbers between 0 and 1
co=["Red","Green","Blue","Yellow"]
print(agglom.labels_)
for i in range (x1.shape[0]): # X, y = make_blobs(n_samples=10, centers=3, n_features=2,random_state=0): print(X.shape):(10, 2)
    plt.text(x1[i,0],x1[i,1],str(y1[i]))
    plt.scatter(x1[i,0], x1[i,1], marker='.',c=co[agglom.labels_[i]])
plt.xticks([])
plt.yticks([])
plt.show()
#####plotting dendrogram
dis_mat=distance_matrix(x1,x1)
dis_mat=np.array(dis_mat)
# print(np.amax(dis_mat))
# print(np.amin(dis_mat))
z=hierarchy.linkage(dis_mat,'complete')
d=hierarchy.dendrogram(z)
plt.show()
########The y-axis is a measure of closeness of either individual data points or clusters.