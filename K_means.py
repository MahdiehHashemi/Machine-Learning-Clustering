import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib.colors as cplt
np.random.seed(0)
x,y=make_blobs(n_samples=5000,centers=[[4,4],[-2,-1],[2,-3],[1,1]],cluster_std=0.9)
print(x)
print(y)
plt.scatter(x[:,0],x[:,1],marker=".")
plt.show()
k_means=KMeans(n_clusters=4,init="k-means++",n_init=12)
k_means.fit(x)
print(k_means.cluster_centers_)
print(k_means.labels_)
a=k_means.labels_
for i in range (4):
    for j in range (len(x)):
        if k_means.labels_[j]==i:
            d=np.sqrt((x[j,0]-k_means.cluster_centers_[i][0])**2+(x[j,1]-k_means.cluster_centers_[i][1])**2)
    print("error of cluster number "+str(i)+ "is " +str(d))
    
#########################
# fig = plt.figure(figsize=(6, 4))
# colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means.labels_))))
# ax = fig.add_subplot(1, 1, 1)
# for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):
#     my_members = (k_means.labels_ == k)
#     cluster_center = k_means.cluster_centers_[k]
#     ax.plot(x[my_members, 0], x[my_members, 1], 'w', markerfacecolor=col, marker='.')
#     ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
# ax.set_title('KMeans')
# ax.set_xticks(())
# ax.set_yticks(())
# plt.show()
########################
c=cplt.ListedColormap(["red","green","blue","yellow"])
plt.scatter(x[:,0],x[:,1],marker=".",c=k_means.labels_,cmap=c)
plt.show()
