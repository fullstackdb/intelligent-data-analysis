# learn more https://en.wikipedia.org/wiki/DBSCAN

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

iris = load_iris()

dbscan = DBSCAN()

# teaching model
dbscan.fit(iris['data'])

# reducing the dimension by using the main components method
pca = PCA(n_components=2).fit(iris['data'])
pca_2d = pca.transform(iris['data'])

# building according to the three classes
for i in range(0, pca_2d.shape[0]):
    if dbscan.labels_[i] == 0:
        c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
    elif dbscan.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
    elif dbscan.labels_[i] == -1:
        c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')

plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2', 'Noise'])
plt.title('DBSCAN found 2 clusters & noise')
plt.show()
