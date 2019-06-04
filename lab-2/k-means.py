# read more https://en.wikipedia.org/wiki/K-means_clustering

from sklearn import datasets
from sklearn.cluster import KMeans

iris_df = datasets.load_iris()

# model defining
model = KMeans(n_clusters=3)

# modeling
model.fit(iris_df['data'])

# single example prediction
predicted_label = model.predict([[7.2, 3.5, 0.8, 1.6]])

# all datas prediction
all_predictions = model.predict(iris_df['data'])

print(predicted_label)
print(all_predictions)
