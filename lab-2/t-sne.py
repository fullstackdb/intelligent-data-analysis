# read more https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding
from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

iris_df = datasets.load_iris()

# model and speed
model = TSNE(learning_rate=100)

# teach model
transformed = model.fit_transform(iris_df['data'])

# representing result in two-dimensional coordinates
x_axis = transformed[:, 0]
y_axis = transformed[:, 1]

plt.scatter(x_axis, y_axis, c=iris_df['target'])
plt.show()
