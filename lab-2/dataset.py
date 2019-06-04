import matplotlib.pyplot as plt
from sklearn import datasets

# using syntax like that because of https://github.com/PyCQA/pylint/issues/2053
iris = datasets.load_iris()

print(dir(iris))

print(iris["feature_names"])

print(iris["target"])

print(iris["target_names"])

iris_data = iris["data"]

x_axis = iris_data[:, 0]  # Sepal Length
y_axis = iris_data[:, 1]  # Sepal Width

# Construction
plt.xlabel(iris["feature_names"][0])
plt.ylabel(iris["feature_names"][1])
plt.scatter(x_axis, y_axis, c=iris["target"])
plt.show()
