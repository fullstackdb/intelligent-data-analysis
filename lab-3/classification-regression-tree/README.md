# CART

This is an implementation of a Classification and Regression Tree (CART) algorithm in Python.

Decision trees are pretty cool, in that they are fairly simple and are a powerful prediction method. This immplementaion will calculate and evaluate potential split points in a training dataset, arrange the splits into a decision tree structure, and then apply the classification and regression tree algorithm to a realworld problem using the Banknote dataset (determining authenticity based on measures taken from photographs).

The Banknote dataset has five columns, from first to last, the varibles are:
1. Variance of the Wavelet Transformed image (continuous)
2. Skewness of the Wavelet Transformed image (continuous)
3. Kurtosis (sharpness of the peak and frequency distribution curve) of the Wavelet Transformed image (continuous)
4. Entropy of the image (continuous)
5. Class (integer)
