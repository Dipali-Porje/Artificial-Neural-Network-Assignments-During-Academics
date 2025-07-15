import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import Perceptron 
# Generate random data 
np.random.seed(0)  #generate same random number every time 
X = np.random.randn(100, 2)  #100 samples with 2 features
#sums two features of each sample
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1) #assign label 1 if sum greater than 0, otherwise -1
# Create perceptron object 
clf = Perceptron(random_state=0) 
# Fit perceptron to the data 
clf.fit(X, y) 
# Plot decision regions 
xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1 
ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1 
xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.1), np.arange(ymin, ymax, 0.1)) 
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  #predict label for each point 
Z = Z.reshape(xx.shape)  #reshapes it to match shape of mesh grid for plotting
plt.contourf(xx, yy, Z, alpha=0.4)   #creates filled contour plot
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8) 
plt.xlim(xmin, xmax) 
plt.ylim(ymin, ymax) 
plt.xlabel('Feature 1') 
plt.ylabel('Feature 2') 
plt.show()