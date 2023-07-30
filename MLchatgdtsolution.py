from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
import numpy as np

# Generate a toy dataset
X, y = make_classification(n_samples=1000, n_features=2)

# Define a classifier
clf = BaggingClassifier(n_estimators=1, max_samples=1.0, max_features=1.0)

# Generate 5 bootstrap samples
bootstrap_samples = [np.random.choice(X.shape[0], X.shape[0], replace=True) for _ in range(5)]

# Plot each bootstrap sample
for i, idx in enumerate(bootstrap_samples):
    # Train the classifier on the bootstrap sample
    clf.fit(X[idx], y[idx])
    
    # Generate a grid of points in the feature space
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
                         np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
    
    # Predict the class probabilities for each point in the grid
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary for the classifier
    plt.contour(xx, yy, Z, colors='k', levels=[0.5])
    
# Plot the original dataset
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title('Decision boundaries for 5 bootstrap samples')
plt.show()
