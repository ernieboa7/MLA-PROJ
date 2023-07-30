import numpy as np

# Create a two-dimensional dataset with 100 data points
np.random.seed(42)
data = np.random.randn(100, 2)

# Set the number of samples you want to generate
n_samples = 5

# Generate the random bootstrap samples
samples = []
for i in range(n_samples):
    bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
    samples.append(bootstrap_sample)



'''import matplotlib.pyplot as plt

# Plot the first bootstrap sample
plt.scatter(samples[0][:,0], samples[0][:,1])
plt.show()'''
