import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Load the dataset
csv_file = "./datasets/smartphone_metrics/chunk_0.0.csv"
data = pd.read_csv(csv_file)

target_y = 'storage'
protected_attribute = 'price_usd'
prediction = 'y_hat'

# Extract relevant columns
x = data[target_y].values.reshape(-1, 1)  # x-axis (wgt)
y = data[protected_attribute].values.reshape(-1, 1)  # y-axis (age)
z = data[prediction].values.reshape(-1, 1)  # z-axis (y_hat)

# Combine x, y into a single matrix for fitting
X = np.hstack([x, y])

# Fit a plane using linear regression
reg = LinearRegression()
reg.fit(X, z)
z_pred = reg.predict(X)

# Plane coefficients
coef = reg.coef_[0]
intercept = reg.intercept_[0]

# Define the fitted plane equation
def fitted_plane(x, y):
    return coef[0] * x + coef[1] * y + intercept

# Visualize the original dataset and the fitted plane
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the data points
ax.scatter(x, y, z, color='blue', label='Data Points')

# Create a meshgrid for the plane
x_range = np.linspace(min(x)[0], max(x)[0], 20)
y_range = np.linspace(min(y)[0], max(y)[0], 20)
x_mesh, y_mesh = np.meshgrid(x_range, y_range)
z_mesh = fitted_plane(x_mesh, y_mesh)

# Plot the fitted plane
ax.plot_surface(x_mesh, y_mesh, z_mesh, alpha=0.5, color='orange', label='Fitted Plane')

### Reparameterization step

# Define vector v1: Orthogonal to y-axis and lies in the plane
a = 1  # Choose a nonzero x-component
c = coef[0] * a  # z-component to ensure v1 lies in the plane
v1 = np.array([a, 0, c])  # [x, y, z] for v1
v1 = v1 / np.linalg.norm(v1)  # Normalize v1

# Define vector v2: Establishes negative correlation between x and y and lies in the plane
p = 1  # Choose a nonzero x-component
q = 1  # Choose a nonzero y-component (negative correlation implies x and y have opposite signs)
r = coef[0] * p + coef[1] * (-q)  # z-component to ensure v2 lies in the plane
v2 = np.array([p, -q, r])  # [x, y, z] for v2
v2 = v2 / np.linalg.norm(v2)  # Normalize v2

# Normalize vectors
v1 = v1 / np.linalg.norm(v1)
v2 = v2 / np.linalg.norm(v2)

# Reparameterize the plane
def reparameterized_plane(a, b):
    return a * v1[0] + b * v2[0], a * v1[1] + b * v2[1], a * v1[2] + b * v2[2] + intercept

# Create meshgrid for reparameterized plane
a_range = np.linspace(-300, 300, 20)
b_range = np.linspace(-300, 300, 20)
a_mesh, b_mesh = np.meshgrid(a_range, b_range)
x_reparam, y_reparam, z_reparam = reparameterized_plane(a_mesh, b_mesh)

# Plot the reparameterized plane
ax.plot_surface(x_reparam, y_reparam, z_reparam, alpha=0.5, color='green', label='Reparameterized Plane')

# Labels and legend
ax.set_xlabel(f'{target_y} (x-axis)')
ax.set_ylabel(f'{protected_attribute} (y-axis)')
ax.set_zlabel('y_hat (z-axis)')
plt.legend()
plt.show()