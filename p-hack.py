import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the dataset
csv_file = "./datasets/student_performance/body_dimensions_predictions.csv"
data = pd.read_csv(csv_file)

target_y = 'wgt'
protected_attribute = 'age'
prediction = 'y_hat'

# Extract relevant columns
x = data[target_y].values  # x-axis (wgt)
y = data[protected_attribute].values  # y-axis (age)
z = data[prediction].values  # z-axis (y_hat)

# Fit a plane using linear regression
X = np.vstack([x, y]).T
reg = LinearRegression()
reg.fit(X, z)
coef = reg.coef_
intercept = reg.intercept_

# Define vectors for reparameterization
a = 1  # Choose a nonzero x-component
c = coef[0] * a  # z-component for v1
v1 = np.array([a, 0, c])  # [x, y, z] for v1
v1 = v1 / np.linalg.norm(v1)  # Normalize v1

p = 1  # Choose a nonzero x-component
q = 1  # Choose a nonzero y-component
r = coef[0] * p + coef[1] * (-q)  # z-component for v2
v2 = np.array([p, -q, r])  # [x, y, z] for v2
v2 = v2 / np.linalg.norm(v2)  # Normalize v2

# Function to solve for a and b
def solve_ab(x, y):
    # Compute the z value from the fitted plane equation
    z_plane = coef[0] * x + coef[1] * y + intercept
    
    # Target vector in the plane
    target = np.array([x, y, z_plane])
    
    # Solve the linear system: target = a*v1 + b*v2 + intercept
    # Subtract the intercept term
    target -= np.array([0, 0, intercept])
    
    # Form the matrix with v1 and v2 as columns
    V = np.vstack([v1, v2]).T
    
    # Solve for [a, b] using least squares
    ab = np.linalg.lstsq(V, target, rcond=None)[0]
    return ab

# Iterate over all observations and assign to chunks
chunk_size = int(len(data)/(len(data) / 100))
data['a'], data['b'] = zip(*[solve_ab(row[target_y], row[protected_attribute]) for _, row in data.iterrows()])
data['chunk'] = pd.cut(data['a'], bins=np.arange(data['a'].min(), data['a'].max() + chunk_size, chunk_size), labels=False)

# Save chunks to separate dataframes
chunked_data = {chunk: group for chunk, group in data.groupby('chunk')}

# Output results
for chunk, df in chunked_data.items():
    print(f"Chunk {chunk}:\n", df.head(), "\n")
    # Optionally save to CSV
    df.to_csv(f'./datasets/student_performance/chunk_{chunk}.csv', index=False)