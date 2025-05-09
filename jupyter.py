# Task 1: Load and Explore the Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the dataset with error handling
try:
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['species'] = iris.target
    species_map = dict(zip(range(3), iris.target_names))
    data['species'] = data['species'].map(species_map)
    print("Dataset loaded successfully.\n")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Display the first few rows
print("First 5 rows of the dataset:")
print(data.head())

# Data structure
print("\nDataset Info:")
print(data.info())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Clean the dataset (not necessary for Iris, but included as example)
data = data.dropna()

# Task 2: Basic Data Analysis

# Descriptive statistics
print("\nDescriptive Statistics:")
print(data.describe())

# Group by species and compute mean
print("\nMean values by species:")
print(data.groupby('species').mean())

# Observations
print("\nObservations:")
print("1. Setosa has significantly shorter petal lengths than the other species.")
print("2. Versicolor and Virginica have similar sepal lengths but different petal widths.")

# Task 3: Data Visualization

# Set seaborn style
sns.set(style="whitegrid")

# 1. Line chart (example: petal length trend by index for each species)
plt.figure(figsize=(10,6))
for species in data['species'].unique():
    subset = data[data['species'] == species]
    plt.plot(subset.index, subset['petal length (cm)'], label=species)
plt.title('Petal Length Over Samples')
plt.xlabel('Sample Index')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.show()

# 2. Bar chart: average petal length per species
plt.figure(figsize=(8,6))
avg_petal = data.groupby('species')['petal length (cm)'].mean()
avg_petal.plot(kind='bar', color='skyblue')
plt.title('Average Petal Length per Species')
plt.ylabel('Petal Length (cm)')
plt.xlabel('Species')
plt.show()

# 3. Histogram: distribution of sepal length
plt.figure(figsize=(8,6))
plt.hist(data['sepal length (cm)'], bins=15, color='orange', edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter plot: sepal length vs petal length
plt.figure(figsize=(8,6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=data, palette='Set1')
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.show()
