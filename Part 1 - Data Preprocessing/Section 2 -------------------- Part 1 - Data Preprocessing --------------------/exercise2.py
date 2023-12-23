# Importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Load the dataset
dataset = pd.read_csv('pima-indians-diabetes.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

# Identify missing data (assumes that missing data is represented as NaN)
# Print the number of missing entries in each column
print(dataset.isnull().sum())
# Configure an instance of the SimpleImputer class
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer on the DataFrame
imputer.fit(X)

# Apply the transform to the DataFrame
X=imputer.transform(X)
#Print your updated matrix of features
print(X)