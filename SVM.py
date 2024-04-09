# Importing the numpy library and aliasing it as np. Numpy is used for numerical operations.
import numpy as np

# Importing the pyplot module from matplotlib library and aliasing it as plt. Matplotlib.pyplot is used for creating static, animated, and interactive visualizations in Python.
import matplotlib.pyplot as plt

# Importing the pandas library and aliasing it as pd. Pandas is used for data manipulation and analysis.
import pandas as pd

# Defining the path of the dataset file.
path="/content/drive/MyDrive/Dataset/Social_Network_Ads.csv"

# Reading the CSV file located at the defined path using pandas and storing the data in a DataFrame.
df = pd.read_csv(path)

# Displaying the first 5 rows of the DataFrame.
df.head()


# Displaying the shape of the DataFrame. This returns a tuple representing the dimensionality of the DataFrame (rows, columns).
df.shape



# Selecting the 2nd and 3rd columns from the DataFrame df and assigning it to x. iloc is used for integer-location based indexing.
# Independent variable
x = df.iloc[:, [2,3]]

# Dependet variable
# Selecting the 4th column from the DataFrame df and assigning it to y.
y = df.iloc[:,4]


# Display the first 5 rows of the dataframe 'y'
y.head()



# Import the 'train_test_split' function from the 'sklearn.model_selection' module
from sklearn.model_selection import train_test_split

# Split the 'x' and 'y' dataframes into training and testing sets
# The test set size is 25% of the entire dataset
# The 'random_state' parameter ensures reproducibility of the split
x_Train, x_Test, y_Train, y_Test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# Print the shape of the training data 'x_Train'
print("Training data : ", x_Train.shape)


# Print the shape of the testing data 'y_Test'
print("Testing data : ", y_Test.shape)




# Import the 'StandardScaler' class from the 'sklearn.preprocessing' module
from sklearn.preprocessing import StandardScaler

# Create an instance of the 'StandardScaler' class
sc_x = StandardScaler()

# Fit the scaler to the training data 'x_Train' and transform it
x_Train = sc_x.fit_transform(x_Train)

# Use the same scaler to transform the testing data 'x_Test'
x_Test = sc_x.transform(x_Test)





