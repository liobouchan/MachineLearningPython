from __future__ import print_function

import pandas as pd
import matplotlib.pyplot as plt

print("\nPrint pandas version " , pd.__version__)

#Create a Series Object
print("\n Print a Series : \n " , pd.Series(['San Francisco', 'San Jose', 'Sacramento']))

#DataFrame objects can be created by passing a dict mapping string column names to their respective Series.
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
print("\n Print a DataFrame: \n " , pd.DataFrame({ 'City name': city_names, 'Population': population }))

#Load an entire file into a DataFrame
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

#DataFrame.describe to show interesting statistics about a DataFrame
print("\n Print data downloaded CSV : \n" , california_housing_dataframe.describe())

#DataFrame.head, which displays the first few records of a DataFrame:
print("\n Print DataFrame.head : \n" , california_housing_dataframe.head())

#DataFrame.hist lets you quickly study the distribution of values in a column
california_housing_dataframe.hist('housing_median_age')
plt.show()