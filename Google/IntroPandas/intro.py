from __future__ import print_function

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

#access DataFrame data using familiar Python dict/list operations
cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
print("\n" , type(cities['City name']))
print(cities['City name'])

#Print the 1 position
print("\n" ,type(cities['City name'][1]))
print(cities['City name'][1])

#Print from 0 to 2
print("\n" ,type(cities[0:2]))
print(cities[0:2])

#pandas Series can be used as arguments to most NumPy functions:
print("\n np.log(population) ", np.log(population))

# Like the Python map function, Series.apply accepts as an argument a lambda function, which is applied to each value.
print("\n population.apply(lambda val: val > 1000000) ", population.apply(lambda val: val > 1000000))

#Modifying DataFrames is also straightforward.
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
print("\n Cities Modified" , cities)

#Modify the cities table by adding a new boolean column that is True if and only if both of the following are True:
#The city is named after a saint.
#The city has an area greater than 50 square miles.
cities['Is wide and has saint name'] = (cities['Area square miles'] > 50) & cities['City name'].apply(lambda name: name.startswith('San'))
print(cities)

#Both Series and DataFrame objects also define an index property that assigns an identifier value to each Series item or DataFrame row.
print("\n city_names.index" , city_names.index)
print("\n cities.index" , cities.index)

#Call DataFrame.reindex to manually reorder the rows.
print("\n cities.reindex([2, 0, 1])" , cities.reindex([2, 0, 1]))

#Reindexing is a great way to shuffle (randomize) a DataFrame. In the example below, we take the index, which is array-like, and pass it to NumPy's random.permutation function, which shuffles its values in place. Calling reindex with this shuffled array causes the DataFrame rows to be shuffled in the same way.
print("\n 1 reindex" , cities.reindex(np.random.permutation(cities.index)))
print("\n 2 reindex" , cities.reindex(np.random.permutation(cities.index)))
print("\n 3 reindex" , cities.reindex(np.random.permutation(cities.index)))