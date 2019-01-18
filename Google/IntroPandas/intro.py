from __future__ import print_function

import pandas as pd

print("\nPrint pandas version " , pd.__version__)

#Create a Series Object
print("\n Print a Series : \n " , pd.Series(['San Francisco', 'San Jose', 'Sacramento']))

#DataFrame objects can be created by passing a dict mapping string column names to their respective Series.
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
print("\n Print a DataFrame: \n " , pd.DataFrame({ 'City name': city_names, 'Population': population }))