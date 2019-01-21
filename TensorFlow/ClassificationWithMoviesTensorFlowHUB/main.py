import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub

from sklearn.preprocessing import MultiLabelBinarizer

data = pd.read_csv('movies_metadata.csv')

descriptions = data['overview']
genres = data['genres']