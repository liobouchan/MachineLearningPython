import numpy as np
import pandas as pd
import pickle
import urllib
import tensorflow as tf
import tensorflow_hub as hub

from sklearn.preprocessing import MultiLabelBinarizer

data = pd.read_csv('movies_metadata.csv')
print("\n Data : \n" , data)

descriptions = data['overview']
print("\n Descriptions : \n" , descriptions)

genres = data['genres']
print("\n Genres : \n" , genres)

top_genres = ['Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime', 'Documentary', 'Adventure', 'Science Fiction']

urllib.request.urlretrieve('https://storage.googleapis.com/bq-imports/descriptions.p', 'descriptions.p')
urllib.request.urlretrieve('https://storage.googleapis.com/bq-imports/genres.p', 'genres.p')

descriptions = pickle.load(open('descriptions.p', 'rb'))
genres = pickle.load(open('genres.p', 'rb'))

train_size = int(len(descriptions) * .8)
print("\n Train Size :" , train_size)

train_descriptions = descriptions[:train_size].astype('str')
print("\n train_descriptions : \n" , train_descriptions)
train_genres = genres[:train_size]
print("\n train_genres : \n" , train_genres)


test_descriptions = descriptions[train_size:].astype('str')
print("\n test_descriptions : \n" , test_descriptions)
test_genres = genres[train_size:]
print("\n test_genres : \n" , test_genres)

description_embeddings = hub.text_embedding_column(
  "movie_descriptions",
  module_spec="https://tfhub.dev/google/universal-sentence-encoder/2"
)

encoder = MultiLabelBinarizer()
encoder.fit_transform(train_genres)
train_encoded = encoder.transform(train_genres)
print("\n encoder.transform(train_genres) : \n" , train_encoded)
test_encoded = encoder.transform(test_genres)
print("\n encoder.transform(test_genres) : \n" , test_encoded)
num_classes = len(encoder.classes_)
print("\n num_classes : \n" , num_classes)
print("\n encoder.classes_ : \n" , encoder.classes_)