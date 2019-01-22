import numpy as np
import pandas as pd
import pickle
import urllib
import tensorflow as tf
import tensorflow_hub as hub

from sklearn.preprocessing import MultiLabelBinarizer

data = pd.read_csv('movies_metadata.csv') #print("\n Data : \n" , data)
descriptions = data['overview'] #print("\n Descriptions : \n" , descriptions)
genres = data['genres'] #print("\n Genres : \n" , genres)

#top_genres = ['Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime', 'Documentary', 'Adventure', 'Science Fiction']

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

encoder = MultiLabelBinarizer()
encoder.fit_transform(train_genres)
train_encoded = encoder.transform(train_genres)
print("\n encoder.transform(train_genres) : \n" , train_encoded)
test_encoded = encoder.transform(test_genres)
print("\n encoder.transform(test_genres) : \n" , test_encoded)
num_classes = len(encoder.classes_)
print("\n num_classes : \n" , num_classes)
print("\n encoder.classes_ : \n" , encoder.classes_)

description_embeddings = hub.text_embedding_column("descriptions", module_spec="https://tfhub.dev/google/universal-sentence-encoder/2", trainable=False)
print("\n description_embeddings : \n" , description_embeddings)

multi_label_head = tf.contrib.estimator.multi_label_head(
    num_classes,
    loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE
)

features = {
  "descriptions": np.array(train_descriptions).astype(np.str)
}
print("\n features : \n" , features)
labels = np.array(train_encoded).astype(np.int32)
print("\n labels : \n" , labels)
train_input_fn = tf.estimator.inputs.numpy_input_fn(features, labels, shuffle=True, batch_size=32, num_epochs=25)
print("\n train_input_fn : \n" , train_input_fn)
estimator = tf.contrib.estimator.DNNEstimator(
    head=multi_label_head,
    hidden_units=[64,10],
    feature_columns=[description_embeddings])
print("\n estimator : \n" , estimator)
estimator.train(input_fn=train_input_fn,     hooks=None,
    steps=None,
    max_steps=None,
    saving_listeners=None)

# Define our eval input_fn and run eval
eval_input_fn = tf.estimator.inputs.numpy_input_fn({"descriptions": np.array(test_descriptions).astype(np.str)}, test_encoded.astype(np.int32), shuffle=False)
estimator.evaluate(input_fn=eval_input_fn)

# Test our model on some raw description data
raw_test = [
    "An examination of our dietary choices and the food we put in our bodies. Based on Jonathan Safran Foer's memoir.", # Documentary
    "After escaping an attack by what he claims was a 70-foot shark, Jonas Taylor must confront his fears to save those trapped in a sunken submersible.", # Action, Adventure
    "A teenager tries to survive the last week of her disastrous eighth-grade year before leaving to start high school.", # Comedy
]

# Generate predictions
predict_input_fn = tf.estimator.inputs.numpy_input_fn({"descriptions": np.array(raw_test).astype(np.str)}, shuffle=False)
print("\n predict_input_fn: \n" ,predict_input_fn)
results = estimator.predict(predict_input_fn)

# Display predictions
for movie_genres in results:
  print("\n Results : " , results)
  print("\n movie_genres : ", movie_genres)
  top_2 = movie_genres['probabilities'].argsort()[-2:][::-1]
  print("Top 2 : " , top_2)
  for genre in top_2:
    print("genre : ", genre)
    text_genre = encoder.classes_[genre]
    print("text_genre : ", text_genre)
    print(text_genre + ': ' + str(round(movie_genres['probabilities'][genre] * 100, 2)) + '%')
  print('')