import tensorflow as tf
from tensorflow import keras

import numpy as np

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(tf.__version__)
print()

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])
print("Lenght of [0] and [1] ", len(train_data[0]), ",", len(train_data[1]))

print()

print("Testing entries: {}, labels: {}".format(len(test_data), len(test_labels)))
print(test_data[0])
print("Lenght of [0] and [1] ", len(test_data[0]), ",", len(test_data[1]))

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

print()
print("Train Data [0] " , decode_review(train_data[0]))
print()
print("Test Data [0] " , decode_review(test_data[0]))

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

print()
print("Standarizate lengths of train_data[0] and [1]", len(train_data[0]), ",", len(train_data[1]))
print()
print("Standarizate lengths of test_data[0] and [1]", len(test_data[0]), ",", len(test_data[1]))