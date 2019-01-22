import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

#Load all files from a directory in a DataFrame
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []

    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory , file_path) , "r") as f:
            data["sentece"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt" , file_path).group(1))
    return pd.DataFrame.from_dict(data)

#Merge positive and negative examples , add a polarity column and shuffle
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory , "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df , neg_df]).sample(frac = 1).reset_index(drop= True)

