import KModelSelector
#import tensorflow as tf
from sklearn.datasets import fetch_mldata
import numpy as np


def PrepareData():
    custom_data_home = "Data"
    mnist=fetch_mldata('mnist-original')
    train_x_l = mnist.data
    train_x_l = train_x_l/255
    train_y_l = mnist.target
    n_samples_l = mnist.data.shape[0]
    n_features_l = mnist.data.shape[1]
    n_classes_l = len(np.unique(mnist.target))
    train_y_l = train_y_l.astype(np.int16)
    train_y_l = np.eye(n_classes_l)[train_y_l]
    return train_x_l, train_y_l, n_samples_l, n_features_l, n_classes_l

train_x,train_y, samplesize, features, num_classes = PrepareData()

k = KModelSelector.KModelSelector().RunExperiments( train_x, train_y, features, num_classes, "ImageClassification")

