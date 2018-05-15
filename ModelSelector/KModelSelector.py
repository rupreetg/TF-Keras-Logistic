import os
import sys
#sys.path.append('D:\\ML-Lab\\TF-Keras-Logistic\\CoreNetwork')
libpath="D:\\ML-Lab\\TF-Keras-Logistic\\CoreNetwork"
if (libpath not in sys.path):
    sys.path.append("D:\\ML-Lab\\TF-Keras-Logistic\\CoreNetwork")

import CoreNetworkModel as cn
import ModelParameters as mp
import optimizers  as asop

import tensorflow as tf
from sklearn.datasets import fetch_mldata
import numpy as np

import json
from pprint import pprint

class KModelSelector(object):

    def __init__(self):
        self.configs = self._getConfig()
        self._learningRate = 0.01
        self._optimizers = self._getOptimizers()
        self._epocs = 5
        self._miniBatch = 200

    def _getConfig(self):
        with open('D:\\ML-Lab\\TF-Keras-Logistic\\ModelSelector\\ModelTagMapping.json') as f:
            return json.load(f)

    def _getOptimizers(self):
        opts = [
            tf.train.GradientDescentOptimizer(learning_rate=self._learningRate),
            asop.ASGradientDescentOptimizer(base_learning_rate=self._learningRate),
            tf.train.RMSPropOptimizer(learning_rate=self._learningRate),
            asop.ASRMSPropOptimizer(base_learning_rate=self._learningRate),
            tf.train.AdamOptimizer(learning_rate=self._learningRate),
            tf.train.MomentumOptimizer(learning_rate=self._learningRate, momentum=.9),
            tf.train.MomentumOptimizer(learning_rate=self._learningRate, momentum=.9, use_nesterov=True),
            tf.train.AdagradOptimizer(learning_rate=self._learningRate)
            ]
        opt_names = ['SGD', 'SGD+AS', 'RMSProp', 'RMSProp+AS', 'ADAM', 'SGD+M', 'SGD+NM', 'Adagrad']
        return zip(opts, opt_names)


    def GetExperiments(self, Tag):
        tagconfigs = self.configs
        tagFunctions = tagconfigs[Tag]
        return tagFunctions

    def RunExperiments(self, X, Y, features, output_classes, Tag):
        tagFunctions = self.GetExperiments(Tag)
        for t in tagFunctions:
            getattr(KModelSelector, t)(self, X, Y, features, output_classes)

    def _train(self, X, Y, model_p):
        losses=[] 
        for optimizer, optmizerName in self._getOptimizers():
            init1 = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init1)
                network = cn.CoreNetwork( model_p)
                loss = network.train(sess, X , Y, self._epocs, optimizer, self._miniBatch, optmizerName)
                losses.append(loss)
            tf.reset_default_graph()


    def BuildLogisticRegressionModel(self, X, Y, features, output_classes):
        print ("Returning Logistic Model")
        model_p = mp.ModelParameters(_input_feature_count=features,_layers=[], _activations =[], _output_classes=output_classes)
        self._train(X, Y, model_p)

    def BuildANNModel2Layers(self, X, Y, features, output_classes):
        print("Returning ANN Model with 2 Layers")
        model_p = mp.ModelParameters(_input_feature_count=features, _layers=[20, 5], _activations=["relu", "relu"], _output_classes=output_classes)
        self._train(X, Y, model_p)

    def BuildANNModel3Layers(self, X, Y, features, output_classes):
        print("Returning ANN Model with 3 Layers")
        model_p = mp.ModelParameters(_input_feature_count=features, _layers=[20, 5, 10], _activations=["relu", "relu", "relu"], _output_classes=output_classes)
        self._train(X, Y, model_p)

    def BuildConvNet1(self):
        print("Convnet 1")

    def BuildConvNet2(self):
        print("Convnet 2")
