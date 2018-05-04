import CoreNetwork.CoreNetworkModel as cn
import CoreNetwork.ModelParameters as mp
import CoreNetwork.Optimizers as asop
import tensorflow as tf
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

learning_rate = 0.01

train_x,train_y, samplesize, features, num_classes = PrepareData()

model_p = mp.ModelParameters(input_feature_count=features, layers=[], activations=[], outputclasses=num_classes)
opts = [
    tf.train.GradientDescentOptimizer(learning_rate=learning_rate),
    # asop.ASGradientDescentOptimizer(base_learning_rate=learning_rate),
    # tf.train.RMSPropOptimizer(learning_rate=learning_rate),
    # asop.ASRMSPropOptimizer(base_learning_rate=learning_rate),
    # tf.train.AdamOptimizer(learning_rate=learning_rate),
    # tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=.9),
    # tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=.9, use_nesterov=True),
    # tf.train.AdagradOptimizer(learning_rate=learning_rate)
]
opt_names = ['SGD',
             # 'SGD+AS', 'RMSProp', 'RMSProp+AS', 'ADAM', 'SGD+M', 'SGD+NM', 'Adagrad'
             ]
epochs = 10
minibatch_size = 200
losses=[]

for i, opt in enumerate(opts):
    init1 = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init1)
        network = cn.CoreNetwork( model_p)
        loss = network.train(sess,train_x,train_y, epochs, opt, minibatch_size, opt_names[i])
        losses.append(loss)
    tf.reset_default_graph()