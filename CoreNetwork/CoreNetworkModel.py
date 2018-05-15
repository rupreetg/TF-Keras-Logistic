import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.losses import categorical_crossentropy
import numpy as np
import os


class CoreNetwork(object):

    def __init__(self,  modelparameters):

        self.modelparameters = modelparameters
        self.model = self._build_model()

    def _build_model(self):
        _activations = self.modelparameters.activations
        _units = self.modelparameters.layers
        _input_dim = self.modelparameters.inputfeaturecount
        _classes = self.modelparameters.outputclasses
        model = Sequential()

        # specific case of logistic regression
        #that
        if len(_activations) == 0 :

            if _classes == 2:
                _layer = Dense(_classes, input_dim=_input_dim, activation='sigmoid')
            else:
                _layer = Dense(_classes, input_dim=_input_dim, activation='softmax')
            
            model.add(_layer)
            return model

        for i in range(len(_activations)):
            if i == 0:
                _layer = Dense(units=_units[i],input_dim=_input_dim,activation=_activations[i])
                model.add(layer=_layer)
            else:
                _layer = Dense(units=_units[i],activation=_activations[i])
                model.add(layer=_layer)
        if _classes == 2:
            _layer = Dense(_classes, input_dim=_input_dim, activation='sigmoid')
        else:
            _layer = Dense(_classes, input_dim=_input_dim, activation='softmax')
            model.add(_layer)
            return model

    def _transform(self,X_):
        Yhat = self.model(X_)
        return Yhat

    def _loss(self,X_,Y_):
        Yhat = self._transform(X_)
        loss = tf.reduce_mean(categorical_crossentropy(y_true=Y_, y_pred=Yhat))
        tf.summary.scalar("cost_function", loss)
        accuracy_ = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Yhat,1), tf.argmax(Y_,1)),tf.float32))
        tf.summary.scalar("accuracy", accuracy_)
        return loss

    def train(self, sess,X, Y, steps, optimizer, minibatch_size=100,optimizer_name="SGD"):
        X_ = self.model.inputs[0]
        Y_ = self.model.outputs[0]
        #_predictions = self._transform()
        loss_ = self._loss(X_,Y_)
        train_ = optimizer.minimize(loss_)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        summary_op = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        root_path = os.getcwd()
        logs_path = os.path.join(root_path,"Logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        newdir = os.path.join(logs_path, optimizer_name)
        if not os.path.exists(newdir):
            os.makedirs(newdir)

        writer = tf.summary.FileWriter(newdir, graph=tf.get_default_graph())
        # Fit all training data
        losses = []
        loss_val = []
        stepctr = 0
        for step in range(steps):
            n_batch = X.shape[0] // minibatch_size + (X.shape[0] % minibatch_size != 0)
            for batch in range(n_batch):
                if batch != (n_batch - 1):
                    batch_X = X[batch:batch + minibatch_size - 1]
                    batch_Y = Y[batch:batch + minibatch_size - 1]
                else:
                    batch_X = X[batch:]
                    batch_Y = Y[batch:]

                _, c , summary = sess.run([train_, loss_,summary_op], feed_dict={X_: batch_X, Y_: batch_Y})
                loss_val += c / n_batch
                stepctr = stepctr + 1
                writer.add_summary(summary, stepctr)
            writer.flush()


            losses.append(loss_val)

            if step % 50 == 0:
                print("Step {} of {}, logloss {}".format(step, steps, loss_val))
        return losses

