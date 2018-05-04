import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activiation
from keras.losses import categorical_crossentropy

class CoreNetwork(object):
    def __init__(self, _X, _Y, _modelParameters):
        self.X = _X
        self.Y = _Y
        self.modelParameters = _modelParameters
        self.model = _build_model()
    
    def _build_model(self):
        _activations = self.modelParameters.activiations
        _neurons = self.modelParameters.layers #units
        _input_dim = self.modelParameters.input_feature_count
        _output_classes = self.modelParameters.output_classes

        model = Sequential()
        if len(_activations) == 0 :

            if _classes == 2:
                _layer = Dense(_output_classes, input_dim=_input_dim, activation='sigmoid')
            else:
                _layer = Dense(_output_classes, input_dim=_input_dim, activation='softmax')
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
            _layer = Dense(_output_classes, input_dim=_input_dim, activation='sigmoid')
        else:
            _layer = Dense(_output_classes, input_dim=_input_dim, activation='softmax')
        model.add(_layer)
        return model

    def _transform(self):
        #assuming single input...
        _output_layer = self.model.layers[-1] #get the last one..
        _y_Hat = _output_layer.output
        return _yHat

    def _loss(self):
        Y = self.model.input[1]
        YHat = self._transform()
        loss = tf.reduce_mean(categorical_crossentropy(y_true = Y, y_pred = YHat))
        return loss 

    def train(self, _session, _noOfEpocs, _optimizer, _optimizer_name, _miniBatchSize = 100):
        #Create placeholder for X_ & Y_ 
        X_ = tf.placeholder(tf.float32, (None, self.modelParameters.input_feature_count, "FeedForward_X")
        Y_ = tf.placeholder(tf.float32, (None, self.modelParameters.output_classes), "FeedForward_Y")
        _predictions = self._transform()
        loss_ = self._loss()

        train_ = optimizer.minimize(loss_)

            global_step = tf.Variable(0, name='global_step', trainable=False)

    summary_op = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    logs_path = "D:/ML-Lab/tf-keras/Logs"
    newdir = os.path.join(logs_path, optimizer_name)
    if not os.path.exists(newdir):
        os.makedirs(newdir)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    writer = tf.summary.FileWriter(newdir, graph=tf.get_default_graph())
    # Fit all training data
    losses = []
    loss_val = []
    stepctr = 0
    for step in range(steps):
        n_batch = self.X.shape[0] // minibatch_size + (self.X.shape[0] % minibatch_size != 0)
        for batch in range(n_batch):
            if batch != (n_batch - 1):
                batch_X = self.X[batch:batch + minibatch_size - 1]
                batch_Y = self.Y[batch:batch + minibatch_size - 1]
            else:
                batch_X = self.X[batch:]
                batch_Y = self.Y[batch:]
            train_val, c, summary = sess.run([train_, loss_, summary_op], feed_dict={X_: batch_X, Y_: batch_Y})
            loss_val += c / n_batch
            stepctr = stepctr + 1
            writer.add_summary(summary, stepctr)
        writer.flush()

        # i_batch = (step % n_batch) * minibatch_size
        # batch_X = X[i_batch:i_batch + minibatch_size]
        # batch_Y = Y[i_batch:i_batch + minibatch_size]
        # train_val, loss_val, summary = sess.run([train_, loss_, summary_op], feed_dict={X_: batch_X, Y_: batch_Y})
        losses.append(loss_val)

        if step % 50 == 0:
            print("Step {} of {}, logloss {}".format(step, steps, loss_val))
    return losses


