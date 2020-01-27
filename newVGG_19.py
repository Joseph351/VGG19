import data_input as pipeline
import data_prep as data
import numpy as np
import tensorflow as tf
from tenosrflow.contrib.layers import xavier_initializer
from tensorflow import keras
class VGG19:
    def __init__(self, weights=None):
        self.decay = 0.0001
    

    def fully_connected(self, input_tensor, name, n_out, activation=True, activation_fn=tf.nn.relu):
        n_in = input_tensor.get_shape()[-1].value
        with tf.variable_scope(name):
            weights = tf.get_variable('weights', [n_in, n_out], tf.float32, 
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      regularizer=tf.contrib.layers.l2_regularizer(VGG19().decay), trainable=True)
            biases = tf.get_variable("bias", [n_out], tf.float32, tf.zeros_initializer())
            logits = tf.add(tf.matmul(input_tensor, weights), biases)
            if activation == True:
                return activation_fn(logits)
            else:
                return logits
      
    def conv(self, input_tensor, name, n_out):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            pre_conv=tf.layers.conv2d(input_tensor, n_out, [3, 3], strides=(1, 1), padding="same", activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),  
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(VGG19().decay), name=name) 
            return pre_conv
   
    def pool(self, layer, scope_name):
        with tf.variable_scope(scope_name) as scope:
            return tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=scope_name)
      
    def batch_norm(self, layer, scope_name,training=None):
        with tf.variable_scope(scope_name) as scope:
            return tf.layers.batch_normalization(layer, momentum=0.99, training=training)
    
    
    def build(self, input_tensor, n_classes, training=None):
        # fully connected
        net = input_tensor
        #keep_prob = 0.5

        net = self.conv(net, name="conv1_1", n_out=64)
        net = self.batch_norm(net, "norm1", training=training)
        net = tf.layers.dropout(net, 0.2, training=training)
        net = self.conv(net, name="conv1_2", n_out=64)
        net = self.batch_norm(net, "norm2", training=training)
        net = self.pool(net, scope_name="pool1")

        # block 2 -- outputs 56x56x128
        net = self.conv(net, name="conv2_1", n_out=128)
        net = self.batch_norm(net, "norm3", training=training)
        net = tf.layers.dropout(net, 0.2, training=training)
        net = self.conv(net, name="conv2_2", n_out=128)
        net = self.batch_norm(net, "norm4", training=training)
        net = self.pool(net, scope_name="pool2")

        # # block 3 -- outputs 28x28x256
        net = self.conv(net, name="conv3_1", n_out=256)
        net = self.batch_norm(net, "norm5", training=training)
        net = tf.layers.dropout(net, 0.3, training=training)
        net = self.conv(net, name="conv3_2", n_out=256)
        net = self.batch_norm(net, "norm6", training=training)
        net = tf.layers.dropout(net, 0.3, training=training)
        net = self.conv(net, name="conv3_3", n_out=256)
        net = self.batch_norm(net, "norm7", training=training)
        net = tf.layers.dropout(net, 0.3, training=training)
        net = self.conv(net, name="conv3_4", n_out=256)
        net = self.batch_norm(net, "norm8", training=training)
        net = self.pool(net, scope_name="pool3")

        # block 4 -- outputs 14x14x512
        net = self.conv(net, name="conv4_1", n_out=512)
        net = self.batch_norm(net, "norm9", training=training)
        net = tf.layers.dropout(net, 0.3, training=training)
        net = self.conv(net, name="conv4_2", n_out=512)
        net = self.batch_norm(net, "norm10", training=training)
        net = tf.layers.dropout(net, 0.3, training=training)
        net = self.conv(net, name="conv4_3", n_out=512)
        net = self.batch_norm(net, "norm11", training=training)
        net = tf.layers.dropout(net, 0.3, training=training)
        net = self.conv(net, name="conv4_4", n_out=512)
        net = self.batch_norm(net, "norm12", training=training)
        net = self.pool(net, scope_name="pool4")

        # block 5 -- outputs 7x7x512
        net = self.conv(net, name="conv5_1", n_out=512)
        net = self.batch_norm(net, "norm13", training=training)
        net = tf.layers.dropout(net, 0.4, training=training)
        net = self.conv(net, name="conv5_2", n_out=512)
        net = self.batch_norm(net, "norm14", training=training)
        net = tf.layers.dropout(net, 0.4, training=training)
        net = self.conv(net, name="conv5_3", n_out=512)
        net = self.batch_norm(net, "norm15", training=training)
        net = tf.layers.dropout(net, 0.4, training=training)
        net = self.conv(net, name="conv5_4", n_out=512)
        net = self.batch_norm(net, "norm16", training=training)
        net = self.pool(net, scope_name="pool5")
        net = tf.layers.dropout(net, 0.5, training=training)
        
        flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
        net = tf.reshape(net, [-1, flattened_shape], name="flatten")
        
        net = self.fully_connected(net, name="fc6", n_out=4096)
        net = self.batch_norm(net, "norm17", training=training)
        net = tf.layers.dropout(net, 0.5, training=training)
        net = self.fully_connected(net, name="fc7", n_out=4096)
        net = self.batch_norm(net, "norm18", training=training)
        net = tf.layers.dropout(net, 0.5, training=training)
        net = self.fully_connected(net, name="fc8", activation=False, n_out=n_classes)
        return net
      
      
    def loss(self, logits, label_tensor):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = label_tensor, logits = logits))
        loss = loss + tf.reduce_sum(tf.losses.get_regularization_losses())
        return loss
    
    def normalize(self, X_train, X_test):
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train - mean)/(std +1e-7)
        X_test = (X_test-mean)/(std +1e-7)
        return X_train, X_test
    
      
    def train(self, init_lrn_rate, num_classes, batch_size):

        display_step = 1
        epochs = 1024
        sub_batch_size = 9
        #(trainX, trainY), (testX, testY) = cifar100.load_data()
        #trainy = keras.utils.to_categorical(trainY, 100)
        #testy = keras.utils.to_categorical(testY, 100)
      
        #placeholders for training
        X = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
        Y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])
        flag_val = tf.placeholder(dtype=tf.bool)
         
        #losses and optimizer
        logits = self.build(X, n_classes=num_classes, training=flag_val)
        cost = self.loss(logits, Y)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
        #step = tf.Variable(0, trainable=False)
        #rate = tf.train.exponential_decay(init_lrn_rate, step, 10, 0.9995, staircase=True)
        optimizer = tf.train.AdamOptimizer(init_lrn_rate)
        #grad_step = optimizer.minimize(cost) #, global_step=step)
      
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grad_step = optimizer.minimize(cost) #, global_step=step)   
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        init = tf.global_variables_initializer()
        with tf.Session(config = config) as sess:
            sess.run(init)
            _data = data.data()
            train_paths, val_paths = _data.generate_data("/engram/nklab/imagenet-data/")
            for i in range(epochs):
                total_loss = 0
                total_acc = 0
                train_iter = pipeline.data_input_pipeline(train_paths[i],apply_batch = True,perform_shuffle=True,batch_size=batch_size)
                batch_features, batch_labels = train_iter.get_next()
                try:
                    try:
                        x, y = sess.run([batch_features["vgg19_input"], batch_labels])
                        y = keras.utils.to_categorical(y, 1001)
                    except tf.errors.OutOfRangeError:
                        pass
                except tf.errors.DataLossError:
                    pass
                num = batch_size//sub_batch_size
                for batch in range(num):
                    rand_idx = np.random.choice(len(x), sub_batch_size)
                    batch_x = x[rand_idx]
                    batch_y = y[rand_idx]
                    _, loss, acc = sess.run([grad_step, cost, accuracy], feed_dict={X: batch_x, Y: batch_y, flag_val: True})
                    total_loss += loss
                    total_acc += acc
                print("Epoch " + str(i) + ", Loss= " + \
                        "{:.6f}".format(total_loss/num) + ", Training Accuracy= " + \
                        "{:.5f}".format(total_acc/num))
                
                #run optimization on validation set after every 8 training set
                if (i+1)%8 == 0:
                    val_loss = 0
                    val_acc = 0
                    idx = (i+1)//8 - 1
                    val_iter = pipeline.data_input_pipeline(val_paths[idx], apply_batch = True, perform_shuffle=True, batch_size=390)
                    val_features, vals_labels = val_iter.get_next()
                    try:
                        try:
                            x_val, y_val = sess.run([batch_features["vgg19_input"], batch_labels])
                            y_val = keras.utils.to_categorical(y_val, 1001)
                        except tf.errors.OutOfRangeError:
                            pass
                    except tf.errors.DataLossError:
                        pass
                    num = 390//sub_batch_size
                    for batch in range(num):
                        rand_idx = np.random.choice(len(x_val), sub_batch_size)
                        batch_x_val = x[rand_idx]
                        batch_y_val = y[rand_idx]
                        v_acc, v_loss = sess.run([accuracy,cost], feed_dict={X: batch_x_val,Y : batch_y_val, flag_val: False})
                        val_loss += v_loss
                        val_acc += v_acc
                    print("Testing Loss: {:.5f}".format(val_loss/num) + ", Testing Accuracy: {:.5f}".format(val_acc/num))
                
                
                
                
if __name__ == "__main__":
    tf.reset_default_graph()
    model = VGG19()
    model.train(init_lrn_rate = 0.005, num_classes=1001, batch_size=1251)