##
#This is first test program written in tensorflow to check the installation
##

#imported tensorflow as tf
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))