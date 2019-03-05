#Tensorboard
import tensorflow as tf
a = tf.add(3,6)

b = tf.add(2,3)

c = tf.multiply(a, b)

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./output", sess.graph)
    print(sess.run(c))
    writer.close()
     
