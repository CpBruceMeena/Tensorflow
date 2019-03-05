import tensorflow as tf

import numpy as np

hello = tf.constant("Hello")
world = tf.constant("world")

#with will automatically close the session
with tf.Session() as sess:
    result = sess.run(hello+world)

a = tf.constant(10)
b = tf.constant(29)

with tf.Session() as sess:
    result = sess.run(a+b)
    
const = tf.constant(10)
fill_mat = tf.fill((4,4), 10)
myzeros = tf.zeros((4,4))
myones = tf.ones((4,4))
myrandn = tf.random_normal((4,4), mean = 0, stddev = 1.0)
myrandu = tf.random_uniform((4,4), minval = 0, maxval = 1)
my_ops = [const, fill_mat, myzeros, myones, myrandn, myrandu]

#It will help us reduce the task of writing with tf.Session() as sess:
sess = tf.InteractiveSession()
'''
for op in my_ops:
    print(sess.run(op))
    #eval is same as using sess.run 
    print(op.eval())
    print('\n')
''' 
a = tf.constant([[1, 2], 
                [3, 4]])
print(a.get_shape())

b = tf.constant([[10], [100]])
print(b.get_shape())

result = tf.matmul(a,b)
print(sess.run(result))

print(result.eval())