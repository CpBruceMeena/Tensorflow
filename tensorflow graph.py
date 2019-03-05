import tensorflow as tf

n1 = tf.constant(1)
n2 = tf.constant(2)

n3 = n1+n2
with tf.Session() as sess:
    result = sess.run(n3)
    print(result)
print(n3)  

# the memory of default graph is somewhere differnet than the memory location of g
print(tf.get_default_graph())

g = tf.Graph()
print(g)

# the memory of graph_one and default graph is same
graph_one = tf.get_default_graph()
print(graph_one)

graph_two = tf.Graph()

with graph_two.as_default():
    print(graph_two is tf.get_default_graph()) #true
#Outside the with loop graph_one is the default graph but within the with loop graph_two is set as default graph
print(graph_two is tf.get_default_graph()) #False

    