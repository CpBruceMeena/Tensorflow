import tensorflow as tf

sess = tf.InteractiveSession()

my_tensor = tf.random_uniform((4,4), 0, 1)

my_var = tf.Variable(initial_value = my_tensor)

# We need to initialize variables

#this tf.global_variables_initializer() is important to run the variables
init = tf.global_variables_initializer()

#and then we need to run the sess.run(init) to let the initializer run
sess.run(init)
print(sess.run(my_var))

#Dont forget to use print while using sess.run() for printing data
  
#dont forget to use tf. in all the places 
# We are using none because we may not know the no of samples in the batches that we will use for testing
ph = tf.placeholder(tf.float32, shape = (None, 5)) 

#Placeholder will create a block of memory for that we'll use later i.e. mainly we use it in the session block where we feed it with the data
# we can use the placeholder earlier without even defining the values that it may take