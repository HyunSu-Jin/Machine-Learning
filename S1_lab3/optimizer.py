import tensorflow  as tf

x_train= [1,2,3]
y_train = [1,2,3]

X = tf.placeholder(tf.float32,shape=[None])
Y = tf.placeholder(tf.float32,shape=[None])

W = tf.Variable(tf.random_normal([1]),name="weight")

hypothesis = W * X
cost = tf.reduce_sum(tf.square(W*X -Y))

learing_rate = 0.1
gradient = tf.reduce_mean(X * (W*X - Y))
descent = W - learing_rate * gradient
update = W.assign(descent)
#update = tf.assign(W,descent)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
	 sess.run(update,feed_dict={X: x_train,Y:y_train})
	 cost_val = sess.run(cost,feed_dict={X:x_train,Y:y_train})
	 W_val = sess.run(W)
	 print(step,cost_val,W_val)
