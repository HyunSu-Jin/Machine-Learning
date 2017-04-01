
# coding: utf-8

# In[10]:

import tensorflow as tf

x_data = [[1,2,1],[1,3,2],[1,3,4],[1,5,5],[1,7,5],[1,2,5],[1,6,6],[1,7,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]

x_test = [[2,1,1],[3,1,2],[3,3,4]]
y_test = [[0,0,1],[0,0,1],[0,0,1]]
nb_classes = 3

X = tf.placeholder(tf.float32,shape=[None,3])
Y = tf.placeholder(tf.float32,shape=[None,nb_classes]) # one_hot structure

W = tf.Variable(tf.random_normal([3,nb_classes]),name="weight")
b = tf.Variable(tf.random_normal([nb_classes]),name="bias")

logits = tf.matmul(X,W)+b
hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis)))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

prediction = tf.arg_max(hypothesis,dimension=1)
correct_prediction = tf.equal(prediction,tf.arg_max(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        cost_val, _ = sess.run([cost,optimizer],feed_dict={
                X : x_data,
                Y : y_data
            })
        if step % 100 == 0:
            print(step,cost_val)
    
    print('Prediction : ',sess.run(prediction,feed_dict={
                    X : x_test
               }))
    print('Accuracy : ',sess.run(accuracy,feed_dict={
                X : x_test,
                Y : y_test
            }))


# In[ ]:



