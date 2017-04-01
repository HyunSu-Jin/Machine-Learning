
# coding: utf-8

# In[2]:

# multinomial classifier
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True) # class label을 one_hot 형식으로 받는다.
nb_classes = 10


# In[9]:

X = tf.placeholder(tf.float32,shape=(None,784))
Y = tf.placeholder(tf.float32,shape=(None,nb_classes))

W = tf.Variable(tf.random_normal([784,nb_classes]), name ="weight")
b = tf.Variable(tf.random_normal([nb_classes]), name = "bias")

logits = tf.matmul(X,W) +b
hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis))) # cross-entropy function

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

prediction = tf.arg_max(hypothesis,dimension = 1)
correct_prediction = tf.equal(prediction,tf.arg_max(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
## epoch, batch_size
# epoch : one forward pass and one backward pass of all training examples.
# batch size : the number of traing examples in one forward/backward pass. The higher the batch size, the more memory space you will need
training_epochs = 15
batch_size = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size) # batch를 수행해야하는 횟수
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) # dataSet에서 batch만큼의 데이터를 가져옴
            cost_val, _ = sess.run([cost,optimizer],feed_dict={
                    X : batch_xs,
                    Y : batch_ys
                })
            avg_cost += cost_val / total_batch
        
        print('Epoch:',epoch+1,'\ncost: ',avg_cost)
        
    ## END training
    
    print('Accuracy : ',accuracy.eval(session=sess,feed_dict={
                X : mnist.test.images,
                Y : mnist.test.labels
            }))


# In[ ]:



