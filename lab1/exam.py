import  tensorflow as tf

a = tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)

add = tf.add(a,b)

sess = tf.Session()

print (sess.run(add,feed_dict={a:[1,2],b:[4,5]}))
