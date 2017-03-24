# Machine-Learning
- 모두를 위한 딥러닝 강좌 + 개인프로젝트 저장
- https://www.youtube.com/user/hunkims

## Lab1
Tensor의 정의, Tensor간 add,mul,sub등 연산을 수행하는 방법

## Lab2
supervised learning : linear regression을 수행.
1. trainning data set : X,Y 를 정의
2. hypothesis : trainning data set을 기반으로하는 linear function Model, H(x) = W*X +b
3. cost(W,b) = tf.reduce_mean(tf.square(hypothesis-Y)) 이며, 이를 최소화시키는것이 목적
4. optimizer 를 GradientDescentOptimizer 사용 > cost를 minimize
5. Conduct trainning
<pre><code>
for step in range(2001):
	cost_val, W_val,b_val,_ = sess.run([cost,W,b,train],feed_dict={X:[1,2,3,4,5],Y:[2.1,3.1,4.1,5.1,6.1]})
	if(step %20 ==0):
		print(step,cost_val,W_val,b_val)
</code></pre>
6. result

![lab2](/S1_lab2/result/lab2_result.png)

## Lab3
tensorflow optimizer의 동작원리
<pre><code>
learing_rate = 0.1
gradient = tf.reduce_mean(X * (W*X - Y))
descent = W - learing_rate * gradient
update = W.assign(descent)
</code></pre>
1. cost(W)를 미분하여 기울기를 얻는다.
2. W := W - learning_rate * 기울기로 변경된다.
> 즉, cost(W),cost function의 기울기가 점차 감소되는 방향으로 W값이 이동한다.
> 기울기의 값이 0이 되는지점. 즉, cost(W)를 미분한 값이 0이 되는지점에서 optimal value를 얻는다.

## Lab4-1
Multi-variable linear regression
1. 원하는 데이터 Y가 여러개의 변수에 의해 결정되는 경우.
   즉, Y = f(x1,x2,x3)와 같은 형태
2. 이러한 경우 hypothesis = x1w1 + x2w2 + x3w3 +b 꼴로 나타낸다.
3. 위 솔루션과 동일한 방법으로 multi-variable 문제를 해결할 수는 있지만 다음과 같이 코드가 정적이며 dynamic한 input에 대해 프로그램이 유연하지 못하다.

4. 솔루션--1
<pre><code>
x1_data = [73.,93.,89.,96.,73.]
x2_data  = [80.,88.,91.,98.,66.]
x3_data = [75.,93.,90.,100.,70.]

X1 = tf.placeholder(tf.float32)
X2 = tf.placeholder(tf.float32)
X3 = tf.placeholder(tf.float32)

y_data = [152.,185.,180.,196.,142.]
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([1]),name="weight1")
W2 = tf.Variable(tf.random_normal([1]),name="weight2")
W3 = tf.Variable(tf.random_normal([1]),name="weight3")
b = tf.Variable(tf.random_normal([1]),name="bias")

hypothesis = X1 * W1 + X2 * W2 + X3 * W3 +b
</code></pre>
따라서,
5. Input variable로써 N개의 multi-variable,
6. Output variable로써 M개의 multi-variable,
7. H개의 # of instance 가 주어진 경우
<pre><code>
x_data = [
	[73,80,75],
	[93,88,93],
	[89,91,90],
	[96,98,100],
	[73,66,70]
]

y_data = [[152],[185],[180],[196],[142]]

X = tf.placeholder(tf.float32,shape=[None,3])
Y = tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([3,1]),name="weight")
b = tf.Variable(tf.random_normal([1]),name="bias")
</code></pre>
이를 (MxN) * (NxM) = (HxM) 꼴의 행렬연산으로 표현할 수 있다.
8. H(X) = XW +b
<pre><code>
hypothesis = tf.matmul(X,W)+b
</code></pre>
그러므로, 각 multi-variable에 대한 weight matrix ,W는
Shape[N,M] 의 Tensor로써 나타내야 한다.
9. Example
<pre><code>
W = tf.Variable(tf.random_normal([3,1]),name="weight")
b = tf.Variable(tf.random_normal([1]),name="bias")
</code></pre>
위와 같은 형식을 이용해서 multi-variable에 대한 linear regression을 유연하게 구현하여 
insteresting value인 1) cost  2) hypothesis 를 다음과 같이 얻을 수 있다.
10. result

![lab4-1](/S1_lab4-1/result/lab4-1_result.png)
