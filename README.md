# Machine-Learning
- 모두를 위한 딥러닝 강좌 + 개인프로젝트 저장
- https://www.youtube.com/user/hunkims
## About Reinforcement Leaning,
- https://github.com/HyunSu-Jin/Reinforcement-Learning
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

![lab2](/lab2/result/lab2_result.png)

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


![lab4-1](/lab4-1/result/lab4-1_result.png)


## Lab4-2
Loading Data from File
1. 일반적으로 .csv 형식 파일으로부터 데이터를 읽어들이고자 하는 경우 numpy lib 이용해 XY matrix을 만들고 여기서 부터 input data : X , Output data : Y 를 정의한다.
2. 그러나, 읽어들어야 할 .csv 파일이 너무 많아 파일을 모두 읽어들이기 위해 요구되는 메모리가 부족한 상황이 생긴 경우, 다음과 같은 Queue Running 을 이용한다
### Queue Running
![lab4-1](/lab4-2/result/queueRunning.png)

1. A,B,C,... 등의 대용량의 데이터를 읽어 들인다.(셔플가능)
2. 읽어들인 데이터를 Queue에 저장한다.
3. Reader를 이용해 큐에 접근하여 한개씩 데이터를 읽어들인다.
4. 여기서 읽어들일 데이터 형식을 다음과 같이 지정한다.
<pre><code>
record_defaults = [[0.],[0.],[0.],[0.]]
xy = tf.decode_csv(value,record_defaults=record_defaults)
</code></pre>
5. Decoder에 의해 파싱된 데이터는 별도의 Queue에 다시 저장된다.
6. 저장된 데이터는 batch에 의해 임의의 size만큼 data를 뽑아낸다.
7. 트레이닝훈련을 진행한 후, 아래의 코드로 결과데이터를 예측한다.
<pre><code>
hypo1 = sess.run(hypothesis,feed_dict={
	X : [[70,60,50]]
	})
hypo2 = sess.run(hypothesis,feed_dict={
	X : [[40,30,10],[90,95,92]]
	})

print("First case : ",hypo1)
print("Second case :",hypo2)
</code></pre>
8. 실행결과
![lab4-1](/lab4-2/result/lab4-2_result.png)

## Lab5
Logistic(regression) classifier
주어진 supervised leaning이 Input variables X1,X2,...Xn에 대한 Binary Classification인 경우, 다음과 같은 방법으로 모델을 구현할 수 있다.
1. linear regression인 경우, hypothesis, 예측값을 다음과 같이 구현하였다.
<pre><code>
hypothesis = tf.matmul(X,W)+b
</code></pre>
그런데, classifier에 경우는 위 수식이 적절치 않은데 그 이유는 다음과 같다.
output의 결과인 Y가 0,1로 Binary한 데 반해, X값은 범위가 매우 다양하다. 따라서, (X,Y)가 (3,1) (100,1) (1000,1) 이 성립할 수 있다. 이 때, linear regression에서 사용하는 hypothesis를 사용하면 lienar하게 함수가 그려지기 때문에 Y를 예측하기 위한 적절한 X가 구해지지 않는다.
2. 따라서, 다음과 같은 sigmod 꼴의 hypothesis를 구현한다.
<pre><code>
hypothesis = tf.sigmoid(tf.matmul(X,W)+b)
</code></pre>
이 모델은 z = W*X , hypothesis,g(z) = 1 / 1+ e^-z 꼴로 표현되며 0 < g(z) <1 의 값을 가진다.
3. 따라서, 위 hypothesis를 통해 다음과 같은 cost function을 도출할 수 있다.
cost function은 실제값,Y 와 예측값,hypothesis간의 difference를 의미하는 것인데, Y =1 인경우 cost(g(z)) = -log(g(z)) , Y = 0인 경우 cost(g(z)) = -log(1-g(z))로써 표현 할 수있다. 위 수식은 그래프를 그려봄으로써 왜 이 모습을 띄게 되었는지 이해할 수 있다.
<pre><code>
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
</code></pre>
4. 모델의 학습정도를 파악하기 위해 다음과 같은 파라미터를 정의하고 trainning data set를 입력했다.(.csv형식 파일)
<pre><code>
predicted = tf.cast(hypothesis > 0.5,dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast( tf.equal(predicted,Y) ,dtype=tf.float32))
h , c, a = sess.run([hypothesis,predicted,accuracy],feed_dict={
		X : x_data,
		Y: y_data
		})
</code></pre>
5. 실행결과
![lab5]=(/lab5/result/lab5_result2.png)
