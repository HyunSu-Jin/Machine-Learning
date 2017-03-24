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
