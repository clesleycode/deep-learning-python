Intro to Data Science with D3 
==================

Brought to you by [Lesley Cordero](http://www.columbia.edu/~lc2958) and [ADI](https://adicu.com)

## Table of Contents

- [0.0 Setup](#00-setup)
    + [0.1 Python and Pip](#01-python--pip)
    + [0.2 Libraries](#02-libraries)
- [1.0 Background](#10-background)
    + [1.1 What is Deep Learning?](#11-what-is-deep-learning)
    + [1.2 Neural Networks](#12-neural-networks)
        * [1.2.1 What is a Neural Network?](#121-what-is-a-neural-network)
    + [1.3 Backpropogation](#13-backpropogation)
    + [1.4 Hardware](#14-hardware)
        * [1.4.1 GPU](#141-gpu)
- [2.0 Python Modules](#30-python-modules)
    + [2.1 Theano](#31-theano)
        * [2.1.1 Why Theano?](#211-why-theano)
        * [2.1.2 Symbolic Variables](#212-symbolic-variables)
        * [2.1.3 Symbolic Functions](#213-symbolic-functions)
    + [2.2 TensorFlow](#32-tensorflow)
    + [2.3 Lasagne](#33-lasagne)
    + [2.4 Caffe ](#34-caffe)
- [3.0 Plotly](#30-plotly)
- [4.0 Shapely & Descartes](#50-Shapely-Descartes)
- [5.0 Final Words](#60-final-words)
    + [5.1 Resources](#61-resources)


## 0.0 Setup

This guide was written in Python 2.7.

### 0.1 Python and Pip

Download [Python](https://www.python.org/downloads/) and [Pip](https://pip.pypa.io/en/stable/installing/).

### 0.2 Libraries

```
pip install lasagne
pip install theano
pip install tensorflow
```

## 1.0 Background

### 1.1 What is Deep Learning? 

Deep Learning is a branch of machine learning that involves pattern recognition on unlabeled and unstructured data.

### 1.2 Neural Networks

#### 1.2.1 What is a Neural Network? 


##### 1.2.1.1 Input Layer

##### 1.2.1.2 Hidden Layers 

These are the intermediate layers between input and output which help the Neural Network learn the complicated relationships involved in data.

#### 1.2.1.3 Output Layer

The final output is extracted from previous two layers. For Example: In case of a classification problem with 5 classes, the output later will have 5 neurons.

#### 1.2.2 Types of Neural Networks

##### 1.2.2.1 Feedforward Neural Networks

Feedforward Neural Networks are the simplest form of Artificial Neural Networks. These networks have the three types of layers we just discussed: Input layer, hidden layer and output layer. 

##### 1.2.2.2 Convolutional Neural Networks

Convolutional neural networks are a type of feed-forward networks. CNNs perform very well on visual recognition tasks. 


##### 1.2.2.3 Recurrent Neural Networks


#### 1.3 Back-Propagation



### 1.4 Hardware


#### 1.4.1 GPU


## 2.0 Python Modules

### 2.1 Theano

Theano is a Python library that is used to define, optimize, and evaluate mathematical expressions with multi-dimensional arrays. Theano accomplishes this through its own data structures integrated with NumPy and the transparent use of the GPU. More specifically, Theano figures out which computational portions should be moved to the GPU.

TTheano isn’t actually a machine learning library since it doesn’t provide you with pre-built models to train on your data. Instead, it's a mathematical library that provides you with tools to build your own machine learning models. 


#### 2.1.1 Why Theano? 

Simply put, Theano's strong suit is efficiency. Its primary purpose is to increase the speed of computation. 

How does it accomplish this? Identifying 'small' changes like (x+y) + (x+y) to 2*(x+y), over time, make a substantial difference. Moreover, because it defines different mathematical expressions in C, it makes for much faster implementations. And because of this, Theano works well in high dimensionality problems. Lastly, it allows GPU implementation. 

#### 2.1.2 Symbolic Variables

In Theano, all algorithms are defined symbolically, meaning that they don't have an explicit value.


#### 2.1.3 Symbolic Functions 

To actually perform computations with Theano, you use symbolic functions, which can later be called with actual values.


First, we import the needed libraries: 

``` python
import theano
import numpy
```

Next, we create the building blocks of our function. Here, x is a vector, W is an array we set up with numpy, and y is the function we'll use to compute the result. 
``` python
x = theano.tensor.fvector('x')
W = theano.shared(numpy.asarray([0.2, 0.7]), 'W')
y = (x * W).sum()
```

Finally, we actually perform the computation with Theano.
``` python
f = theano.function([x], y)

output = f([1.0, 1.0])
```

If we print output, we get (0.2*1.0) + (0.7*1.0): 

```
0.9
```

### 2.2 TensorFlow

TensorFlow is an open source library for numerical computation using data flow graphs. Unlike Theano, however, TensorFlow handles distributed computing through the use of multiple-GPUs.

We'll go through a classic deep learning problem involving hand-written digit recognition, using the MNIST dataset. First, we'll implement the single layer version and follow up with a multi-layer model. 

#### 2.2.1 Single Layer Neural Network

As always, we'll need to input the needed modules. input_data is available on the github link [here]() - make sure to download it and include it in the same directory as your workspace. This will allow you to download the needed data.

``` python
import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

We'll need to create two variables to keep track of the weights and bias. Since we don't know those values yet, we initialize them to zeros. 


``` python
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
```

In this example, we also create a tensor of two dimensions to keep the information of the x points, with the following line of code:


``` python
x = tf.placeholder("float", [None, 784])
```

Next we multiply the image vector x and the weight matrix W, adding b:


``` python
y = tf.nn.softmax(tf.matmul(x,W) + b)
```

Next, we create another placeholder for the correct labels. 


``` python
y_ = tf.placeholder("float", [None,10])
```

Here, we figure out our cost function. 

``` python
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
```

Using the backpropogation algorithm, we minimize the cross-entropy using the gradient descent algorithm and a learning rate of 0.01:


``` python 
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
```

Now we can start the computation by instantiating tf.Session(). This is in charge of executing the TensorFlow operations in the available CPUs or GPUs. Then, we can execute the operation initializing all the variables:

``` python
sess = tf.Session()
sess.run(tf.initialize_all_variables())
```

Now, we can start training our model! 


``` python
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```

``` python
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
```

``` python
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
```

#### 2.2.2 Multi-Layer Neural Network

``` python
import tensorflow as tf
import input_data

``` python 
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
```

``` python
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
```

``` python
x_image = tf.reshape(x, [-1,28,28,1])
```

``` python
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
```

``` python
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
```

``` python
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
```

``` python
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
```

``` python
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
```
``` python
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()

sess.run(tf.initialize_all_variables())

for i in range(200):
batch = mnist.train.next_batch(50)
if i%10 == 0:
train_accuracy = sess.run( accuracy, feed_dict={ x:batch[0], y_: batch[1], keep_prob: 1.0})
print("step %d, training accuracy %g"%(i, train_accuracy))
sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"% sess.run(accuracy, feed_dict={ 
x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
```


### 2.3 Lasagne

Lasagne is a lightweight library used to construct and train networks in Theano.


### 2.4 Caffe

Caffe is a deep learning framework developed written in C++. 


## 3.0 Shapely & Descartes


``` python
```

``` python 
```


``` python
```


``` python
```


``` python
```


``` python
```

``` python
```

## 4.0 




``` python


```



``` python
```

``` python
```


``` python
```


## 5.0 Final Words


### 5.1 Resources

[]() <br>
[]()
