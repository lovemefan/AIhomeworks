# 从零开始搭建神经网络-ANN

[github](https://github.com/lovemefan/AIhomeworks/tree/master/week07)地址


## 1.1 神经网络

​		每层都有若干个节点，每个节点就好比一个神经元（neuron），它与上一层的每个节点都保持着连接，且它的输入是上一层每个节点输出的线性组合。每个节点的输出是其输入的函数，把这个函数叫激活函数（activation function）。人工神经网络通过“学习”不断优化那些线性组合的参数，它就越有能力完成人类希望它完成的目标。

![image-20201112195759523](https://pan-lovemefan.oss-cn-shenzhen.aliyuncs.com/img/image-20201112195759523.png)

除了输入层（第1层）以外，第 l+1 层第i个节点的输入为：

$ z^{(l+1)}_i = W^{(l)}_{i1}a^{(l)}_{1} + W^{(l)}_{i2}a^{(l)}_{2}+ \dots +W^{(l)}_{is_l}a^{(l)}_{s_l} + b^{(l)}_{i}$ 

其中$s_l$表示第$l$层的节点数


第$l+1$层第和i个节点的输出为

$a^{(l+1)}_{i} = f(z^{(l+1)}_{i}) $



## 1.2 mnist手写数字数据集

下载数据集http://yann.lecun.com/exdb/mnist/

其中有四个文件：


* train-images-idx3-ubyte.gz: training set images (9912422 bytes)`
* train-labels-idx1-ubyte.gz: training set labels (28881 bytes)
* t10k-images-idx3-ubyte.gz:  test set images (1648877 bytes)
* t10k-labels-idx1-ubyte.gz:  test set labels (4542 bytes)


数据集中有60000个训练集，10000个测试集



## 1.3 正向传播





## 1.4 反向传播

