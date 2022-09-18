---
title: Transformer的attention相关问题
tags: transformer
categories: Machine Learning
date: 2022-09-15 21:20:30
index_img: http://longls777.oss-cn-beijing.aliyuncs.com/img/image-20220915212106860.png
banner_img: http://longls777.oss-cn-beijing.aliyuncs.com/img/image-20220915212309079.png
---

#### Transformer为什么Q和K使用不同的权重矩阵生成，为何不能使用同一个值进行自身的点乘？

![image-20220915212106860](http://longls777.oss-cn-beijing.aliyuncs.com/img/image-20220915212106860.png)

![image-20220915212309079](http://longls777.oss-cn-beijing.aliyuncs.com/img/image-20220915212309079.png)

先从点乘的物理意义说，两个向量的点乘表示两个向量的相似度

Q，K，V物理意义上是一样的，都表示同一个句子中不同token组成的矩阵。矩阵中的每一行，是表示一个token的word embedding向量。假设一个句子"Hello, how are you?"长度是6，embedding维度是300，那么Q，K，V都是(6, 300)的矩阵

简单的说，K和Q的点乘是为了计算一个句子中每个token相对于句子中其他token的相似度，这个相似度可以理解为attetnion score，关注度得分。比如说 "Hello, how are you?"这句话，当前token为”Hello"的时候，我们可以知道”Hello“对于” , “, "how", "are", "you", "?"这几个token对应的关注度是多少。有了这个attetnion score，可以知道处理到”Hello“的时候，模型在关注句子中的哪些token。

这个attention score是一个(6, 6)的矩阵。每一行代表每个token相对于其他token的关注度。比如说上图中的第一行，代表的是Hello这个单词相对于本句话中的其他单词的关注度。添加softmax只是为了对关注度进行归一化。

然后我们拿这个attention score矩阵与代表着句子特征的V相乘，得到的是一个加权后结果。也就是说，原本V里的各个单词只用word embedding表示，相互之间没什么关系。但是经过与attention score相乘后，V中每个token的向量（即一个单词的word embedding向量），在300维的每个维度上（每一列）上，都会对其他token做出调整（关注度不同）。与V相乘这一步，相当于提纯，让每个单词关注该关注的部分。

好了，该解释为什么不把K和Q用同一个值了。

经过上面的解释，我们知道K和Q的点乘是为了得到一个attention score 矩阵，用来对V进行提纯。K和Q使用了不同的$W_k$,$ W_Q$来计算，可以理解为是在不同空间上的投影。

但是如果不用Q，直接拿K和K点乘的话，你会发现attention score 矩阵是一个对称矩阵。因为是同样一个矩阵，都投影到了同样一个空间，通过矩阵相乘的方式并不能获取每个token对其他token的attention score。俩个向量越相似，内积越大，当一个向量与自己做内积，再与其他不同词的向量做内积后(形成一个打分向量)，该向量经过softmax后，就会变为一个有一个位置的值很大(自己与自己相乘)，其他位置的值非常非常小的状况出现，比如[0.98,0.01,0.05,0.05]，那么，这样的得分再与V矩阵相乘后得出的加权向量就是一个基本上跟自己本身差不多的矩阵，那就失去了self attention的意义

> https://www.zhihu.com/question/319339652/answer/730848834
>
> https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3

#### Transformer计算attention的时候为何选择点乘而不是加法？两者计算复杂度和效果上有什么区别？

attention有两种形式，Add和Mul，即：
$$
score(h,s)=<v, tanh(W_1h+W_2s)> \\
score(h,s)=<W_1h,W_2s>
$$
$<>$代表矩阵点积，至于为什么要用Mul来完成Self-attention，作者的说法是为了计算更快。因为虽然矩阵加法的计算更简单，但是Add 形式套着$tanh$和$o$，相当于一个完整的隐层。在整体计算复杂度上两者接近，但是矩阵乘法已经有了非常成熟的加速实现。在$d_k$(即attention-dim）较小的时候，两者的效果接近。但是随着d增大，Add开始显著超越Mul。

![img](http://longls777.oss-cn-beijing.aliyuncs.com/img/v2-4ce33c847c71c3092e1a557c857369fb_1440w.jpg)



作者分析Mul性能不佳的原因，认为是极大的点积值将整个softmax推向梯度平缓区，使得收敛困难。也就是出现了“梯度消失”。

这才有了scaled。所以，Add是天然地不需要scaled，Mul在$d_k$较大的时候必须要做scaled。

那么，极大的点积值是从哪里来的呢?

对于Mul来说，如果$s$和$h$都分布在[0,1]，在相乘时引入一次对所有位置的$\sum$求和，整体的分布就会扩大到[0, dk]。

反过来看Add，右侧是被$tanh()$钳位后的值，分布在[-1,1]。整体分布和$d_k$没有关系。



> https://zhuanlan.zhihu.com/p/31547842











#### 为什么在进行softmax之前需要对attention进行scaled（为什么除以dk的平方根），使用公式推导解释

这取决于softmax函数的特性，如果softmax内计算的数数量级太大，会输出近似one-hot编码的形式，导致梯度消失的问题，所以需要scale

那么至于为什么需要用维度开根号，假设向量q，k满足各分量独立同分布，均值为0，方差为1，那么qk点积均值为0，方差为$d_k$，从统计学计算，若果让qk点积的方差控制在1，需要将其除以$d_k$的平方根，使得softmax更加平滑

