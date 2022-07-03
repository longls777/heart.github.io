---
title: word2vec详解
tags: nlp
categories: ai
date: 2022-07-03 9:45:30
index_img: http://longls777.oss-cn-beijing.aliyuncs.com/img/v2-6224dddd34017aee696a88289245604c_720w.jpg
banner_img: http://longls777.oss-cn-beijing.aliyuncs.com/img/v2-6224dddd34017aee696a88289245604c_720w.jpg
math: true
---

## 跳字模型（skip-gram）

跳字模型的**概念**是在每一次迭代中都取一个词作为中心词汇，尝试去预测它一定范围内的上下文词汇

![v2-6224dddd34017aee696a88289245604c_720w](http://longls777.oss-cn-beijing.aliyuncs.com/img/v2-6224dddd34017aee696a88289245604c_720w.jpg)

目标函数为
$$
\prod_{t=1}^{T}{\prod_{-m\le j\le m,j=0}{P(W^{(t+j)}|w^{(t)})}}
$$

**该函数又称为似然函数，这里表示在给定中心词的情况下，在2m窗口内的所有其他词出现的概率（T表示词库里所有词的总数）。我们的目标是要通过调节参数，从而最大化这个函数（因为这个函数越大，表示与实际情况越吻合）。**（注意：这里假设给定中心词的情况下背景词的生成相互独立）

另外，根据平时的习惯，我们通常喜欢最小化损失函数，而不是最大化损失函数。因此我们对该函数取负对数，且除以T，得到新的损失函数（对数似然函数）：
$$
J(\theta)=-\frac{1}{T}\prod_{t=1}^{T}{\prod_{-m\le j\le m}{\log{P(W_{t+j}|w_{t})}}}
$$
