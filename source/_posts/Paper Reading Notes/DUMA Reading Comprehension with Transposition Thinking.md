---
title: DUMA Reading Comprehension with Transposition Thinking
tags: MCQA
categories: Paper Reading Notes
date: 2022-09-29 15:48:00
index_img: http://longls777.oss-cn-beijing.aliyuncs.com/img/image-20220929162023650.png
banner_img: http://longls777.oss-cn-beijing.aliyuncs.com/img/image-20220929162023650.png
math: true
---

**标题：**《DUMA: Reading Comprehension with Transposition Thinkin》

**论文来源：** IEEE 2021

**原文链接：** https://arxiv.org/pdf/2001.09415v5.pdf

**源码：**https://github.com/pfZhu/duma_code

## 概述

当前用来解决MRC问题的一般是两层结构：

1. 预训练模型的encoder表示层，如ALBERT
2. 对齐网络层，用来捕捉文本，问题，回答，三元组之间的关系，例如OCN和DCMN

![可以看到ALBERT非常强大](http://longls777.oss-cn-beijing.aliyuncs.com/img/image-20220929155842159.png)

该文不注重设计复杂的对齐网络，而是从人类的思考方式中寻找经验：

1. 快速阅读全文和问题答案，建立整体印象
2. 基于问题和选项的重复信息，从文章中寻找支持证据
3. 基于文章信息，考虑问题和选项，并确定正确答案

该文使用attention来实现这种双向的换位思考模式，称之为passage-to-question attention 或者question-to-passage attention

![image-20220929162023650](http://longls777.oss-cn-beijing.aliyuncs.com/img/image-20220929162023650.png)



