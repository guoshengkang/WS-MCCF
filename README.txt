# MCCF

Source code for AAAI2020 paper ["**Multi-Component Graph Convolutional Collaborative Filtering**"](https://arxiv.xilesou.top/abs/1911.10699)



## Environment Settings

* Python == 3.6.9
* torchvision == 0.4.2
* numpy == 1.17.3
* scikit-learn == 0.21.3



## Parameter Settings

- epochs: the number of epochs to train	训练的迭代次数
- lr: learning rate	学习率
- embed_dim: embedding dimension	嵌入尺寸
- N: a parameter of L0, the default is the number of triples	参数为L0，默认为三组数
- droprate: dropout rate	下坠率
- batch_size: batch size for training	训练的批处理大小



## Files in the folder

~~~~
MCCF/
├── run.py: training the model	训练模型
├── utils/
│   ├── aggregator.py: aggregating the feature of neighbors	聚合邻居的特征
│   ├── l0dense.py: implementation of L0 regularization for a fully connected layer	完整连接层的L0正则化的实现
│   ├── attention.py: implementation of the node-level attention	节点级注意力的实现
│   ├── encoder.py: together with aggregator to form the decomposer	与聚合器一起组成分解器
│   └── combiner.py: implementation of the combiner	合成器的实现
├── datasets/
│   ├── yelp/
│   │		├── business_user.txt
│   │   ├── preprocess.py: data preprocessing example	数据预处理示范
│   │   └── _allData.p
│   ├── amazon/ 
│   │   ├── user_item.dat
│   │   └── _allData.p
│   └── movielens/
│   		├── ub.base
│       ├── ub.test
│   		├── ua.base
│       ├── ua.test
│   		├── u5.base
│       ├── u5.test
│   		├── u4.base
│       ├── u4.test
│   		├── u3.base
│       ├── u3.test
│   		├── u2.base
│       ├── u2.test
│   		├── u1.base
│       ├── u1.test
│   		├── u.data
│       ├── u.user
│       ├── u.item
│       └── _allData.p
└── README.md
~~~~



## Data

### Input training data	输入训练数据

* u_adj: user's purchased history (item set in training set)	用户购买历史(训练集中的项目集)
* i_adj: user set (in training set) who have interacted with the item	与项目交互的用户集(在训练集中)
* u_train, i_train, r_train: training set (user, item, rating)	训练集(用户，项目，等级)
* u_test, i_test, r_test: testing set (user, item, rating)		测试集(用户，项目，等级)



### Input pre-trained data	输入预先训练过的数据

* u2e, i2e: for small data sets, the corresponding vectors in the rating matrix can be used as initial embeddings; for large data sets, we recommend using the embeddings of other models (e.g., GC-MC) as pre-training, which greatly reduces the complexity.

对于小数据集，可以使用评级矩阵中对应的向量作为初始嵌入;对于大数据集，我们建议使用其他模型的嵌入(如GC-MC)作为预训练，这样可以大大降低复杂性。


## Basic Usage

~~~
python run.py 
~~~

HINT: the sampling thresholds in aggregator.py change with dataset.	注：采样阈值随数据集的变化而变化。




# Reference

```
@article{wang2019multi,
  title={Multi-Component Graph Convolutional Collaborative Filtering},
  author={Wang, Xiao and Wang, Ruijia and Shi, Chuan and Song, Guojie and Li, Qingyong},
  journal={arXiv preprint arXiv:1911.10699},
  year={2019}
}
```