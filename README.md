# DIN

## 数据处理

### 基础数据

论文中用的是Amazon Product Data数据，包含两个文件：reviews_Electronics_5.json, meta_Electronics.json.

文件格式链接中有说明，其中reviews主要是用户买了相关商品产生的上下文信息，包括商品id, 时间，评论等。meta文件是关于商品本身的信息，包括商品id, 名称，类别，买了还买等信息。

### 数据处理

这里主要涉及到两个脚本，他们的功能分别是提取原始数据、处理数据。

#### 1_convert_pd.py

（1）将reviews_Electronics_5.json转换成dataframe,列分别为reviewID ,asin, reviewerName等，

（2）将meta_Electronics.json转成dataframe,并且只保留在reviewes文件中出现过的商品，去重。

（3）转换完的文件保存成pkl格式。

#### 2_remap_id.py

（1）将reviews_df只保留reviewerID, asin, unixReviewTime三列；

（2）将meta_df保留asin, categories列，并且类别列只保留三级类目；（至此，用到的数据只设计5列，（reviewerID, asin, unixReviewTime），（asin, categories））;

（3）用asin,categories,reviewerID分别生产三个map(asin_map, cate_map, revi_map),key为对应的原始信息，value为按key排序后的index（从0开始顺序排序），然后将原数据的对应列原始数据转换成key对应的index;各个map的示意图如下：

![img](https://i.loli.net/2020/05/17/aRotHCEeGgU7w6b.png)

（4）将meta_df按asin对应的index进行排序，如图：

![img](https://i.loli.net/2020/05/17/QeSLqfnoz32TxPv.png)

（5）将reiviews_df中的asin转换成asin_map中asin对应的value值，并且按照reviewerID和时间排序。如图：

![img](https://i.loli.net/2020/05/17/LD7uABxSFPHElta.png)

（6）生成cate_list, 就是把meta_df的'categories'列取出来。

### 生成训练集和测试集

（1）将reviews_df按reviewerID进行聚合，聚合的结果就是用户的点击序列。

![img](https://i.loli.net/2020/05/17/GwqPMyzXFa87LD5.png)

（2）将hist的asin列作为每个reviewerID(也就是用户)的正样本列表（pos_list）,注意这里的asin存的已经不是原始的item_id了，而是通过asin_map转换过来的index。负样本列表(neg_list)为在item_count范围内产生不在pos_list中的随机数列表。

#### 训练集

因为当前点击的商品只和该用户之前的点击记录有关，因此我们要获取该用户之前的历史数据。

- 正样本：(reviewerID, hist, pos_item, 1)。
- 负样本：(reviewerID, hist, neg_item, 0)。注意负样本的逻辑是在item_count范围内不再pos_list中的随机数列表。

#### 测试集

对于每个pos_list和neg_list的最后一个item，用做生成测试集，测试集的格式为（reviewerID, hist, (pos_item, neg_item)）

#### 举例

比如对于reviewerID=0的用户，她的pos_list是[13179, 17993, 28326, 29247, 62275]，那么生成的训练集和测试集分别是：

![img](https://i.loli.net/2020/05/17/zRVAYax83DC2PIt.png)

![img](https://i.loli.net/2020/05/17/PCQGkszIrcv7tAq.png)

## 代码解析

作者在写DIN代码model部分时，是将train、eval和test放到一起来写，看起来不太易懂，因此这里我将这三个部分分开，更好的分析模型运作的过程。

### train代码

```python
self.u = tf.placeholder(tf.int32, [None, ])  # [B],用户id
self.i = tf.placeholder(tf.int32, [None, ])  # [B]，正样本
self.y = tf.placeholder(tf.float32, [None, ])  # [B]，label
self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T]，用户浏览记录
self.sl = tf.placeholder(tf.int32, [None, ])  # [B]，真实记录数量
self.lr = tf.placeholder(tf.float64, [])  #学习率

hidden_units = 128

#初始化几个embedding矩阵。注意这里对cate也进行embedding化，item的最终embedding=id_emb+cate_emb
user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units]) #user的emb，模型中没用到
item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2]) #item的emb
item_b = tf.get_variable("item_b", [item_count], initializer=tf.constant_initializer(0.0)) #item的bias
cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2]) #cate的emb
cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

#获取正样本的embedding
ic = tf.gather(cate_list, self.i) #取出正样本对应的cate
i_emb = tf.concat(values=[
      tf.nn.embedding_lookup(item_emb_w, self.i),
      tf.nn.embedding_lookup(cate_emb_w, ic),
], axis=1)
#我的理解是充当item固有属性了，bias的意思，是一个随机数，可训练
i_b = tf.gather(item_b, self.i)
#获取历史行为的embedding
hc = tf.gather(cate_list, self.hist_i)
h_emb = tf.concat([
      tf.nn.embedding_lookup(item_emb_w, self.hist_i),
      tf.nn.embedding_lookup(cate_emb_w, hc),
], axis=2)
#将正样本embedding和历史行为embedding送到attention网络中
hist_i = attention(i_emb, h_emb, self.sl)
# -- attention end ---

hist_i = tf.layers.batch_normalization(inputs=hist_i)
hist_i = tf.reshape(hist_i, [-1, hidden_units], name='hist_bn')
hist_i = tf.layers.dense(hist_i, hidden_units, name='hist_fcn')
u_emb_i = hist_i #u_emb_i就是attention的输出向量

print(u_emb_i.get_shape().as_list())
print(i_emb.get_shape().as_list())

# -- fcn begin -------
din_i = tf.concat([u_emb_i, i_emb], axis=-1)
din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
d_layer_1_i = tf.layers.dense(din_i, 80, activation=tf.nn.sigmoid, name='f1')
d_layer_1_i = dice(d_layer_1_i, name='dice_1')
d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.nn.sigmoid, name='f2')
d_layer_2_i = dice(d_layer_2_i, name='dice_2')
d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')

d_layer_3_i = tf.reshape(d_layer_3_i, [-1])

self.logits = i_b + d_layer_3_i
self.loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits,
            labels=self.y)
    )

def test(self, sess, uij):
    return sess.run(self.logits_sub, feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.j: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
    })
```

### eval代码

一般情况下，我们做验证集的评价，都是复用train的过程，只不过最后不进行反向bp，但是这里将正负样本一起送进去，分别得到postive_score和negative_score，同时记录两者及差值，方便后续计算gauc。

在训练时候，每隔固定的step最好进行eval，保存一个分数最好的模型。

```python
self.u = tf.placeholder(tf.int32, [None, ])  # [B],用户id
self.i = tf.placeholder(tf.int32, [None, ])  # [B]，正样本
self.j = tf.placeholder(tf.int32, [None, ])  # [B]，负样本
self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T]，用户浏览记录
self.sl = tf.placeholder(tf.int32, [None, ])  # [B]，真实记录数量
self.lr = tf.placeholder(tf.float64, [])  #学习率

hidden_units = 128

#初始化几个embedding矩阵。注意这里对cate也进行embedding化，item的最终embedding=id_emb+cate_emb
user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units]) #user的emb，模型中没用到
item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2]) #item的emb
item_b = tf.get_variable("item_b", [item_count], initializer=tf.constant_initializer(0.0)) #item的bias
cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2]) #cate的emb
cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

#获取正样本的embedding
ic = tf.gather(cate_list, self.i) #取出正样本对应的cate
i_emb = tf.concat(values=[
      tf.nn.embedding_lookup(item_emb_w, self.i),
      tf.nn.embedding_lookup(cate_emb_w, ic),
], axis=1)
i_b = tf.gather(item_b, self.i) #我的理解是充当item固有属性了，bias的意思，是一个随机数，可训练

#获取负样本的embedding
jc = tf.gather(cate_list, self.j)
j_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.j),
        tf.nn.embedding_lookup(cate_emb_w, jc),
    ], axis=1)
j_b = tf.gather(item_b, self.j) #同上面的逻辑

#获取历史行为的embedding
hc = tf.gather(cate_list, self.hist_i)
h_emb = tf.concat([
      tf.nn.embedding_lookup(item_emb_w, self.hist_i),
      tf.nn.embedding_lookup(cate_emb_w, hc),
], axis=2)

# 处理正样本
#将正样本embedding和历史行为embedding送到attention网络中
hist_i = attention(i_emb, h_emb, self.sl)

hist_i = tf.layers.batch_normalization(inputs=hist_i)
hist_i = tf.reshape(hist_i, [-1, hidden_units], name='hist_bn')
hist_i = tf.layers.dense(hist_i, hidden_units, name='hist_fcn')
u_emb_i = hist_i #u_emb_i就是attention的输出向量

din_i = tf.concat([u_emb_i, i_emb], axis=-1)
din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
d_layer_1_i = tf.layers.dense(din_i, 80, activation=tf.nn.sigmoid, name='f1')
d_layer_1_i = dice(d_layer_1_i, name='dice_1')
d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.nn.sigmoid, name='f2')
d_layer_2_i = dice(d_layer_2_i, name='dice_2')
d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')

d_layer_3_i = tf.reshape(d_layer_3_i, [-1])

# 处理负样本
hist_j = attention(j_emb, h_emb, self.sl)

hist_j = tf.layers.batch_normalization(inputs=hist_j)
hist_j = tf.reshape(hist_j, [-1, hidden_units], name='hist_bn')
hist_j = tf.layers.dense(hist_j, hidden_units, name='hist_fcn', reuse=True)
u_emb_j = hist_j

din_j = tf.concat([u_emb_j, j_emb], axis=-1)
din_j = tf.layers.batch_normalization(inputs=din_j, name='b1', reuse=True)
d_layer_1_j = tf.layers.dense(din_j, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
d_layer_1_j = dice(d_layer_1_j, name='dice_1')
d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
d_layer_2_j = dice(d_layer_2_j, name='dice_2')
d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name='f3', reuse=True)

d_layer_3_j = tf.reshape(d_layer_3_j, [-1])

# 验证结果
x = i_b - j_b + d_layer_3_i - d_layer_3_j  # [B]
self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))
self.score_i = tf.sigmoid(i_b + d_layer_3_i)
self.score_j = tf.sigmoid(j_b + d_layer_3_j)
self.score_i = tf.reshape(self.score_i, [-1, 1])
self.score_j = tf.reshape(self.score_j, [-1, 1])
self.p_and_n = tf.concat([self.score_i, self.score_j], axis=-1)

def eval(self, sess, uij):
    u_auc, socre_p_and_n = sess.run([self.mf_auc, self.p_and_n], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.j: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
    })
    return u_auc, socre_p_and_n
```

### test代码

test的过程是输入一些用户hist商品，一些待排序的item集合，然后得到每个item的打分。在model的部分，为了方便，只取了前predict_ads_num个广告计算分数。

在实际推荐过程中，送入模型的是一个待排序的广告id的list，然后对这个list中的所有广告计算分数，也就是排序的过程。

```python
self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T]，用户浏览记录
self.sl = tf.placeholder(tf.int32, [None, ])  # [B]，真实记录数量
self.lr = tf.placeholder(tf.float64, [])

hidden_units = 128

item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2])
item_b = tf.get_variable("item_b", [item_count],initializer=tf.constant_initializer(0.0))
cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2])
cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

hc = tf.gather(cate_list, self.hist_i)
h_emb = tf.concat([
    tf.nn.embedding_lookup(item_emb_w, self.hist_i),
    tf.nn.embedding_lookup(cate_emb_w, hc),
], axis=2)

# prediciton for selected items
# logits for selected item:
item_emb_all = tf.concat([
        item_emb_w,
        tf.nn.embedding_lookup(cate_emb_w, cate_list)
], axis=1)
item_emb_sub = item_emb_all[:predict_ads_num, :]
item_emb_sub = tf.expand_dims(item_emb_sub, 0)
item_emb_sub = tf.tile(item_emb_sub, [predict_batch_size, 1, 1])
hist_sub = attention_multi_items(item_emb_sub, h_emb, self.sl)

hist_sub = tf.layers.batch_normalization(inputs=hist_sub, name='hist_bn', reuse=tf.AUTO_REUSE)
hist_sub = tf.reshape(hist_sub, [-1, hidden_units])
hist_sub = tf.layers.dense(hist_sub, hidden_units, name='hist_fcn', reuse=tf.AUTO_REUSE)
u_emb_sub = hist_sub

item_emb_sub = tf.reshape(item_emb_sub, [-1, hidden_units])
din_sub = tf.concat([u_emb_sub, item_emb_sub], axis=-1)
din_sub = tf.layers.batch_normalization(inputs=din_sub, name='b1', reuse=True)
d_layer_1_sub = tf.layers.dense(din_sub, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
d_layer_1_sub = dice(d_layer_1_sub, name='dice_1_sub')
d_layer_2_sub = tf.layers.dense(d_layer_1_sub, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
d_layer_2_sub = dice(d_layer_2_sub, name='dice_2_sub')
d_layer_3_sub = tf.layers.dense(d_layer_2_sub, 1, activation=None, name='f3', reuse=True)
d_layer_3_sub = tf.reshape(d_layer_3_sub, [-1, predict_ads_num])

self.logits_sub = tf.sigmoid(item_b[:predict_ads_num] + d_layer_3_sub)
self.logits_sub = tf.reshape(self.logits_sub, [-1, predict_ads_num, 1])
```

























