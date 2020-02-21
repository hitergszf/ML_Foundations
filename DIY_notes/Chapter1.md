# Chapter 1 The Learning Problem

## 1.1 PLA 算法
1. 比较简单的一个线性模型，针对二元分类 {-1, 1}

2. 基本思想:

   设置一个阈值和权重，当输入权重和向量的点积超过阈值的时候，输出+1,否则输出-1

   

$$
h(x) = sign((\sum_{i=1}^{d}{w_ix_i})+b)
$$

​			Vectorized version: add $$w_0=b$$ 
$$
h(x) = sign((W^TX))
$$

3. 条件：线性可分

4. 更新方法：
   $$
   W_{t+1} = W_t +y_tX_t
   $$

5. 算法流程：
   * 找到下一个错误分类的点$$(x_n,y_n)$$ 
   * 应用公式$$W_{t+1} = W_t +y_tX_t$$
   * 直到没有错误为止 （有错误怎么办）

6. 改进算法：**Pocket**——针对不一定linear separable的情况
   - 找到下一个错误分类的点$$(x_n,y_n)$$
   - 应用公式$$W'_{t+1} = W_t +y_tX_t$$
   - 如果$$W_{t+1}$$比$$W_t$$效果好，就用$$W'_{t+1}$$代替$$W_{t+1}$$ 
   - 直到有限次结束

## 1.2 学习类型

1. 监督学习、非监督学习、强化学习

2. 其他视角：
   - online learning
   - active learning

## 1.3学习是否可行？

1. $$E_{in}$$ : train error - train set
   $$
   E_{in}(h) = \frac1N\sum^N_{n=1}1[h(x_n)\neq f(x_n)]
   $$
   
2. $$E_{out}$$: test error  - infinite test set
   $$
   E_{out}(h) = P[h(x)\neq f(x_n)]
   $$
   

3. *Hoeffding Inequality*:
   $$
   P[|v-\mu|>\epsilon]\leq2e^{-2\epsilon^2N} , for\ any\ \epsilon>0
   $$
   Application: *对于假设h*
   $$
   P[|E_{in}-E_{out}|>\epsilon]\leq2e^{-2\epsilon^2N} , for\ any\ \epsilon>0
   $$
