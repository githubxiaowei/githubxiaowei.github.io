---
title: RNN 训练算法 —— BPDC (Backpropagation-Decorrelation)
date: 2021-03-23 19:30:22
tags: [RNN, BPDC, 训练, 神经网络]
categories: 神经网络
mathjax: true
---

# 问题描述
考虑模型循环网络模型：
$$
x(k+1) = (1-\Delta t)x(k) +  \Delta t Wf[x(k)] \tag1{}
$$
其中 $x(k) \in R^N$表示网络节点在激活前的状态，$W\in R^{N\times N}$表示网络结点之间相互连接的权重，网络的输出节点为 $\{x_i(k)| i\in O\}$，$O$为所有输出（或称“观测”）单元的下标集合

训练的目标是为了减少观测状态和预期值之间误差，即最小化损失函数：
$$
E = \frac{1}{2}\sum_{k=1}^K \sum_{i\in O} [x_i(k) - d_i(k)]^2 \tag{2}
$$
其中 $d_i(k)$ 表示 $k$ 时刻第 $i$ 个节点的预期值

<!--more-->

![RNN](RNN.png)

# 符号约定
$$
W \equiv
\begin{bmatrix}
\text{-----}  w_1^T \text{-----} \\
\vdots \\
\text{-----}  w_N^T \text{-----} 
\end{bmatrix}_{N\times N}
$$
将矩阵 $W$ 拉成列向量，记为 $w$
$$
w = [w_1^T, \cdots, w_N^T]^T \in R^{N^2}
$$
把所有时间的状态拼成列向量，记为 $x$
$$
x = [x^T(1), \cdots, x^T(K)]^T \in R^{NK}
$$
将RNN 的训练视为约束优化问题，(1)式转化成约束条件:
$$
g(k+1) \equiv  -x(k+1) + (1-\Delta t)x(k) +  \Delta t Wf[x(k)] , \quad k=1,\ldots ,K \tag{3}
$$
记
$$
g = [g^T(1), \ldots, g^T(K)]^T \in R^{NK}
$$


# Atiya-Parlos 算法回顾
以上是经典的梯度下降法的思维，但是 Atiya-Parlos 提出了另一种优化思路：不是朝着参数的梯度方向更新，但仍使代价函数下降

该算法的思想是互换网络状态 $x(k)$ 和权重矩阵 $W$ 的作用：将状态视为控制变量，并根据 $x(k)$ 的变化确定权重的变化。 换句话说，我们计算 $E$ 相对于状态 $x(k)$ 的梯度，并假设状态在该梯度的负方向 $\displaystyle{\Delta x_i(k) = -\eta\frac{\partial E}{\partial x_i(k)} }$ 上有微小变化。

接下来，我们确定权重 $W$ 的变化 $\Delta w$，以使由权重变化导致的状态变化尽可能地接近目标变化 $\Delta x$

该算法的细节如下：
$$
\begin{aligned}
\Delta x &= -\eta \left(\frac{\partial E}{\partial x_i} \right)^T \\
&= -\eta e^T\\
&= -\eta [e(1), \ldots, e(K)]^T \\\\
e_i(k)&= \begin{cases}
   x_i(k) - d_i(k), &\text{if } i\in O, \\
   0, &\text{otherwise. } 
\end{cases} 
k \in 1,\ldots,K.
\end{aligned}
$$


由约束条件得：
$$
\frac{\partial g}{\partial x} \Delta x = - \frac{\partial g}{\partial w} \Delta w
$$
故已知 $\Delta x$ 时，可得：
$$
\Delta w = -\left[\left(\frac{\partial g}{\partial w}\right)^T \left(\frac{\partial g}{\partial w}\right)\right]^{-1} \left(\frac{\partial g}{\partial w}\right)^T\left( \frac{\partial g}{\partial x}\right) \Delta x
$$
需要注意逆矩阵不一定存在，故
$$
\Delta w = -\left[\left(\frac{\partial g}{\partial w}\right)^T \left(\frac{\partial g}{\partial w}\right) + \epsilon I \right]^{-1} \left(\frac{\partial g}{\partial w}\right)^T\left( \frac{\partial g}{\partial x}\right) \Delta x
$$
这就是权重 $W$ 的更新规则

# 计算细节

计算 $\frac{\partial g}{\partial w}$
$$
\frac{\partial g}{\partial w} =
 \begin{bmatrix}
\frac{\partial g(1)}{\partial w}\\
\vdots \\
\frac{\partial g(K)}{\partial w}
\end{bmatrix}
=  \Delta t \begin{bmatrix}
\frac{\partial  Wf[x(0)] }{\partial w}\\
\vdots \\
\frac{\partial Wf[x(K-1)] }{\partial w}
\end{bmatrix}
$$

其中
$$
\begin{aligned}
\frac{\partial Wf[x(k)]}{\partial w}
&= \begin{bmatrix}
\frac{\partial w_1^Tf[x(k)]}{\partial w}\\
\vdots \\
\frac{\partial w_N^Tf[x(k)]}{\partial w}
\end{bmatrix} \color{red}{记 f_k = [f(x_1), \ldots, f(x_N(k))]^T}\\\\
& = \begin{bmatrix}
f_k^T &&& \\
 & f_k^T&& \\
 && \ddots & \\
 &&& f_k^T
\end{bmatrix}_{N\times N^2} \\\\
&\triangleq  F(k)
\end{aligned}
$$

故
$$
\frac{\partial g}{\partial w} = 
\Delta t \begin{bmatrix}
F(0)\\
\vdots \\
F(K-1)
\end{bmatrix}_{NK \times N^2}
$$
$$
\begin{aligned}
&\frac{1}{\Delta t^2}\left(\frac{\partial g}{\partial w}\right)^T \left(\frac{\partial g}{\partial w}\right)  \\
&= 
 \begin{bmatrix}
F^T(0) &
\cdots &
F^T(K-1)
\end{bmatrix}
 \begin{bmatrix}
F(0)\\
\vdots \\
F(K-1)
\end{bmatrix} \\\\
&= 
\sum_{k=0}^{K-1} F^T(k)F(k) \\\\
&=\begin{bmatrix}
 \sum_{k=0}^{K-1} f_k f_k^T &&& \\
 &  \sum_{k=0}^{K-1} f_k f_k^T  && \\
 && \ddots & \\
 &&&  \sum_{k=0}^{K-1} f_k f_k^T 
\end{bmatrix}_{N^2 \times N^2} \\\\
&\triangleq diag\{C_{K-1}\}
\end{aligned}
$$


令
$$
\gamma = 
\begin{bmatrix}
\gamma(1)\\ 
\gamma(2) \\ 
\vdots \\
 \gamma(K)
\end{bmatrix}_{NK}
= \frac{\partial g}{\partial x} \Delta x 
$$
$\gamma$ 表示由 $\Delta x$ 提供的误差信息，它的计算放在本文最后，先假设它已经求出来了

则
$$
\begin{aligned}
& \left(\frac{\partial g}{\partial w}\right)^T\left( \frac{\partial g}{\partial x}\right) \Delta x  \\
&= 
\Delta t \begin{bmatrix}
F^T(0) &
\cdots &
F^T(K-1)
\end{bmatrix}_{N^2 \times NK}
\begin{bmatrix}
\gamma(1)\\ 
\gamma(2) \\ 
\vdots \\
 \gamma(K)
\end{bmatrix}_{NK} \\
&= \Delta t\sum_{k=1}^K F^T(k-1)\gamma(k) \\
&=\Delta t \sum_{k=1}^K   \begin{bmatrix}
f_{k-1} &&& \\
 & f_{k-1}&& \\
 && \ddots & \\
 &&& f_{k-1}
\end{bmatrix}_{N^2 \times N}
\begin{bmatrix}
\gamma_1(k)\\ 
\gamma_2(k) \\ 
\vdots \\
 \gamma_N(k)
\end{bmatrix}_{N}\\\\
&=\Delta t \begin{bmatrix}
 \sum_{k=1}^K f_{k-1} \gamma_1(k)\\ 
 \sum_{k=1}^K f_{k-1} \gamma_2(k) \\ 
\vdots \\
 \sum_{k=1}^K f_{k-1} \gamma_N(k)
\end{bmatrix}_{N^2}\\\\
\end{aligned} 
$$

所以
$$
\begin{aligned}
\Delta w 
&=  -\left[\left(\frac{\partial g}{\partial w}\right)^T \left(\frac{\partial g}{\partial w}\right) + \epsilon I\right]^{-1} \left(\frac{\partial g}{\partial w}\right)^T\left( \frac{\partial g}{\partial x}\right) \Delta x \\
&= - \frac{1}{\Delta t} \begin{bmatrix}
C_{K-1}^{-1} \sum_{k=1}^K f_{k-1} \gamma_1(k)\\ 
C_{K-1}^{-1} \sum_{k=1}^K f_{k-1} \gamma_2(k) \\ 
\vdots \\
C_{K-1}^{-1} \sum_{k=1}^K f_{k-1} \gamma_N(k)
\end{bmatrix}_{N^2}\\\\
\Delta W 
&= - \frac{1}{\Delta t} \begin{bmatrix}
 \sum_{k=1}^K f_{k-1}^TC_{K-1}^{-1} \gamma_1(k)\\ 
 \sum_{k=1}^K f_{k-1}^TC_{K-1}^{-1} \gamma_2(k) \\ 
\vdots \\
 \sum_{k=1}^K f_{k-1}^TC_{K-1}^{-1} \gamma_N(k)
\end{bmatrix}_{N\times N} \\
&= - \frac{1}{\Delta t}  \sum_{k=1}^K\begin{bmatrix}
 f_{k-1}^T \gamma_1(k)\\ 
 f_{k-1}^T \gamma_2(k) \\ 
\vdots \\
f_{k-1}^T \gamma_N(k)
\end{bmatrix}_{N\times N} C_{K-1}^{-1} \\ 
\end{aligned} 
$$

其中
$$
C_{K-1} = \epsilon I + \sum_{r=0}^{K-1} f_r f_r^T
$$
注意：上述 $\Delta W$ 是基于 $1,2,\ldots, K$ 整个时间段的更新，不妨称之为 $\Delta W_{batch}$

下面将更新公式拆解在线更新（online updating）的形式：
$$
\Delta W^{batch}(K)= \Delta W(1) + \cdots + \Delta W(K)
$$

等式右端对应每一时刻的更新量

在第 $K$ 时刻的第 $i$ 个神经元的输入权重的更新量：
$$
\begin{aligned}
\Delta w^T_{i}(K) &=  - \frac{1}{\Delta t} \sum_{k=1}^{K} f_{k-1}^TC_{K-1}^{-1}  \gamma_i(k) + \frac{1}{\Delta t} \sum_{k=1}^{K-1} f_{k-1}^TC_{K-2}^{-1}  \gamma_i(k)\\\\
&= - \frac{1}{\Delta t}   f_{K-1}^TC_{K-1}^{-1} \gamma_i(K)  - \frac{1}{\Delta t} \sum_{k=1}^{K-1} f_{k-1}^T (C_{K-1}^{-1} - C_{K-2}^{-1}) \gamma_i(k) \\\\
&=- \frac{1}{\Delta t}   f_{K-1}^T C_{K-1}^{-1}\gamma_i(K)  - \frac{1}{\Delta t} \sum_{k=1}^{K-1} f_{k-1}^T C_{K-2}^{-1}\gamma_i(k)(C_{K-2}C_{K-1}^{-1} - I) \\\\
&= - \frac{1}{\Delta t}   f_{K-1}^TC_{K-1}^{-1} \gamma_i(K)  -   \Delta w_i^{batch}(K-1)(C_{K-2}C_{K-1}^{-1}- I) \\\\
&= - \frac{1}{\Delta t}   f_{K-1}^TC_{K-1}^{-1} \gamma_i(K)  -  \sum_{k=1}^{K-1} \Delta w^T_i(k)  (C_{K-2}C_{K-1}^{-1}- I)
\end{aligned}
$$
可以看出，APRL 的更新规则由当前时刻的误差和 w 的累计更新（动量）组成

随着 $K \to \infty$，易知$\sum_{k=1}^{K-1} \Delta w^T_i(k) \to const, C_{K-2}C_{K-1}^{-1} \to I$，所以第二项趋于零

# BPDC 更新规则
BPDC 对 APRL 的在线算法做了简单粗暴的近似

该近似不试图累积完整的相关矩阵 $C_k$，也舍弃了先前误差的累积，而且只计算瞬时相关 $C(k)$：
$$
\begin{aligned}
\Delta w^T_{i}(k+1) &= - \frac{1}{\Delta t}  f_{k}^TC(k)^{-1}  \gamma_i(k+1)  \\\\
C(k) &= \epsilon I + f_k f_k^T
\end{aligned}
$$
利用[矩阵求逆引理](https://blog.csdn.net/itnerd/article/details/105612704)：
$$
\begin{aligned}
C(k)^{-1} &= (\epsilon I + f_k f_k^T)^{-1} \\
&= \frac{1}{\epsilon}I - \frac{1}{\epsilon} \frac{ff^T}{\epsilon + f^Tf}
\end{aligned}
$$
所以
$$
\begin{aligned}
\Delta w^T_{i}(k+1) &= - \frac{1}{\Delta t}  f_{k}^T\left( \frac{1}{\epsilon}I - \frac{1}{\epsilon} \frac{ff^T}{\epsilon + f^Tf}\right)  \gamma_i(k+1)  \\\\
&=  - \frac{1}{\Delta t}   \frac{f^T}{\epsilon + f^Tf}  \gamma_i(k+1)  
\end{aligned}
$$

# 计算 $\gamma$
$$
\gamma = 
\begin{bmatrix}
\gamma(1)\\ 
\gamma(2) \\ 
\vdots \\
 \gamma(K)
\end{bmatrix}_{NK}
= \frac{\partial g}{\partial x} \Delta x = -\eta \frac{\partial g}{\partial x} [e(1), \ldots, e(K)]^T
$$
关键在与计算 $\frac{\partial g}{\partial x}$
$$
\begin{aligned}
\frac{\partial g}{\partial x} &= 
 \begin{bmatrix}
\frac{\partial g_1}{\partial x(1)} & \ldots & \frac{ \partial g_1}{\partial x(K)}\\
\vdots & \ddots & \vdots\\
\frac{\partial g_K}{\partial x(1)} & \ldots & \frac{ \partial g_K}{\partial x(K)}
\end{bmatrix}\\\\
&=  
\begin{bmatrix}
\frac{\partial g_1}{\partial x(1)} & 0&  0 &\ldots & 0\\
\frac{\partial g_2}{\partial x(1)} & \frac{\partial g_2}{\partial x(2)}& 0 &\ldots & 0  \\
0 & \frac{\partial g_3}{\partial x(2)} & \frac{\partial g_3}{\partial x(3)} & \ldots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \frac{\partial g_K}{\partial x(K-1)}& \frac{\partial g_K}{\partial x(K)}
\end{bmatrix} \\\\
&=  
\begin{bmatrix}
-I & 0&  0 &\ldots & 0\\
(1-\Delta t )I + \Delta t W D(1) & -I& 0 &\ldots & 0  \\
0 & (1-\Delta t )I + \Delta t W D(2) & -I & \ldots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 &(1-\Delta t )I + \Delta t W D(K-1)& -I
\end{bmatrix}
\end{aligned}
$$

其中
$$
D(k) =  \begin{bmatrix}
 f'(x_1(k)) & \cdots&0\\
\vdots & \ddots & \vdots\\
0& \ldots &  f'(x_N(k))
\end{bmatrix}_{N \times N}
$$
所以
$$
\begin{aligned}
\gamma &= -\eta \frac{\partial g}{\partial x} [e(1), \ldots, e(K)]^T\\
&= -\eta\begin{bmatrix}
-I & 0&  0 &\ldots & 0\\
(1-\Delta t )I + \Delta t WD(1) & -I& 0 &\ldots & 0  \\
0 & (1-\Delta t )I + \Delta t W D(2) & -I & \ldots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 &(1-\Delta t )I + \Delta t W D(K-1)& -I
\end{bmatrix}
\begin{bmatrix}
e(1) \\
e(2)  \\
e(3) \\
\vdots  \\
e(K)
\end{bmatrix} \\\\
&= 
-\eta \begin{bmatrix}
-e(1) \\
[(1-\Delta t )I + \Delta t W D(1)]e(1) - e(2)  \\
[(1-\Delta t )I + \Delta t W D(2)]e(2) - e(3)  \\
\vdots  \\
[(1-\Delta t )I + \Delta t W D(K-1)]e(K-1) - e(K) 
\end{bmatrix}_{NK \times 1}
\end{aligned}
$$
代入到 BPDC 更新规则：
$$
\begin{aligned}
\Delta w^T_{i}(k+1) 
&=  - \frac{1}{\Delta t}   \frac{f^T}{\epsilon + f^Tf}  \color{red}{ \gamma_i(k+1)}  \\
&= \frac{\color{red}{\eta}}{\Delta t}   \frac{f^T}{\epsilon + f^Tf} \color{red}\{ (1-\Delta t )e_i(k) + \Delta t \sum_{s\in O}w_{is} f'(x_s(k))e_s(k) - e_i(k+1) \} 
\end{aligned}
$$
# 参考文献
- J.J. Steil, Backpropagation-decorrelation: online recurrent learning with O(N) complexity, in: Proceedings of the International Joint Conference on Neural Networks (IJCNN), vol. 1, 2004, pp. 843–848.
- J.J. Steil, Online stability of backpropagation-decorrelation recurrent learning, Neurocomputing 69 (2006) 642–650.
- J.J. Steil, Online reservoir adaptation by intrinsic plasticity for backpropagation-decorrelation and echo state learning, Neural Networks 20 (3) (2007) 353–364.

