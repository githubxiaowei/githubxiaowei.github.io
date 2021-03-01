---
layout: test
title: new file
date: 2021-02-24 14:55:32
tags:
---

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201021162745191.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2l0bmVyZA==,size_16,color_FFFFFF,t_70#pic_center)

1999 年 Barabási 和 Albert 提出了无标度网络模型（简称 BA 模型）。无标度网络的重要特征为： 无标度网络的节点度分布服从幂律分布。

无标度网络的度分布 $p(d)$ 满足$$p(d)\sim d^{-\alpha}，$$其中 $d$ 代表度的大小， $\alpha$ 为度分布的幂律指数。 真实网络 $\alpha$ 值一般介于 2~3之间。

近年来越来越多的研究表明， 真实世界网络既不是规则网络， 也不是随机网络， 而是兼具小世界和无标度特性的复杂网络， 具有与规则网络和随机网络截然不同的统计特性。

本文采用的无标度网络生成模型是由 Barabási 和 Albert 于 1999 年提出的增长网络网络模型（BA 模型）。在该模型中，网络初始时具有 $m_0$ 个节点，两两互连。 之后每过一个时间单位增加一个新节点。新节点从当前网络中选择$m(m ≤ m_0)$ 个节点与之连接， 某节点 $v_i$ 被选中的概率 $p(v_i)$ 与其节点度 $d_i$ 的大小成正比，即$$p(v_i) = \frac{d_i}{\sum_j d_j}$$经过 t 个时间单位后，网络中含有 $m_0+t$ 个节点，$m_0(m_0-1)/2+mt$条边。可以证明当 t 足够大时， 按此规律增长的网络的度分布为幂指数等于 3 的幂律分布。

依据新节点的连接规律，建立节点度演化的动力学模型：
![在这里插入图片描述](https://img-blog.csdn.net/20181008113859419?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2l0bmVyZA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
其中最后一个等式在 $t$ 足够大时近似成立。 将节点 $i$ 加入网络的时间记为 $t_i$，
则有初始条件 $d_i(t_i) = m$。解得
![在这里插入图片描述](https://img-blog.csdn.net/20181008114022515?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2l0bmVyZA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
在 $t$ 足够大， 对任意节点 $i$， 其度的大小满足
![在这里插入图片描述](https://img-blog.csdn.net/20181008114111695?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2l0bmVyZA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181008114126346?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2l0bmVyZA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
其中第三个等式成立的原因是加入节点的时间是等间隔的。上式正是网络节点度的概率分布函数， 可以求出节点度的概率密度函数 $p(d)$ 为
![在这里插入图片描述](https://img-blog.csdn.net/20181008114239376?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2l0bmVyZA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
可知所生成网络的幂律分布的指数为 3。下面的matlab程序模拟了BA网络的演化过程：

```matlab
function scale_free(N,m0,m)
%
%param  N: num of vertices 期望节点数
%param m0: num of initial vertices 初始边数
%param  m: num of vertices a new node try to connect 新节点连接的边数
%
tic;

I = 2 ;    %生成的网络个数，只为统计需要

realization_of_distribution = sparse( I , N ) ;
for J = 1 : I
    format long;

 	%初始化邻接矩阵，前m0个节点两两互连
    adjacent_matrix = sparse( m0 , m0 ) ;
    parfor i = 1 : m0
        for j = 1 : m0
            if j ~= i
                adjacent_matrix( i , j ) = 1 ;
            end
        end
    end
    adjacent_matrix = sparse( adjacent_matrix ) ;

	% 计算当前节点度分布
    node_degree = sparse( 1 , m0 ) ;
    for p = 1 : m0
        node_degree( p ) = sum( adjacent_matrix( 1 : m0 , p ) ) ;
    end

	% 开始演化
    for iteration = m0 + 1 : N
        total_degree = 2 * m * ( iteration - m0 -1 ) + m0*(m0-1) ; % m*2
        degree_frequency = node_degree / total_degree ;
        cum_distribution = cumsum( degree_frequency ) ;

        choose = zeros( 1 , m ) ;
        for new_edge = 1:m
            r = rand(1) ;
            choose_edge = find( cum_distribution >= r ,1) ;
            while any(choose == choose_edge)
                r = rand(1) ;
                choose_edge = find(  cum_distribution >= r,1) ;
            end
            choose(new_edge) = choose_edge;
        end

        for k = 1 : m
            adjacent_matrix( iteration , choose(k) ) = 1 ;
            adjacent_matrix( choose(k) , iteration ) = 1 ;
        end

        for p = 1 : iteration
            node_degree(p) = sum( adjacent_matrix( 1 : iteration , p ) ) ;
        end
    end

    number_of_nodes_with_equal_degree = zeros( 1 , N ) ;

    parfor i = 1 : N
        number_of_nodes_with_equal_degree(i) = length( find( node_degree == i ) ) ;
    end
    realization_of_distribution( J , : ) = number_of_nodes_with_equal_degree ;

    save(['adj_',num2str(J)],'adjacent_matrix');
end

%{
%plot degree distribution 在双对数坐标下画图

average = sum( realization_of_distribution )/ ( I * N );
loglog( 1:N , average , '*' )
axis([1 N 0.0000001 0.9])
hold on;
x = 1:N;
y = 2 * m^2 * x .^ ( -3 ) ;
loglog( x , y , 'r' ) ;  %  p(k)=2*m^2*k^(-3)

%}
toc;

end
```
---
人工生成网络的概率质量函数（网络节点数 $N$ 分别为 50、 100、 200、 400）
![在这里插入图片描述](https://img-blog.csdn.net/20181008113233633?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2l0bmVyZA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
图中直线为理论结果： $p(d)=2\frac{m^2}{d^3}$。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201028100746209.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2l0bmVyZA==,size_16,color_FFFFFF,t_70#pic_center)


