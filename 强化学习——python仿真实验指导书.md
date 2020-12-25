# 强化学习——python 仿真实验指导书

## 整体策略

1. 每个节点维护一个 Q 表格，记录所有邻居节点的可靠性（队列长度+距离）；
2. 每收到邻居节点的数据包，进行 Q 表格更新：
3. 更新方式：如果该邻居节点为目的节点，Q 值设定为 100；否则，Q 值设定为：β× 当前值 + (1-β) × 该邻居节点 Q 表格中的最大值 × 衰减因子 γ × 邻居节点优先级因子 α。
4. 每次发送数据包选择 Q 值最大的邻居节点。

## 代码分析

### 准备工作

模块和包：

```python
import numpy as np
import matplotlib.pyplot as plt
import re
import random
import time
```

系统参数：

```python
communication_range = 300
area_range = 2000
num_slots = 10
slots = 1000
len_slot = 0.01  #slot参数在后续中完成采样功能及定位采样位置的功能
maxQ = 10
lamb = 0.005
fixFrequence = 10
a1 = 0.5  # RL的参数
a2 = 0.8  # RL的参数
p1 = 10
AP_pos = [1220, 1040]
total_num = 0
isVisual = False  # True#由于版本升级，原有的可视化代码中的部分模块已无法使用，保持false即可，实验指导书中已删除可视化相关代码
```

### 开始执行

确定采集时间段内一直在采集区域中的车辆，作为消息源：

```python
def positionsOfFile(inputFile):  # 节点位置读入节点数目及
    '''
从mobility文件读入所有节点各时刻的位置信息，并根据给定源数目从仿真时间内始终在仿真区域的节点中随机选择出源节点集
input：    inputfile：输入的文件路径
output：   position(np(length,nodenum,2))：某时刻某点的坐标
        nodenum(int)：节点数目（读取文件的总数目）
    '''
    num_perSlot = int(slots / num_slots)  # 时隙数/s
    x_bias = 200  # 用于校正两个场景之间的坐标差异，校正为0~2000
    y_bias = 0
    nodeNum = 0
    timeBegin = 400
    timeEnd = 0
    f = open(inputFile, "r")
    line = f.readline()
    while line:
        if line.startswith("$ns_"):
            nums = re.split('[ ()]', line)
            timeBegin = min(timeBegin, float(nums[2]))
            timeEnd = max(timeEnd, float(nums[2]))
            nodeNum = max(nodeNum, int(nums[4]))  # Ch
        line = f.readline()
    f.close()
    nodeNum = nodeNum + 1
    print("The total number of nodes is:")  # 把标号处理为总结点数目
    print(nodeNum)  # 指出给定的节点数目
    timeBegin = int(timeBegin)
    timeEnd = int(timeEnd)
    length = min(slots, (timeEnd - timeBegin + 1) * num_perSlot)  # 总长度#slots
    positions = np.ones([length, nodeNum, 2]) * 4000  # 初始化位置远离仿真区域 每个时隙|每个点|对应一个位置x,y#查看一下position的值
    speed = np.ones(nodeNum) * (-1)
    f = open(inputFile, "r")
    line = f.readline()
    while line:
```

line 中的存储格式：

| $ns\_ | at  | 59.0 |     | "$node\_( | 68  | )   | setdest | 1304.55 | 1068.33 | 27.23 | "   |
| ----- | --- | ---- | --- | --------- | --- | --- | ------- | ------- | ------- | ----- | --- |
| 0     | 1   | 2    | 3   | 4         | 5   | 6   | 7       | 8       | 9       | 10    | 11  |

```python
        if line.startswith("$ns_"):
            nums = re.split(r'[ ()"]', line)
            now = int(float(nums[2]))
            id = int(nums[5])
            x = float(nums[8]) - x_bias
            y = float(nums[9]) - y_bias
            sp = float(nums[10])  # 节点移动速度
            if now <= timeBegin + num_slots:  # 只用了开始-》numslot的一段时间内的数据
                if speed[id] == -1 and now < timeBegin + num_slots or (now > timeBegin and now < timeBegin + num_slots and
                                                                   positions[(now - timeBegin - 1) * num_perSlot][id][
                                                                       0] == 4000):  # 无速度 & 十秒内 |（访问过且在十秒且内在范围外）
                    speed[id] = sp
                    positions[(now - timeBegin) * num_perSlot][id][0] = x  # 多少秒*时隙/s=对应到的时隙
                    positions[(now - timeBegin) * num_perSlot][id][1] = y
                else:
                    if positions[(now - timeBegin - 1) * num_perSlot][id][0] != 4000:  # 位置信息更新，时间段超出或有速度
                        for i in range(num_perSlot - 1):  # 因为num_是10，所以-1*10+8最后一个不赋值
                            positions[(now - timeBegin - 1) * num_perSlot + i + 1][id][0] = positions[(now - timeBegin - 1) * num_perSlot][id][0] + (x - positions[(now - timeBegin - 1) * num_perSlot][id][0]) / num_perSlot * (i + 1)  # 平均一秒内位置的变化至每个时隙x
                        positions[(now - timeBegin - 1) * num_perSlot + i + 1][id][1] =positions[(now - timeBegin - 1) * num_perSlot][id][1] + (y - positions[(now - timeBegin - 1) * num_perSlot][id][1]) / num_perSlot * (i + 1)  # 平均一秒内位置的变化至每个时隙y

                    if now < timeBegin + num_slots:  # 给最后一个时隙赋值
                        positions[(now - timeBegin) * num_perSlot][id][0] = x
                        positions[(now - timeBegin) * num_perSlot][id][1] = y
                    speed[id] = sp

            else:
                break
        line = f.readline()
    f.close()
    set_begin = []
    set_end = []
    set_same = []
    for m in range(nodeNum):
        if positions[0][m][0] != 4000:  # 横坐标初始化过。
            set_begin.append(m)
        if positions[length - 1][m][0] != 4000:  # 最后一个时隙。
            set_end.append(m)
    for a in set_begin:
        if a in set_end:
            set_same.append(a)  # 取交集
    print('always in the area:')
    print(len(set_same))  # 统计了整个过程中都在界面内的数目。
    sources = random.sample(set_same, sense_num)
    sources.sort()
    print('sources num')
    print(sources)  # 在所有一直在的节点中抽sense_num个。
    i = 0
    for a in sources:  # 把开始随机生成的表搞成顺序的。
        temp = positions[:, i, :]
        positions[:, i, :] = positions[:, a, :]
        positions[:, a, :] = temp
        i += 1
    return positions, nodeNum  # 所有来过的节点
```

开始学习：

确定好发送节点后，使用 begin_RL 开始使用强化学习进行路由：函数初始化 neighbor，Qtablel，mQ 决定是否采集新的数据包至队列(update)；作出传输决策并完成传输，更新队列变化并统计时延(transform)；更新 Qtable 和 mQ(move)

```python
def begin_RL(positions):
    '''
初始化neighbor，Qtablel，mQ决定是否采集新的数据包至队列(update)；作出传输决策并完成传输，更新队列变化并统计时延(transform)；更新Qtable和mQ(move)
global: total_num（int）：节点个数
input:  position(np(length,nodenum,2))：某时刻某点的坐标
output: avgdelay（float）：平均时延
    '''
    queue = np.zeros([total_num, 0]).tolist()  # [i][生成时间]
    qlen = np.zeros(total_num)  # 记录每个节点的长度，total_num,一开始队列长度都是0
    neighbors, Qtable, mQ = initial_RL(positions)  # 初始化Qtable,mQ值
    sumdelay = 0  # 到目的节点的数据包的总时延
    count = 0  # 记录传输到目的节点的数据包数目
    for i in range(0, slots):
        # i是第n个时隙，帮助计算时延

        qlen, queue = update(qlen, queue, i)  # 是否采集新的数据包加入到队列#每个节点：队多长，队列中的数据包（数据包的源头和生成时间）
        qlen, queue, sumdelay, count = transform_RL(qlen, queue, i, neighbors, positions, Qtable, sumdelay,
                                                    count)  # 做出传输决策并完成传输过程，更新队列变化，统计时延
        neighbors, Qtable, mQ = move_RL(positions, i, Qtable, neighbors, mQ, qlen)  # 更新Qtable,mQ值
    print(sumdelay)
    print(count)
    if (count == 0):
        count = 1
    avgdelay = sumdelay / count
    print(avgdelay)
    return avgdelay
```

初始化路由过程：

首先对 Q 学习用到的参数值进行初始化，计算节点与接入点之间的距离。确定每个节点的邻居节点的数目和标号

```python
def initial_RL ( positions ):

    '''
    初始化Qtable,mQ值：根据位置信息设置mQ的值，并根据距离决定是否在通信距离中更新邻居列表。
    global:total_num
    input:position(np[length,nodenum,2])：某时刻某点的坐标
    output:neighbors([int])：邻居列表
           Qtable([[[j,0]]]):第i个位置存储了可以和i通信的节点及该节点的联通度
           mQ[int]：节点i到目标节点的联通度（远近）
           :type positions: object
    '''
    mQ = -1 * np.ones(total_num)  # 初始化表格信息
    neighbors = []
    Qtable = []
    for i in range(0, total_num):
        res = []  # 邻居的标号
        res1 = []  # 邻居的标号，设置连接度为0
        for j in range(0, total_num):
            if j != i:
                distance_i_j = dis(positions[0][i], positions[0][j])
                if(distance_i_j<=communication_range):
                    res.append(j)
                    res1.append([j, 0])
                    mQ[i] = 0  # 若i，j可传输设为0
        neighbors.append(res)  # 在邻居列表中添加邻居
        Qtable.append(res1)  # 添加邻居和连接度
        if dis(positions[0][i], AP_pos) <= communication_range:
            mQ[i] = 100  # 若i和AP（目的节点）可传输设为100
    return neighbors, Qtable, mQ  # mQ越高越好，是一个奖励的值
```

执行完初始化过程，我们已经获得了每个节点的邻居，现在，我们继续执行 begin 中的语句：在第 i 个时隙中，我们需要查看可以发包的车辆，查看它周边是否有还能帮助转发的其他车辆，确定它的发包策略，并且根据车辆的移动更新邻居表：

```python
 for i in range(0, slots):
        # i是第n个时隙，帮助计算时延
    qlen, queue = update(qlen, queue, i)  # 是否采集新的数据包加入到队列#每个节点：队多长，队列中的数据包（数据包的源头和生成时间）
    qlen, queue, sumdelay, count = transform_RL(qlen, queue, i, neighbors, positions, Qtable, sumdelay,
                                                count)  # 做出传输决策并完成传输过程，更新队列变化，统计时延
    neighbors, Qtable, mQ = move_RL(positions, i, Qtable, neighbors, mQ, qlen)  # 更新Qtable,mQ值
```

在这里我们解释一下“**更新方式：如果该邻居节点为目的节点，Q 值设定为 100；否则，Q 值设定为：β× 当前值 + (1-β) × 该邻居节点 Q 表格中的最大值 × 衰减因子 γ × 邻居节点优先级因子 α。**”相关的权值更新操作。

```python
def move_RL(positions, t, Qtable, PreNeighbor, mQ, qlen):  # 更新Qtable,mQ值
    '''
根据位置信息更新Qtable和mQ，去掉Qtable中已经离开通信节点范围的点并加入新进入的点并添加新进入的点的mQ值。
local：gamma联通度随时间的的衰减
input：positions, t, Qtable, PreNeighbor, mQ, qlen
output：neighbors, Qtable, mQ
    '''
    gamma=100
    # 如果超出通信范围
    for i in range(total_num):
        for j in Qtable[i]:
            if dis(positions[t][i], positions[t][j[0]]) > communication_range:
                Qtable[i].remove(j)
    # 设一个空的邻居
    neighbors = []
    for i in range(0, total_num):
        res = []  # 临时邻居节点
        for j in range(0, total_num):
            if j != i:
                if (dis(positions[t][i], positions[t][j]) <= communication_range):
                    res.append(j)
                    if j not in PreNeighbor[i]:  # 如果j不在之前的邻居表里，就更新大家的节点、联通值
                        Qtable[i].append([j, mQ[j]])
        neighbors.append(res)

    for i in range(0, total_num):
        mQ[i] = -1
        for p in Qtable[i]:
            p[1] = (1 - a1) * p[1] + a1 * (a2 * (1 - dis(positions[t][i], positions[t][p[0]]) / 3000) * mQ[p[0]])
            mQ[i] = max(mQ[i], p[1])
        if dis(positions[t][i], AP_pos) <= communication_range:
            mQ[i] = 100 - dis(positions[t][i], AP_pos) /gamma  # 给连通度一个随操作次数发生的衰减

    return neighbors, Qtable, mQ
```

一番操作后，我们得到了在 3/5/7/9 个信源的情况下十次仿真的时延的平均值。

## 参数修改

经过上述的代码描述过程，我们会发现，move_RL 函数使用了强化学习的多个参数，我们可以尝试修改**衰减的强度**和 **a1**,**a2**的值查看这些参数是如何影响**不同数量的信源**在通信效果的。

由于代码一轮需要至少执行 4✖️10✖️nodenum✖️slot 次，运行时间在 20 分钟以上，所以给出衰减强度不同时的图片结果。

如果需要缩短运行时间，可以尝试修改 main 中的参数‘’10“。

周日早八点半：

- 随着源节点数目的增加，整体的规律
- 单个节点数目在不同遗忘的情况下的时延

ppt：tcl 处理/代码&流程：结果图（会给）：周六中午 13：00

环境：python：网址/版本：周四中午 20：00
