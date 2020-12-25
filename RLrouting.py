import random
import re
import time

import matplotlib.pyplot as plt
import numpy as np

communication_range = 300
area_range = 2000
num_slots = 10  # 10秒
slots = 1000  # 1000个时隙
len_slot = 0.01  #
maxQ = 10
lamb = 0.005
fixFrequence = 10
a1 = 0.5  # RL
a2 = 0.8  # RL
p1 = 10
AP_pos = [1220, 1040]
total_num = 0


def dis(x, y):
    """
    根据节点坐标计算任意两点之间的实际距离
    """
    return ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5


def update(qlen, queue, t):
    """
    确定是否采集新的数据包并相应的更新队列信息.

    @param qlen: 队列长度
    @param queue: 队列内容
    @param t: 当前的时隙
    @return: 更新队列长度, 更新队列内容
    """
    # 之前进行了交换操作，所以现在源节点就在前sensenum个
    for i in range(0, sense_num):
        '''
        如果队列长度小于maxQ，那么向队列中增加新节点, 同时队列长度+1
        '''
        if qlen[i] < maxQ:  # 某个源节点的t时刻的队列长度如果小于该节点的队列容量
            queue[i].append([i, t])
            qlen[i] = qlen[i] + 1
    return qlen, queue


def route_rl(send_nodes, _, positions, qlen, queue, t, qtable, sumdelay, count):
    for i in send_nodes:
        max_r = -1  #
        next_node = -1  # 找节点
        if dis(positions[t][i], AP_pos) <= communication_range:  # 判断是否在AP附近，某时刻某点是否在ap附近
            # 统计时延
            sumdelay += t - queue[i][0][1]  # 发送节点的第i，发送时间，计算时延
            count += 1  # 目标节点收到的数据包数目
            queue[i].pop(0)  # 认为传出了
            qlen[i] = qlen[i] - 1  # 发出去了所以排队的包少一个
        else:
            for p in qtable[i]:  # 如果不能直接传给AP，挑一个节点作为下一跳
                if p[1] / p1 + (maxQ - qlen[p[0]]) > max_r and qlen[p[0]] < maxQ:  # 找队短的传
                    next_node = p[0]  # 吓一跳的节点
                    max_r = p[1] / p1 + (maxQ - qlen[p[0]])  # 更新权值
            if next_node != -1:  # 有吓一跳节点，别人的+1，自己的-1
                queue[next_node].append(queue[i][0])
                qlen[next_node] = qlen[next_node] + 1
                queue[i].pop(0)
                qlen[i] = qlen[i] - 1

    return qlen, queue, sumdelay, count  # 某个节点队有多长，队里有谁，sum=delay*count系统，目前为止传输到目的节点的数据包数量，


def transform_rl(qlen, queue, t, neighbors, positions, qtable, sumdelay, count):
    """
    确定发送节点集合并调用函数 route_rl 完成传输过程

    @param qlen:
    @param queue:
    @param t:
    @param neighbors:
    @param positions:
    @param qtable:
    @param sumdelay:
    @param count:
    @return: 队列长度, 队列内容, 当前路由的时延, count:
    """
    send_nodes = []  # 后续有机会数据包的节点
    list_need = []  # 想要发送数据包的节点
    for i in range(0, total_num):  # 每个节点
        if qlen[i] > 0:  # 如果节点队列中有包
            list_need.append(i)  # 想要发送数据包的节点
    while len(list_need) > 0:  # 如果有人要发
        pos = np.random.randint(0, len(list_need))  # 从要发的里面挑一个

        send_nodes.append(list_need[pos])  # totalnum中的某个之前说自己要发的节点加入了sendnode的集合里
        res = list_need[pos]  # 记录这个节点
        for j in neighbors[res]:  # 找这个节点的邻居
            if j in list_need:
                list_need.remove(j)  # 如果他的邻居也要发，就mute掉邻居的请求
        list_need.remove(res)  # 自身也离开
    qlen, queue, sumdelay, count = route_rl(send_nodes, neighbors, positions, qlen, queue, t, qtable, sumdelay,
                                            count)  # sendnode真的可以发数据包的节点;t是第n个时隙
    return qlen, queue, sumdelay, count


def move_rl(positions, t, qtable, pre_neighbor, m_q, _):
    """
    根据位置信息更新Qtable和mQ，去掉Qtable中已经离开通信节点范围的点并加入新进入的点并添加新进入的点的mQ值。

    @param positions:
    @param t:
    @param qtable:
    @param pre_neighbor:
    @param m_q:
    @param _:
    @return:
    """
    # 如果超出通信范围
    for i in range(total_num):
        for j in qtable[i]:
            if dis(positions[t][i], positions[t][j[0]]) > communication_range:
                qtable[i].remove(j)
    # 设一个空的邻居
    neighbors = []
    for i in range(0, total_num):
        res = []  # 临时邻居节点
        for j in range(0, total_num):
            if j != i:
                if dis(positions[t][i], positions[t][j]) <= communication_range:
                    res.append(j)
                    if j not in pre_neighbor[i]:  # 如果j不在之前的邻居表里，就更新大家的节点、联通值
                        qtable[i].append([j, m_q[j]])
        neighbors.append(res)

    for i in range(0, total_num):
        m_q[i] = -1
        for p in qtable[i]:
            p[1] = (1 - a1) * p[1] + a1 * (a2 * (1 - dis(positions[t][i], positions[t][p[0]]) / 3000) * m_q[p[0]])
            m_q[i] = max(m_q[i], p[1])
        if dis(positions[t][i], AP_pos) <= communication_range:
            m_q[i] = 100 - dis(positions[t][i], AP_pos) / 100  # 给连通度的值一个衰减

    return neighbors, qtable, m_q


def initial_rl(positions):
    """
    初始化Qtable,mQ值: 根据位置信息设置mQ的值，并根据距离决定是否在通信距离中更新邻居列表.

    @param positions: 某时刻某点的坐标
    @return: 邻居列表, 第i个位置存储了可以和i通信的节点及该节点的联通度, 节点i到目标节点的联通度（远近）
    """
    m_q = -1 * np.ones(total_num)  # 初始化表格信息
    neighbors = []
    qtable = []
    for i in range(0, total_num):
        res = []  # 邻居的标号
        res1 = []  # 邻居的标号，设置连接度为0
        for j in range(0, total_num):
            if j != i:
                distance_i_j = dis(positions[0][i], positions[0][j])
                if distance_i_j <= communication_range:
                    res.append(j)
                    res1.append([j, 0])
                    m_q[i] = 0  # 若i，j可传输设为0
        neighbors.append(res)  # 在邻居列表中添加邻居
        qtable.append(res1)  # 添加邻居和连接度
        if dis(positions[0][i], AP_pos) <= communication_range:
            m_q[i] = 100  # 若i和AP（目的节点）可传输设为100
    return neighbors, qtable, m_q  # mQ越高越好，是一个奖励的值


def begin_rl(positions):
    """
    初始化neighbor，Qtablel，mQ决定是否采集新的数据包至队列(update)；作出传输决策并完成传输，更新队列变化并统计时延(transform)；更新Qtable和mQ(move).

    @param positions: 某时刻某点的坐标
    @return: 平均时延
    """
    queue = np.zeros([total_num, 0]).tolist()  # [i][生成时间]
    qlen = np.zeros(total_num)  # 记录每个节点的长度，total_num,一开始队列长度都是0
    neighbors, qtable, m_q = initial_rl(positions)  # 初始化Qtable,mQ值
    sumdelay = 0  # 到目的节点的数据包的总时延
    count = 0  # 记录传输到目的节点的数据包数目
    for i in range(0, slots):
        # i是第n个时隙，帮助计算时延
        qlen, queue = update(qlen, queue, i)  # 是否采集新的数据包加入到队列#每个节点: 队多长，队列中的数据包（数据包的源头和生成时间）
        qlen, queue, sumdelay, count = transform_rl(qlen, queue, i, neighbors, positions, qtable, sumdelay,
                                                    count)  # 做出传输决策并完成传输过程，更新队列变化，统计时延
        neighbors, qtable, m_q = move_rl(positions, i, qtable, neighbors, m_q, qlen)  # 更新Qtable,mQ值
    print(sumdelay)
    print(count)
    if count == 0:
        count = 1
    avgdelay = sumdelay / count
    print(avgdelay)
    return avgdelay


def positions_of_file(input_file):
    """
    从mobility文件读入所有节点各时刻的位置信息，并根据给定源数目从仿真时间内始终在仿真区域的节点中随机选择出源节点集.

    @param input_file: 输入的文件路径
    @return: 某时刻某点的坐标, 节点数目（读取的数目）
    """
    num_per_slot = int(slots / num_slots)  # 时隙数/s
    x_bias = 200  # 用于校正两个场景之间的坐标差异，校正为0~2000
    y_bias = 0
    node_num = 0
    time_begin = 400
    time_end = 0
    f = open(input_file, "r")
    line = f.readline()
    while line:
        if line.startswith("$ns_"):
            nums = re.split('[ ()]', line)
            time_begin = min(time_begin, float(nums[2]))
            time_end = max(time_end, float(nums[2]))
            node_num = max(node_num, int(nums[4]))  # Ch
        line = f.readline()
    f.close()
    node_num = node_num + 1
    print("The total number of nodes is:")  # 把标号处理为总结点数目
    print(node_num)  # 指出给定的节点数目
    time_begin = int(time_begin)
    time_end = int(time_end)
    length = min(slots, (time_end - time_begin + 1) * num_per_slot)  # 总长度#slots
    positions = np.ones([length, node_num, 2]) * 4000  # 初始化位置远离仿真区域 每个时隙|每个点|对应一个位置x,y#查看一下position的值
    speed = np.ones(node_num) * (-1)
    f = open(input_file, "r")
    line = f.readline()
    while line:
        '''
        文件内的存储格式：
        分割后的表现形式：
        '''
        if line.startswith("$ns_"):
            nums = re.split(r'[ ()"]', line)
            now = int(float(nums[2]))
            identity = int(nums[5])
            x = float(nums[8]) - x_bias
            y = float(nums[9]) - y_bias
            sp = float(nums[10])  # 节点移动速度
            if now <= time_begin + num_slots:  # 只用了开始-》numslot的一段时间内的数据
                index = (now - time_begin - 1) * num_per_slot
                if speed[identity] == -1 and now < time_begin + num_slots or (
                        time_begin < now < time_begin + num_slots and
                        positions[index][identity][
                            0] == 4000):  # 无速度 & 十秒内 |（访问过且在十秒且内在范围外）
                    speed[identity] = sp
                    positions[(now - time_begin) * num_per_slot][identity][0] = x  # 多少秒*时隙/s=对应到的时隙
                    positions[(now - time_begin) * num_per_slot][identity][1] = y
                else:
                    if positions[index][identity][0] != 4000:  # 位置信息更新，时间段超出或有速度
                        for i in range(num_per_slot - 1):  # 因为num_是10，所以-1*10+8最后一个不赋值
                            # 平均一秒内位置的变化至每个时隙x
                            positions[index + i + 1][identity][0] = positions[index][identity][0] + (
                                        x - positions[index][identity][0]) / num_per_slot * (i + 1)
                            # 平均一秒内位置的变化至每个时隙y
                            positions[index + i + 1][identity][1] = positions[index][identity][1] + (
                                        y - positions[index][identity][1]) / num_per_slot * (i + 1)

                    if now < time_begin + num_slots:  # 给最后一个时隙赋值
                        positions[(now - time_begin) * num_per_slot][identity][0] = x
                        positions[(now - time_begin) * num_per_slot][identity][1] = y
                    speed[identity] = sp

            else:
                break
        line = f.readline()
    f.close()
    set_begin = []
    set_end = []
    set_same = []
    for m in range(node_num):
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
    print(sources)  # 抽出来所有在的，抽sense——num个。
    i = 0
    for a in sources:  # 把开始随机生成的表搞成顺序的。
        temp = positions[:, i, :]
        positions[:, i, :] = positions[:, a, :]
        positions[:, a, :] = temp
        i += 1

    return positions, node_num  # 来了的都算上


if __name__ == '__main__':
    start = time.time()
    lam = 300
    lamb = 1 / lam  # 仿真参数
    sense_numset = [3, 5, 7, 9]  # 源节点的数量
    out_RL = np.zeros(len(sense_numset))
    times = 1  # 循环次数，可修改
    for _ in range(0, times):
        n = 0
        for sense_num in sense_numset:
            poses, total_num = positions_of_file(
                "mapAoTi_50_150_v3.mobility.tcl")  # mobility文件读入地址，按自己实际保存位置修改
            out_RL[n] += begin_rl(poses)
            n += 1
    for idx in range(0, len(sense_numset)):  # 将十次不同源节点数目的图像取均值
        out_RL[idx] /= times
    print(out_RL)
    end = time.time()

    print("begin to end: ", end - start)
    plt.plot(sense_numset, out_RL, label="RL")
    plt.legend()
    plt.savefig("result.png")
    plt.show()
