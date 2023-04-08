import numpy as np
import random as rd
import matplotlib.pyplot as plt
import csv


dim = ['alcohol', 'apple-acid', 'grey', 'ash-content', 'Mg', 'phenolics', 'flavonoid',
       '非类黄酮酚', '原花青素', '颜色强度', '色调', '稀释葡萄酒的OD280 / OD315', '脯氨酸']


def calculate_dis(centroid, data):
    dis = 0
    for i in range(13):
        dis += pow(centroid[i] - float(data[i]), 2)
    return dis


def read_data(data_file):
    file = open(data_file)
    reader = csv.reader(file)
    data = []
    for line in reader:
        line = list(map(float, line))
        data.append(line)
    return data


def init_centroids(k):
    centroids = []
    for i in range(k):
        center = []
        for dim in range(13):
            center.append(rd.random())
        centroids.append(center)
    return centroids


def calculate_sse(d_min, num):
    cen_sse = [0, 0, 0]
    nums = [0, 0, 0]
    for i in range(num):
        cen_sse[int(d_min[i, 0]) - 1] += d_min[i, 1]
        nums[int(d_min[i, 0]) - 1] += 1
    print("第一个聚类的SSE为：{}, 数量为 {}.".format(cen_sse[0], nums[0]))
    print("第二个聚类的SSE为：{}, 数量为 {}.".format(cen_sse[1], nums[1]))
    print("第三个聚类的SSE为：{}, 数量为 {}.".format(cen_sse[2], nums[2]))

    sse = cen_sse[0] + cen_sse[1] + cen_sse[2]
    print("SSE为：{}.".format(sse))
    return sse


# centroid 1,2,3
def get_cluster_points(data, d_min, num, centroid):
    points = []
    for i in range(num):
        if d_min[i, 0] == centroid:
            points.append(data[i])
    return points


def calculate_acc(data, d_min, num, k):
    hit = 0
    for i in range(k):
        cluster_temp = [0, 0, 0]
        points = get_cluster_points(data, d_min, num, i+1)
        # print(len(points))
        for point in points:
            cluster_temp[int(point[0]) - 1] += 1
            # print(point[0])
        hit += max(cluster_temp)
    acc = hit / len(data)
    print("准确度为：{}.".format(acc))
    return acc


def plot_clusters(data, d_min, centers, num, k):
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    print('请输入x,y轴对应的属性，范围0~12')
    print('x: ')
    x = int(input())
    print('y: ')
    y = int(input())
    for i in range(k):
        points = get_cluster_points(data, d_min, num, i+1)
        points = np.array(points)
        # print(np.shape(points))
        plt.scatter(points[:, x], points[:, y], c=colors[i])
    plt.scatter(centers[:, x], centers[:, y], marker='*', s=200, c='black')
    plt.show()


if __name__ == "__main__":
    data = read_data('./normalizedwinedata.csv')
    d = np.array(data)
    train_data = d[:, 1:]
    number_of_sample = len(train_data)
    data_min = np.mat(np.zeros((number_of_sample, 2)))  # 第0列存放质心，第一列存放距离
    # print(number_of_sample)
    k = 3
    # 初始化质心
    centroids = init_centroids(k)

    change = True
    epc = 0
    while change:
        epc += 1
        change = False
        for i in range(number_of_sample):
            min_dis = 100000000.0
            min_center = -1
            # 计算样本到每个质心的距离
            for j in range(k):
                dis = calculate_dis(centroids[j], train_data[i])
                if dis < min_dis:
                    min_center = j + 1
                    min_dis = dis
            if data_min[i, 0] != min_center or data_min[i, 1] != min_dis:
                # 划分
                data_min[i, :] = min_center, min_dis
                change = True

        centroids = np.array(centroids)
        for j in range(1, k+1):
            points = get_cluster_points(train_data, data_min, number_of_sample, j)
            old = centroids[j - 1]
            # print("before : {}".format(old))
            centroids[j - 1, :] = np.mean(points, axis=0)
            # print("after : {}".format(centroids[j-1]))
            # print(centroids[j - 1])
            # if (old == centroids[j - 1, :]).all() == False:
                # change = True

    print("循环次数为：{}.".format(epc))

    calculate_sse(data_min, number_of_sample)
    calculate_acc(data, data_min, number_of_sample, k)

    plot_clusters(train_data, data_min, centroids, number_of_sample, k)

