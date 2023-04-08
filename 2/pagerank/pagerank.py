import numpy as np


def read_data(data_file):
    f = open(data_file)
    edges = [line.strip('\n').split(',') for line in f]
    edges = edges[1:]
    nodes = []
    for edge in edges:
        if edge[1] not in nodes:
            nodes.append(edge[1])
        if edge[2] not in nodes:
            nodes.append(edge[2])

    return edges, nodes


def init_matrix(edges, nodes):
    l = len(edges)
    n = len(nodes)

    m = np.zeros([n, n])

    for edge in edges:
        i = nodes.index(edge[1])
        j = nodes.index(edge[2])
        m[j, i] = 1

    for i in range(n):
        di = sum(m[:, i])
        for j in range(n):
            if di != 0:
                m[j, i] /= di

    r = np.ones(n) / n

    return m, r


if __name__ == '__main__':
    edges, nodes = read_data('../data/sent_receive.csv')
    n = len(nodes)
    m, r = init_matrix(edges, nodes)
    next_r = np.zeros(n)
    e = 300000000
    epoch = 0
    b = 0.85

    while e >= 0.00000001:
        next_r = np.dot(m, r) * b + (1-b) / n * np.ones(n)
        di = sum(next_r)
        next_r = next_r / di
        e_vector = abs(next_r-r)
        e = sum(e_vector)
        r = next_r
        epoch += 1

    print('rank vector:')
    for i, rank in enumerate(r):
        print("id: {}, rank score: {}".format(nodes[i], rank))



