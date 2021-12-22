import sys
import numpy as np

sys.path.extend(['../'])


def edge2mat(link, num_node):  # 计算节点间的邻接矩阵
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


# 计算 度矩阵然后得到拉普拉斯矩阵 L^~ 随机游走
def normalize_digraph(A):  # 除以每列的和
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    DA = np.dot(Dn, A)
    return DA


def get_spatial_graph_list(num_node, node_list):
    new_list = []
    for i in range(len(node_list)):
        each_node = normalize_digraph(edge2mat(node_list[i], num_node))
        new_list.append(each_node)
    A = np.stack(new_list)
    return A


num_node_two = 30
self_link_two = [(i, i) for i in range(num_node_two)]
inward_ori_index_two = [(3, 2), (3, 4), (3, 7), (3, 10), (3, 13), (2, 1), (2, 4), (2, 7), (4, 5), (5, 6), (7, 8),
                        (8, 9), (10, 11), (11, 12), (13, 14), (14, 15), (18, 17), (18, 19), (18, 22), (18, 25),
                        (18, 28), (17, 16), (17, 19), (17, 22), (19, 20), (20, 21), (22, 23), (23, 24), (25, 26),
                        (26, 27),
                        (28, 29), (29, 30), (1, 16), (2, 17), (3, 18), (4, 22), (5, 23), (6, 24), (7, 19), (8, 20),
                        (9, 21), (10, 28), (11, 29), (12, 30), (13, 25), (14, 26), (15, 27)]

inward_mirror = [(i - 1, j - 1) for (i, j) in inward_ori_index_two]
outward_mirror = [(j, i) for (i, j) in inward_mirror]
neighbor_mirror = inward_mirror + outward_mirror


class TwoGraph:
    def __init__(self, labeling_mode='spatial'):

        # double people adj
        self.num_node_two = num_node_two
        self.self_link_two = self_link_two
        self.inward_mirror = inward_mirror
        self.outward_mirror = outward_mirror
        self.neighbor_mirror = neighbor_mirror

        self.node_list = [self_link_two, inward_mirror, outward_mirror]
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A_neighbor = self.get_neighbor_matrix(self.num_node_two, [self.self_link_two, self.neighbor_mirror])

    def get_neighbor_matrix(self, num_node, node_list, labeling_mode='spatial'):
        if labeling_mode == 'spatial':
            A_neighbor = get_spatial_graph_list(num_node, node_list)
        else:
            raise ValueError()
        return A_neighbor

    def get_adjacency_matrix(self, labeling_mode=None):

        if labeling_mode == 'spatial':
            A = get_spatial_graph_list(self.num_node_two, self.node_list)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    A = TwoGraph().A_neighbor
    for i in A:
        plt.imshow(i, cmap='Blues')
        plt.show()
    print(A)
