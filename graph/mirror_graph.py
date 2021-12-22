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


# 独立邻接矩阵
num_node_two = 50
self_link_two = [(i, i) for i in range(num_node_two)]
inward_index_independence = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                             (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                             (20, 19), (22, 23), (23, 8), (24, 25), (25, 12), (26, 27), (27, 46), (28, 46), (29, 28),
                             (30, 46), (31, 30), (32, 31), (33, 32), (34, 46), (35, 34), (36, 35), (37, 36), (38, 26),
                             (39, 38), (40, 39), (41, 40), (42, 26), (43, 42), (44, 43), (45, 44), (47, 48), (48, 33),
                             (49, 50), (50, 37)]
inward_independence = [(i - 1, j - 1) for (i, j) in inward_index_independence]
outward_independence = [(j, i) for (i, j) in inward_independence]
neighbor_independence = inward_independence + outward_independence

# 独立邻接矩阵 + 镜像对称
inward_ori_index_mirror = [(29, 4), (28, 3), (46, 21), (27, 2), (26, 1), (49, 22), (50, 23), (37, 8), (36, 7),
                           (35, 6), (34, 5), (30, 9), (31, 10), (32, 11), (33, 12), (48, 25), (47, 27), (42, 17),
                           (43, 18), (44, 19), (45, 20), (38, 13), (39, 14), (40, 15), (41, 16)]
inward_ori_index_mirror = inward_ori_index_mirror + inward_index_independence
# print(inward_ori_index_mirror)
inward_mirror = [(i - 1, j - 1) for (i, j) in inward_ori_index_mirror]
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
