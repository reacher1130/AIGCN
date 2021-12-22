from torch.autograd import Variable
from .util import *
import torch
import torch.nn as nn


class conv_1x1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(conv_1x1, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class SelfAttention(nn.Module):
    def __init__(self, in_channels, n_heads, dropout, scale=3):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.embeding_channels = in_channels // scale
        assert in_channels % n_heads == 0

        self.wq = nn.Linear(in_channels, self.embeding_channels)
        self.wk = nn.Linear(in_channels, self.embeding_channels)
        self.wv = nn.Linear(in_channels, self.embeding_channels)

        self.fc = nn.Linear(self.embeding_channels, in_channels)
        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.embeding_channels // n_heads]))

    def forward(self, query, key, value, mask=None):
        N = query.size(0)
        Q = self.wq(query)
        K = self.wk(key)
        V = self.wv(value)
        self.scale = self.scale.cuda()
        Q = Q.view(N, -1, self.n_heads, self.embeding_channels //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(N, -1, self.n_heads, self.embeding_channels //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(N, -1, self.n_heads, self.embeding_channels //
                   self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(torch.softmax(energy, dim=-1))
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(N, -1, self.n_heads * (self.embeding_channels // self.n_heads))

        x = self.fc(x)

        return x


class MLP(nn.Module):
    def __init__(self, n_in, n_hid, n_out, act_function=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout = nn.Dropout(p=0.2)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.init_weights()
        self.act_function = act_function

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.batch_norm(x)
        x = self.leaky_relu(x) if self.act_function else x
        return x


class attention_encoding_network(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(attention_encoding_network, self).__init__()
        self.mlp = MLP(in_channels * 2, out_channels, out_channels)
        self.nlp = MLP(in_channels * 2, out_channels, out_channels)
        self.init_weights()
        self.avg_pooling = nn.AdaptiveAvgPool1d(in_channels)
        self.attention1 = SelfAttention(out_channels, 8, 0.2, scale=1)
        self.attention2 = SelfAttention(out_channels, 8, 0.2, scale=1)
        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node_encoding(self, x, encode1, encode2):
        self_node = torch.matmul(encode1, x)
        other_node = torch.matmul(encode2, x)
        node_cat = torch.cat([self_node, other_node], dim=-1)
        return node_cat

    def edge_encoding(self, x, encode1):
        return_node = torch.matmul(encode1.t(), x)
        nodes = return_node / return_node.size(1)
        return nodes

    def forward(self, x1, x2, encode1, encode2):
        assert x1.size() == x2.size()
        N, C, T, V = x1.size()
        x1 = x1.permute(0, 2, 3, 1).contiguous().view(N, V, -1)
        x1 = self.avg_pooling(x1)
        x2 = x2.permute(0, 2, 3, 1).contiguous().view(N, V, -1)
        x2 = self.avg_pooling(x2)
        x1 = self.mlp(self.node_encoding(x1, encode1, encode2))
        x2 = self.nlp(self.node_encoding(x2, encode1, encode2))
        x1 = self.edge_encoding(x1, encode1)
        x2 = self.edge_encoding(x2, encode1)
        att_x1 = self.attention1(x2, x1, x1)
        att_x2 = self.attention2(x1, x2, x2)
        att1 = torch.cat([att_x1, att_x2], dim=1)
        att2 = torch.cat([att_x2, att_x1], dim=1)
        AttInfo = self.softmax(torch.matmul(att1, att2.permute(0, 2, 1).contiguous()))
        return normalize_dynamic_digraph(AttInfo)


class IAEGCN(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_point):
        super(IAEGCN, self).__init__()
        self.PA_double = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = self.A.size(0)
        self.num_joint = num_point
        self.embedding_channels = out_channels // 4

        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        if in_channels != out_channels:  # res
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        # interactive encoding attention graph
        self.attention_graph = attention_encoding_network(in_channels, self.embedding_channels)
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

        bn_init(self.bn, 1e-6)

        self.Linear_weight = nn.Parameter(torch.zeros(
            in_channels, out_channels * self.num_subset, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(
            0.5 / (out_channels * self.num_subset)))
        self.bn0 = nn.BatchNorm2d(out_channels * self.num_subset)
        bn_init(self.bn0, 1e-6)
        self.Linear_bias = nn.Parameter(torch.zeros(
            1, out_channels * self.num_subset, 1, 1, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.constant(self.Linear_bias, 1e-6)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA_double
        diag_joint = np.ones([self.num_joint, self.num_joint]) - np.eye(self.num_joint, self.num_joint)
        self_joint_pos = torch.FloatTensor(np.array(encode_onehot(np.where(diag_joint)[1]), dtype=np.float32)).cuda()
        other_joint_pos = torch.FloatTensor(np.array(encode_onehot(np.where(diag_joint)[0]), dtype=np.float32)).cuda()
        x1, x2 = torch.split(x, self.num_joint, dim=-1)
        # TODO 是否聚合单人 邻接矩阵信息

        x_st = torch.einsum('nctw,cd->ndtw', (x, self.Linear_weight)).contiguous()
        x_st = x_st + self.Linear_bias
        x_st = self.bn0(x_st)

        n, kc, t, v = x_st.size()
        x_st = x_st.view(n, self.num_subset, kc // self.num_subset, t, v)
        x_st = torch.einsum('nkctv,kvw->nctw', (x_st, A))

        att_encod_graph = self.attention_graph(x1, x2, self_joint_pos, other_joint_pos)   # 只需要一个 50*50 的邻接矩阵就行
        x_dy = x.view(N, C * T, V)
        x_dy = self.conv(torch.matmul(x_dy, att_encod_graph).view(N, C, T, V))

        z = x_st + x_dy
        z = self.bn(z)
        z += self.down(x)
        return self.relu(z)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class IAMTemporalConv(nn.Module):

    def __init__(self, in_channels, out_channels, temporal, kernel_size=5, stride=1,
                 dilations=[1, 2]):
        super(IAMTemporalConv, self).__init__()
        assert out_channels % (len(dilations) + 2) == 0
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.apply(weights_init)
        self.self_attention1 = SelfAttention(temporal, 5, 0.2)
        self.self_attention2 = SelfAttention(temporal, 5, 0.2)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        N, C, T, V = x.size()
        x1, x2 = torch.split(x, V // 2, dim=-1)
        x1_skip, x2_skip = x1, x2

        x1_skip = self.avg_pooling(x1_skip.permute(0, 2, 3, 1).contiguous().view(N, T * V // 2, C)).squeeze(-1)
        x2_skip = self.avg_pooling(x2_skip.permute(0, 2, 3, 1).contiguous().view(N, T * V // 2, C)).squeeze(-1)
        x1_skip = x1_skip.view(N, T, V // 2).permute(0, 2, 1).contiguous()
        x2_skip = x2_skip.view(N, T, V // 2).permute(0, 2, 1).contiguous()
        # TODO motion1 和 motion2 做attention融合 得到一个新的mask

        #  N V T
        mask1 = self.self_attention1(x2_skip, x1_skip, x1_skip).unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        mask2 = self.self_attention2(x1_skip, x2_skip, x2_skip).unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        mask1 = self.bn(mask1)
        mask2 = self.bn(mask2)
        x1 = torch.sigmoid(mask1) * x1 + x1
        x2 = torch.sigmoid(mask2) * x2 + x2
        x = torch.cat([x1, x2], dim=-1)

        branch_outs = []
        for tempConv in self.branches:
            out = tempConv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        return out


class AIGCBs(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_point, temporal, stride=1, residual=True):
        super(AIGCBs, self).__init__()
        self.gcn = IAEGCN(in_channels, out_channels, A, num_point)
        self.tcn = IAMTemporalConv(out_channels, out_channels, temporal, kernel_size=7, stride=stride, dilations=[1, 2])
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = conv_1x1(in_channels, out_channels, kernel_size=1, stride=stride)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x):

        x = self.tcn(self.gcn(x)) + self.residual(x)
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, num_class=26, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        # A = self.graph.A
        A = self.graph.A_neighbor
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.layers = nn.ModuleList([
            AIGCBs(3, 64, A, num_point, temporal=300, residual=False),
            AIGCBs(64, 64, A, num_point, temporal=300),
            AIGCBs(64, 128, A, num_point, temporal=300, stride=2),
            AIGCBs(128, 128, A, num_point, temporal=150),
            AIGCBs(128, 256, A, num_point, temporal=150, stride=2),
            AIGCBs(256, 256, A, num_point, temporal=75),
        ])
        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):

        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 3, 4, 1, 2).contiguous().view(N, C, T, M * V)
        for layer in self.layers:
            x = layer(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.mean(3).mean(1)
        return self.fc(x)
