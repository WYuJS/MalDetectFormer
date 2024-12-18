import dgl
import torch
import numpy as np
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from torch.utils.data import Dataset, DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练准备：定义一个简单的图网络
class IPGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats):
        super(IPGCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats, allow_zero_in_degree=True).to(device)
        self.conv2 = GraphConv(hidden_feats, hidden_feats, allow_zero_in_degree=True).to(device)
        self.Linear = nn.Linear(hidden_feats * 2, hidden_feats).to(device)  # 用于边的分类
        # self.relu = nn.ReLU().double()

    def forward(self, g, features, src, dst, node_name_to_index):
        '''g:图，features:全部特征
        src:源ip，dst:目标ip，node_name_to_index:ip在g图中的索引'''
        x = self.conv1(g, features)
        x = F.relu(x)
        node_feats = self.conv2(g, x)

        # 对应节点的属性
        src_node = [node_name_to_index[key.item()] for key in src if key.item() in node_name_to_index]
        dst_node = [node_name_to_index[key.item()] for key in dst if key.item() in node_name_to_index]

        edge_repr = torch.cat((node_feats[src_node], node_feats[dst_node]), dim=1)  # 聚合边的两个节点表示
        edge_pred = self.Linear(edge_repr) #边的预测
        return edge_pred



class NetGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats):
        super(NetGCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats, allow_zero_in_degree=True).to(device)
        self.conv2 = GraphConv(hidden_feats, hidden_feats, allow_zero_in_degree=True).to(device)
        self.Linear = nn.Linear(hidden_feats * 2, hidden_feats).to(device)  # 用于边的分类

    def forward(self, g, features, src, dst, node_name_to_index):
        '''g:图，features:全部特征
                src:源ip，dst:目标ip，node_name_to_index:ip在g图中的索引'''
        x = self.conv1(g, features)
        x = F.relu(x)
        node_feats = self.conv2(g, x)

        # 对应节点的属性
        src_node = [node_name_to_index[key.item()] for key in src if key.item() in node_name_to_index]
        dst_node = [node_name_to_index[key.item()] for key in dst if key.item() in node_name_to_index]

        edge_repr = torch.cat((node_feats[src_node], node_feats[dst_node]), dim=1)  # 聚合边的两个节点表示
        edge_pred = self.Linear(edge_repr)  # 边的预测
        return edge_pred

class EGCN(nn.Module):
    # def __init__(self, in_feats, hidden_feats, num_classes):
    def __init__(self, in_feats, hidden_feats):
        super(EGCN, self).__init__()
        self.ipGCN = IPGCN(2, hidden_feats).to(device) #可以改成in_feats,此处改成2是只考虑其空间性
        self.netGCN = NetGCN(2, hidden_feats).to(device)#同上
        self.Linear = nn.Linear(hidden_feats * 2, in_feats).to(device)
        self.norm = nn.LayerNorm(in_feats)
        self.relu = nn.ReLU()
        # self.classifier = nn.Linear(in_feats, num_classes).to(device)

    def forward(self, g, G, src_ip, dst_ip, src_net, dst_net, g_node_name_to_index, G_node_name_to_index):
        edge1 = self.ipGCN(g, g.ndata['features'], src_ip, dst_ip, g_node_name_to_index)
        edge2 = self.netGCN(G, G.ndata['features'], src_net, dst_net, G_node_name_to_index)
        # edge2 = edge1
        edge_repr = torch.cat((edge1, edge2), dim=1)
        edge_prer = self.Linear(edge_repr) #输入进NetSentinelformer的部分
        edge_prer = self.relu(self.norm(edge_prer))
        # edge_pred = self.classifier(edge_prer) #训练时的预测部分
        # return edge_prer.to(device), edge_pred.to(device)
        return edge_prer.to(device)

def getGraph(flow_data):
    # g, ip_node_name_to_index = create_graph_from_flows(flow_data, is_net=False)
    # G, net_node_name_to_index = create_graph_from_flows(flow_data, is_net=True)
    g, ip_node_name_to_index = build_graph_with_features(flow_data, is_net=False)
    G, net_node_name_to_index = build_graph_with_features(flow_data, is_net=True)
    egcn = {}
    egcn['g'] = g
    egcn['G'] = G
    egcn['ip_node_name_to_index'] = ip_node_name_to_index
    egcn['net_node_name_to_index'] = net_node_name_to_index
    return egcn


# 训练EGCN的模型
def train_egcn_model(flow_data, labels, hidden_feats=16, epochs=100, learning_rate=0.01, batch_size=32):
    g, ip_node_name_to_index = create_graph_from_flows(flow_data, is_net=False)
    G, net_node_name_to_index = create_graph_from_flows(flow_data, is_net=True)
    in_feats = flow_data.shape[1]
    num_classes = torch.unique(labels).numel()  # 假设标签是二分类的，0表示正常，1表示异常

    model = EGCN(in_feats, hidden_feats, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(flow_data, labels.float())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    best_loss = float('inf') 
    for epoch in range(epochs):
        current_loss = 0
        for idx, (data_batch, labels_batch) in enumerate(dataloader):
            _, edge_pred = model(g, G, data_batch[:, 0], data_batch[:, 1], data_batch[:, 2], data_batch[:, 3], \
                              ip_node_name_to_index, net_node_name_to_index)

            edge_pred = edge_pred.float().to(device)
            labels_batch = labels_batch.long().to(device)
            loss = criterion(edge_pred, labels_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            current_loss += loss.item()

            if idx % 100 == 0:
                print(f"Epoch {epoch}   {idx}: Loss {loss.item()}")
        # 判断终止训练
        if current_loss < best_loss:
            best_loss = current_loss
        else:
            break
        if epoch > 0:
            if np.abs(current_loss - best_loss) < 0.0005:
                break

    egcn = {}
    egcn['model'] = model
    egcn['g'] = g
    egcn['G'] = G
    egcn['ip_node_name_to_index'] = ip_node_name_to_index
    egcn['net_node_name_to_index'] = net_node_name_to_index
    return egcn


def build_graph_with_features(data, is_net=False):
    """
    构建图，并为每个IP地址计算作为源地址和目的地址时的特征均值。
    
    参数:
    - data: 二维列表，包含[src_ip, dst_ip, src_feature1, dst_feature1, src_feature2, dst_feature2]
    
    返回:
    - g: DGL图对象。
    - ip_to_index: IP地址到图节点索引的映射字典。
    """
    # 将数据转换为NumPy数组
    data_np = np.array(data.detach().cpu(), dtype=object)

    # 提取所有唯一的IP地址
    if is_net:
        ips = np.unique(data_np[:,2:4])
    else:
        ips = np.unique(data_np[:, :2])

    # 创建IP到索引的映射
    ip_to_index = {ip: i for i, ip in enumerate(ips)}
    # 初始化特征聚合字典
    features_dict = {ip: [] for ip in ips}
    
    # 聚合特征
    for row in data_np:
        if is_net:
            src_ip , dst_ip = row[2], row[3]
        else:
            src_ip, dst_ip = row[0], row[1]
        features_dict[src_ip].append([float(row[0]), float(row[2])])  # 第一列和第三列数据作为源地址特征
        features_dict[dst_ip].append([float(row[1]), float(row[3])])  # 第二列和第四列数据作为目的地址特征

    # 计算特征均值
    num_features = 2  # 每个IP有2个特征
    features = np.zeros((len(ips), num_features))
    for ip, feats in features_dict.items():
        features[ip_to_index[ip]] = np.mean(feats, axis=0)
    
    # 创建DGL图
    if is_net:
        src_indices = [ip_to_index[ip] for ip in data_np[:, 2]]
        dst_indices = [ip_to_index[ip] for ip in data_np[:, 3]]
    else:
        src_indices = [ip_to_index[ip] for ip in data_np[:, 0]]
        dst_indices = [ip_to_index[ip] for ip in data_np[:, 1]]
    g = dgl.graph((src_indices, dst_indices)).to(device)
    # 将特征数组转换为张量并分配给图的节点
    g.ndata['features'] = torch.tensor(features, dtype=torch.float32).to(device)
    
    return g, ip_to_index

def create_graph_from_flows(flows, is_net=False):
    """
    创建一个DGL图，根据流量数据，其中前两列是节点ID，其他列是边的属性。
    每个节点的特征由连接到该节点的所有边的属性的平均值确定。

    参数:
    - flow_data: 流量数据，numpy数组或列表的列表形式。
    - is_net:True是net， False是ip
    返回:
    - DGL图
    """
    src_nodes = []
    dst_nodes = []
    edge_features = []
    node_name_to_index = {}  # 节点名称到索引的映射
    current_index = 0

    for idx, flow in enumerate(flows):
        # print(idx)
        src_ip = flow[2].item() if is_net else flow[0].item()
        dst_ip = flow[3].item() if is_net else flow[1].item()
        # 检查源IP节点是否已经在映射中
        if src_ip not in node_name_to_index:
            node_name_to_index[src_ip] = current_index
            current_index += 1
        src_index = node_name_to_index[src_ip]

        # 检查目的IP节点是否已经在映射中
        if dst_ip not in node_name_to_index:
            node_name_to_index[dst_ip] = current_index
            current_index += 1
        dst_index = node_name_to_index[dst_ip]

        src_nodes.append(src_index)
        dst_nodes.append(dst_index)
        # edge_features.append(list(flow)) #全部信息的均值
        edge_features.append([flow[0], flow[1], flow[2], flow[3]]) # 

    # edge_features = np.array(edge_features)
    edge_features = torch.tensor(edge_features).to(device)

    # 创建图
    g = dgl.graph((src_nodes, dst_nodes))
    g = g.to(device)
    g.retain_graph = True

    # 为每条边添加特征
    g.edata['feat'] = (edge_features).clone().to(device)

    # 计算每个节点的特征（边特征的平均值）
    node_features = defaultdict(list)
    for src, dst, feat in zip(src_nodes, dst_nodes, edge_features):
        node_features[src].append(feat)
        node_features[dst].append(feat)

    # 计算平均特征并转换为tensor
    avg_node_features = []
    for node_id in range(g.number_of_nodes()):
        if node_id in node_features.keys():
            avg_feat = torch.stack(node_features[node_id]).mean(dim=0)
        else:
            avg_feat = torch.zeros(edge_features.shape[1], )  # 无边的节点使用0向量
        avg_node_features.append(avg_feat)

    g.ndata['feat'] = torch.stack(avg_node_features).to(device)

    return g, node_name_to_index