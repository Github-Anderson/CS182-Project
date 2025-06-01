from argparse import Namespace
import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from fpgnn.data import GetPubChemFPs, create_graph, get_atom_features_dim
import torch.nn.functional as F
import csv
import torch
import torch.nn as nn

atts_out = []

# GATLayer: 图注意力网络层
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.dropout = nn.Dropout(dropout)

        # 权重矩阵初始化
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 注意力权重向量
        self.a = nn.Parameter(torch.empty(size=(2 * out_features + 1, 1)))  # 2 * out_features + 1 是因为拼接了两节点特征 + 边特征
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, edge_feat):
        # 确保输入 h 的形状是 (N, in_features)
        Wh = torch.mm(h, self.W)  # (N, out_features)
        N = Wh.size(0)

        # 构造节点对特征矩阵，形状为 (N, N, 2 * out_features)
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)  # (N * N, 2 * out_features)
        all_combinations_matrix = all_combinations_matrix.view(N, N, 2 * self.out_features)  # (N, N, 2 * out_features)

        # 处理边特征并拼接
        if edge_feat.dim() == 2:
            edge_feat_expanded = edge_feat.unsqueeze(2)  # (N, N, 1)
        else:
            edge_feat_expanded = edge_feat  # 已经是 (N, N, d) 类型

        # 拼接边特征
        concat_features = torch.cat([all_combinations_matrix, edge_feat_expanded], dim=2)  # (N, N, 2 * out_features + 1)

        # 计算注意力分数
        e = self.leakyrelu(torch.matmul(concat_features, self.a).squeeze(2))  # (N, N)

        # 只对邻居节点计算注意力，非邻居设为极小
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        # 归一化注意力权重
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        # 加权邻居特征聚合
        h_prime = torch.matmul(attention, Wh)  # (N, out_features)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


# GATOne: 多头图注意力网络
class GATOne(nn.Module):
    def __init__(self, args):
        super(GATOne, self).__init__()
        self.nfeat = 133  # 输入节点特征维度
        self.nhid = args.nhid  # 每头隐藏维度
        self.dropout = args.dropout_gat
        self.nheads = args.nheads
        self.alpha = 0.2
        self.out_dim = args.hidden_size  # 最终输出维度

        # 多头注意力层
        self.attentions = nn.ModuleList(
            [GATLayer(self.nfeat, self.nhid, self.dropout, self.alpha, concat=True) for _ in range(self.nheads)]
        )

        # 最后一层注意力层，将多头拼接输出映射到目标维度
        self.out_att = GATLayer(self.nhid * self.nheads, self.out_dim, self.dropout, self.alpha, concat=False)

    def forward(self, h, adj, edge_feat):
        # 多头注意力计算并拼接
        x = torch.cat([att(h, adj, edge_feat) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj, edge_feat))
        return x


# GATEncoder: 编码器，提取分子图特征
class GATEncoder(nn.Module):
    def __init__(self, args):
        super(GATEncoder, self).__init__()
        self.args = args
        self.gat = GATOne(args)
        self.device = torch.device('cuda' if args.cuda else 'cpu')

    def get_edge_features(self, mol):
        """
        从 RDKit 分子对象中提取边特征。
        返回一个 N x N 的矩阵，表示每一对原子之间的边特征。
        例如，用化学键类型的数值编码（0 = no bond, 1 = single, 2 = double, 3 = triple, 4 = aromatic）。
        """
        N = mol.GetNumAtoms()  # 原子数目
        edge_feat = torch.zeros((N, N), device=self.device)  # 初始化边特征矩阵，形状为 (N, N)

        # 遍历所有键
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()  # 获取键的开始原子索引
            j = bond.GetEndAtomIdx()  # 获取键的结束原子索引
            bond_type = bond.GetBondType()  # 获取化学键类型

            # 用数值表示不同类型的化学键
            if bond_type == Chem.rdchem.BondType.SINGLE:
                val = 1.0
            elif bond_type == Chem.rdchem.BondType.DOUBLE:
                val = 2.0
            elif bond_type == Chem.rdchem.BondType.TRIPLE:
                val = 3.0
            elif bond_type == Chem.rdchem.BondType.AROMATIC:
                val = 4.0
            else:
                val = 0.0  # 没有连接

            # 对称地设置边特征矩阵，因为化学键是无方向的
            edge_feat[i, j] = val
            edge_feat[j, i] = val  # 化学键是无方向的

        return edge_feat

    def forward(self, mols, smiles):
        # 处理 SMILES 列表，生成分子图结构
        mol = create_graph(smiles, self.args)
        atom_feature, atom_index = mol.get_feature()
        if self.args.cuda:
            atom_feature = atom_feature.cuda()

        outputs = []
        for i, smi in enumerate(smiles):
            mol_i = Chem.MolFromSmiles(smi)
            adj = Chem.rdmolops.GetAdjacencyMatrix(mol_i)  # 获取邻接矩阵
            adj = torch.tensor(adj, dtype=torch.float32, device=self.device)

            # 获取边特征
            edge_feat = self.get_edge_features(mol_i)

            atom_start, atom_size = atom_index[i]
            h = atom_feature[atom_start:atom_start + atom_size]

            # 计算 GAT 输出
            gat_out = self.gat(h, adj, edge_feat)
            mol_repr = gat_out.mean(dim=0)
            outputs.append(mol_repr)

        outputs = torch.stack(outputs)
        return outputs


# GAT: 模型主类
class GAT(nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        self.encoder = GATEncoder(args)

    def forward(self, smiles):
        # 直接把 SMILES 列表传给 encoder
        gat_out = self.encoder(None, smiles)
        return gat_out

import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from fpgnn.data import GetPubChemFPs  # 确保你的项目里有这个函数

import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from fpgnn.data import GetPubChemFPs  # 请根据项目实际调整导入

class FPN(nn.Module):
    def __init__(self,args):
        super(FPN, self).__init__()
        self.fp_2_dim=args.fp_2_dim
        self.dropout_fpn = args.dropout
        self.cuda = args.cuda
        self.hidden_dim = args.hidden_size
        self.args = args
        if hasattr(args,'fp_type'):
            self.fp_type = args.fp_type
        else:
            self.fp_type = 'mixed'
        
        if self.fp_type == 'mixed':
            self.fp_dim = 1489 # 167 + 881 + 441
        else:
            self.fp_dim = 1024
        
        if hasattr(args,'fp_changebit'):
            self.fp_changebit = args.fp_changebit
        else:
            self.fp_changebit = None
        
        self.fc1=nn.Linear(self.fp_dim, self.fp_2_dim)
        self.act_func = nn.ReLU()
        self.fc2 = nn.Linear(self.fp_2_dim, self.hidden_dim)
        self.dropout = nn.Dropout(p=self.dropout_fpn)
    
    def forward(self, smile):
        fp_list=[]
        for i, one in enumerate(smile):
            fp=[]
            mol = Chem.MolFromSmiles(one)
            
            if self.fp_type == 'mixed':
                fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
                fp_phaErGfp = AllChem.GetErGFingerprint(mol,fuzzIncrement=0.3,maxPath=21,minPath=1)
                fp_pubcfp = GetPubChemFPs(mol)
                fp.extend(fp_maccs)
                fp.extend(fp_phaErGfp)
                fp.extend(fp_pubcfp)
            else:
                fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fp.extend(fp_morgan)
            # raise RuntimeError(type(fp_maccs), type(fp_phaErGfp), type(fp_pubcfp))
            # rdkit.DataStructs.cDataStructs.ExplicitBItVect, numpy array, numpy array
            fp_list.append(fp)
                
        if self.fp_changebit is not None and self.fp_changebit != 0:
            fp_list = np.array(fp_list)
            fp_list[:,self.fp_changebit-1] = np.ones(fp_list[:,self.fp_changebit-1].shape)
            fp_list.tolist()
        fp_list = torch.Tensor(fp_list)
        # raise RecursionError(fp_list.shape) # torch.Size([50, 1489])
        # Indicating that simply using torch.Tensor, we can change these three into a joint tensor
        if self.cuda:
            fp_list = fp_list.cuda()
        fpn_out = self.fc1(fp_list)
        fpn_out = self.dropout(fpn_out)
        fpn_out = self.act_func(fpn_out)
        fpn_out = self.fc2(fpn_out)
        return fpn_out



# === Cross-Attention 融合模块 ===
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim):
        super(CrossAttentionFusion, self).__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, x_q, x_kv):
        q = self.q(x_q)  # (B, D)
        k = self.k(x_kv)
        v = self.v(x_kv)

        # 注意力计算
        score = torch.matmul(q.unsqueeze(1), k.unsqueeze(-1)) / (q.size(-1) ** 0.5)  # (B, 1, 1)
        weight = torch.softmax(score, dim=-1)  # (B, 1, 1)
        context = weight.squeeze(-1) * v  # (B, D)

        fused = q + context  # 残差连接
        return self.act(self.out(fused))


# === 主模型类 FpgnnModel ===
class FpgnnModel(nn.Module):
    def __init__(self, is_classif, gat_scale, cuda, dropout_fpn):
        super(FpgnnModel, self).__init__()
        self.gat_scale = gat_scale
        self.is_classif = is_classif
        self.cuda = cuda
        self.dropout_fpn = dropout_fpn
        if self.is_classif:
            self.sigmoid = nn.Sigmoid()

    def create_gat(self, args):
        self.encoder3 = GAT(args)

    def create_fpn(self, args):
        self.encoder2 = FPN(args)

    def create_scale(self, args):
        linear_dim = args.hidden_size
        if 0 < self.gat_scale < 1:
            self.cross_attn = CrossAttentionFusion(dim=linear_dim)
        elif self.gat_scale == 1:
            self.fc_gat = nn.Linear(linear_dim, linear_dim)
        elif self.gat_scale == 0:
            self.fc_fpn = nn.Linear(linear_dim, linear_dim)
        self.act_func = nn.ReLU()

    def create_ffn(self, args):
        linear_dim = args.hidden_size
        self.ffn = nn.Sequential(
            nn.Dropout(self.dropout_fpn),
            nn.Linear(in_features=linear_dim, out_features=linear_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_fpn),
            nn.Linear(in_features=linear_dim, out_features=args.task_num)
        )

    def forward(self, input):
        if self.gat_scale == 1:
            output = self.encoder3(input)
        elif self.gat_scale == 0:
            output = self.encoder2(input)
        else:
            gat_out = self.encoder3(input)  # 图结构向量
            fpn_out = self.encoder2(input)  # 指纹向量
            output = self.cross_attn(gat_out, fpn_out)  # 深度融合

        output = self.ffn(output)

        if self.is_classif and not self.training:
            output = self.sigmoid(output)
        return output
def get_atts_out():
    return atts_out

def FPGNN(args):
    # First we decide it is a classification task 
    if args.dataset_type == 'classification':
        is_classif = 1
    else:
        is_classif = 0
    # Instantiating model
    # The latter two parameters are easy to understand: the device we are using, and the drop rate during training
    model = FpgnnModel(is_classif,args.gat_scale,args.cuda,args.dropout)
    '''
    In the paper, there are three types of extracted molecule fingerprint features
    And there is a Graphical Attentional Network to extract features
    For the fully connnected layers, there are three mode of concatenation to choose:
    1. gat_scale == 1: means that we don't use the fingerprint features in FFN
    2. gat_scale == 0: means that we don't use the Graphical Attentional Network features in FFN
    3. In (0, 1), then : 
        self.gat_dim = int((linear_dim*2*self.gat_scale)//1)
        self.fc_gat = nn.Linear(linear_dim,self.gat_dim)
        self.fc_fpn = nn.Linear(linear_dim,linear_dim*2-self.gat_dim)
    It means that the original dimension of the both GAN and Fingerprint's aftew-FFN feature are linear_dim. and the concatenated dimension is 2*linear_dim
    With scale, the portion in dimension can be altered through linear operation. If linear_dim = 128, and gat_scale = 0.25, then in the final 256-dim concatenated tensor
    GAT takes up 64 dims, and Fingerprint takes up 192, and the change for GAT from 128-dim to 64-dim is realizaed through nn.Linear
    '''
    # raise RuntimeError(args)
    '''
    Namespace(batch_size=50, cuda=True, data_path='Data/MoleculeNet/bace.csv', dataset_type='classification', dropout_gat=0.0, epochs=30, final_lr=0.0001, fp_2_dim=512,
    gat_scale=0.5, hidden_size=300, init_lr=0.0001, is_multitask=0, log_path='log/bace', max_lr=0.001, metric='auc', nheads=8, nhid=60, num_lrs=1, num_folds='model_save/bace/Seed_0',
    seed=0, split_ratio=[0.8, 0.1, 0.1], split_type='random', task_num=1, test_path=None, train_data_size=1210, val_pupch=1.0, warmup_epochs=2.0)
    '''
    if args.gat_scale == 1:
        model.create_gat(args)
        model.create_ffn(args)
    elif args.gat_scale == 0:
        model.create_fpn(args)
        model.create_ffn(args)
    else:
        model.create_gat(args)
        model.create_fpn(args)
        model.create_scale(args)
        model.create_ffn(args)
    
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
    
    return model