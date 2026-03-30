


import random                              # 随机工具（这份代码里基本没直接用）
                                           
import numpy as np                         # 数值计算库
import scipy.sparse as sp                  # 稀疏矩阵库（图邻接矩阵会很稀疏）
from copy import deepcopy                  # 深拷贝（BGRL会用到）
import math                                # 数学库
from itertools import product              # 笛卡尔积工具（本文件里未直接使用）

import torch                               # PyTorch 主库
import torch.nn as nn                      # 神经网络模块
import torch.nn.functional as F            # 常用函数（normalize、激活等）
from torch.nn import BatchNorm1d           # BN层（本文件里用 torch.nn.BatchNorm1d 形式更多）

import GCL.losses as L                     # 图对比学习损失函数
import GCL.augmentors as A                 # 图数据增强器
from GCL.models import DualBranchContrast, WithinEmbedContrast, BootstrapContrast  # 对比学习模型头

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender  # 跨域推荐基类
from recbole.model.init import xavier_normal_initialization, xavier_uniform_initialization  # 参数初始化
from recbole.model.loss import BPRLoss, EmbLoss   # BPR排序损失 + embedding正则损失
from recbole.utils import InputType               # 输入类型定义
from torch_sparse import matmul                   # 稀疏矩阵乘（本文件里没直接用）
from torch_geometric.nn import LGConv             # LightGCN的卷积层


class LightGCN(torch.nn.Module):                          # 轻量图协同过滤编码器
    def __init__(self, num_layers=2, hidden_dim=64, alpha=None):
        super(LightGCN, self).__init__()                  # 调用父类初始化
        if alpha is None:                                 # 如果没给层权重
            alpha = 1. / (num_layers + 1)                # 默认每层平均权重

        if isinstance(alpha, torch.Tensor):               # 如果alpha本身是tensor
            assert alpha.size(0) == num_layers + 1        # 长度必须=层数+1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))  # 标量扩成每层同权重
        self.register_buffer('alpha', alpha)              # 注册成buffer（随模型走但不训练）
        
        self.layers = torch.nn.ModuleList()               # 卷积层列表
        for _ in range(num_layers):                       # 按层数创建LGConv
            self.layers.append(LGConv(normalize=True))    # normalize=True 表示内部做规范化传播

    def forward(self, x, edge_index, edge_weight=None):   # 前向：输入节点向量和图边
        z = x                                             # 当前表示，初始=输入
        out = z * self.alpha[0]                           # 准备做多层加权和（第0层）
        for i, conv in enumerate(self.layers):            # 遍历每层图卷积
            z = conv(z, edge_index, edge_weight)          # 传播：聚合邻居信息更新表示
            out = out + z * self.alpha[i + 1]             # 累加到多层融合结果
        return z                        



class HAGO(CrossDomainRecommender):                       # 主模型：跨域推荐 + 图对比学习

    input_type = InputType.PAIRWISE                       # 训练输入类型：成对(正/负样本)

    def __init__(self, config, dataset):
        super(HAGO, self).__init__(config, dataset)       # 先初始化父类（拿到很多字段）

        # load dataset info
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field   # source标签字段名
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field   # target标签字段名
        self.dataset = dataset                                           # 保存dataset引用

        # load parameters info
        self.device = config['device']                    # 设备：cpu/cuda

        # load parameters info
        self.latent_dim = config['embedding_size']        # embedding维度
        self.n_layers = config['n_layers']                # GNN层数
        self.reg_weight = config['reg_weight']            # 正则损失系数
        self.domain_lambda_source = config['lambda_source']   # 域迁移系数（source）
        self.domain_lambda_target = config['lambda_target']   # 域迁移系数（target）
        self.drop_rate = config['drop_rate']              # dropout概率
        self.connect_way = config['connect_way']          # 连接方式配置（本文件里未深用）
        self.pretrain_method = config['pretrain_method']  # 预训练方法: grace/bgrl/costa
        self.num_coods = config['num_coods']              # 每组coordinator节点数
        self.reconstruct = config["reconstruct"]          # 配置项（本文件中未直接用）
        self.pe = config["pe"]                            # EdgeRemoving概率
        self.pf = config["pf"]                            # FeatureMasking概率
        self.tau = config["tau"]                          # InfoNCE温度参数
        self.long_short_aware = config['long_short_aware'] if 'long_short_aware' in config else False  # 是否启用长短期感知
        self.short_window = max(1, int(config['short_window'] if 'short_window' in config else 5))      # 短期窗口长度
        self.lambda_orig = config['lambda_orig'] if 'lambda_orig' in config else 1.0     # 原始用户表示权重
        self.lambda_long = config['lambda_long'] if 'lambda_long' in config else 0.5     # 长期兴趣权重
        self.lambda_short = config['lambda_short'] if 'lambda_short' in config else 0.5   # 短期兴趣权重
        
        # # define layers and loss
        
        self.num_graphs = len(dataset.source_domain_dataset.inter_feat["source_item_type"].unique())+1
        # 子图数量 = source里item_type的种类数 + 1（+1通常给target域）

        self.total_num_coods = self.num_graphs*self.num_coods*2
        # cood总数 = 子图数 * 每图cood数 * 2（代码设计成两组cood）

        self.coord_embedding = torch.nn.Embedding(
            num_embeddings=self.total_num_coods,
            embedding_dim=self.latent_dim
        ).to(self.device)
        # coordinator节点向量表

        self.source_user_embedding = torch.nn.Embedding(
            num_embeddings=self.total_num_users,
            embedding_dim=self.latent_dim
        ).to(self.device)
        # source用户embedding

        self.target_user_embedding = torch.nn.Embedding(
            num_embeddings=self.total_num_users,
            embedding_dim=self.latent_dim
        ).to(self.device)
        # target用户embedding（注意数量仍按total）

        self.source_item_embedding = torch.nn.Embedding(
            num_embeddings=self.total_num_items,
            embedding_dim=self.latent_dim
        ).to(self.device)
        # source物品embedding

        self.target_item_embedding = torch.nn.Embedding(
            num_embeddings=self.total_num_items,
            embedding_dim=self.latent_dim
        ).to(self.device)
        # target物品embedding
                self.gnn = LightGCN(self.n_layers, self.latent_dim)  # 图编码器

        self.dropout = nn.Dropout(p=self.drop_rate)          # dropout层
        
        self.loss = BPRLoss()                                # 排序损失
        self.reg_loss = EmbLoss()                            # embedding正则损失
        self.project = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.ELU(),
            torch.nn.Linear(self.latent_dim, self.latent_dim)
        )
        # 对比学习投影头（把z映射到对比空间）
        
        
       # generate intermediate data
        self.source_interaction_matrix = dataset.inter_matrix(
            form='coo', value_field=None, domain='source'
        ).astype(np.float32)
        # source交互稀疏矩阵（COO格式）

        self.target_interaction_matrix = dataset.inter_matrix(
            form='coo', value_field=None, domain='target'
        ).astype(np.float32)
        # target交互稀疏矩阵（COO格式）

        self.source_norm_adj_matrix = self.get_norm_adj_mat(
            self.source_interaction_matrix, self.total_num_users, self.total_num_items
        ).to(self.device)
        # source普通邻接图（torch稀疏张量）

        self.target_norm_adj_matrix = self.get_norm_adj_mat(
            self.target_interaction_matrix, self.total_num_users,
            self.total_num_items, domain="target"
        ).to(self.device)
        # target普通邻接图（torch稀疏张量）

        self.build_user_history_cache()  # 建用户全历史/短历史缓存（长短期兴趣用）

        self.source_norm_adj_matrix_with_cood = self.get_norm_adj_mat_with_cood(
            self.source_interaction_matrix+self.target_interaction_matrix,
            self.total_num_users, self.total_num_items
        ).to(self.device)
        # source增强图：把cood节点和边也加进来
        
        
        self.source_user_degree_count = torch.from_numpy(
            self.source_interaction_matrix.sum(axis=1)
        ).to(self.device)
        # source每个用户的度（交互数）

        self.target_user_degree_count = torch.from_numpy(
            self.target_interaction_matrix.sum(axis=1)
        ).to(self.device)
        # target每个用户的度

        self.source_item_degree_count = torch.from_numpy(
            self.source_interaction_matrix.sum(axis=0)
        ).transpose(0, 1).to(self.device)
        # source每个物品的度

        self.target_item_degree_count = torch.from_numpy(
            self.target_interaction_matrix.sum(axis=0)
        ).transpose(0, 1).to(self.device)
        # target每个物品的度

        # storage variables for full sort evaluation acceleration
        self.target_restore_user_e = None  # 评估缓存：用户表示
        self.target_restore_item_e = None  # 评估缓存：物品表示


        
        if self.pretrain_method == "grace":  # 若预训练方法=GRACE
            
            self.aug1 = A.Compose([A.EdgeRemoving(pe=self.pe), A.FeatureMasking(pf=self.pf)])
            # 增强器1：随机删边 + 随机mask特征

            self.aug2 = A.Compose([A.EdgeRemoving(pe=self.pe), A.FeatureMasking(pf=self.pf)])
            # 增强器2：同上（独立随机）

             self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=self.tau), mode='L2L', intraview_negs=True).to(self.device)
            # GRACE 对比学习头：InfoNCE损失，L2L=节点对节点，对比时包含视图内负样本

        elif self.pretrain_method == "bgrl":
            # 如果预训练方法选 BGRL，就走这个分支

            self.online_encoder = self.gnn
            # 在线编码器：直接用当前的图编码器 gnn

            self.target_encoder = None
            # 目标编码器先不建，后续按需 deepcopy 一份 online_encoder

            self.projection_head = torch.nn.Sequential(
                        torch.nn.Linear(self.latent_dim, self.latent_dim),
                        torch.nn.BatchNorm1d(self.latent_dim),
                        torch.nn.PReLU(),
                        torch.nn.Dropout(self.drop_rate))
            # 投影头：线性 -> BN -> PReLU -> Dropout
            # 作用：把编码结果映射到对比空间，增强训练稳定性

            self.predictor = torch.nn.Sequential(
                        torch.nn.Linear(self.latent_dim, self.latent_dim),
                        torch.nn.BatchNorm1d(self.latent_dim),
                        torch.nn.PReLU(),
                        torch.nn.Dropout(self.drop_rate))
            # 预测头：BGRL里 online 分支会预测 target 分支的表示

            self.batch_norm = torch.nn.BatchNorm1d(self.latent_dim)
            # 额外BN层：对节点表示再做一次规范化

            self.aug1 = A.Compose([A.EdgeRemoving(pe=self.pe), A.FeatureMasking(pf=self.pf)])
            self.aug2 = A.Compose([A.EdgeRemoving(pe=self.pe), A.FeatureMasking(pf=self.pf)])
            # BGRL 也使用两路图增强（删边+特征mask）

            self.contrast_model = BootstrapContrast(loss=L.BootstrapLatent(), mode='L2L').to(self.device)
            # BGRL 的目标：bootstrap latent，不强调显式负样本

        elif self.pretrain_method == "costa":
            # 如果预训练方法选 COSTA

            self.aug1 = A.Compose([A.EdgeRemoving(pe=self.pe), A.FeatureMasking(pf=self.pf)])
            self.aug2 = A.Compose([A.EdgeRemoving(pe=self.pe), A.FeatureMasking(pf=self.pf)])
            # COSTA 也做两路增强

            self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=self.tau), mode='L2L', intraview_negs=False).to(self.device)
            # COSTA 这里同样是 InfoNCE，但 intraview_negs=False（不使用视图内负样本）

        # parameters initialization
        self.apply(xavier_normal_initialization)
        # 对模型参数做 Xavier 正态初始化（常见初始化方式）

        with torch.no_grad():
            self.target_user_embedding.weight[self.target_num_users:].fill_(0)
            self.target_item_embedding.weight[self.target_num_items:].fill_(0)
        # 不需要梯度地把“target域有效范围外”的embedding置0
        # 含义：target域只允许真实存在的用户/物品有可用向量

        self.other_parameter_name = ['target_restore_user_e', 'target_restore_item_e']
        # 指定额外参数名（主要给评估缓存）

    
    def get_norm_adj_mat(self, interaction_matrix, n_users=None, n_items=None, domain="source"):
        # 函数：把交互矩阵转成图邻接稀疏张量（用户+物品二部图）

        # build adj matrix
        if n_users == None or n_items == None:
            n_users, n_items = interaction_matrix.shape
        # 若没传用户/物品数，就从矩阵shape推断

        A = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
        # 建立一个 (U+I)×(U+I) 的稀疏邻接矩阵，DOK格式便于赋值

        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        # 交互矩阵和它的转置（用于双向边）

        data_dict = dict(zip(zip(inter_M.row, inter_M.col + n_users), [1] * inter_M.nnz))
        # user -> item 边：
        # 行是用户id，列是(物品id + n_users)（把物品节点编号放到后半区）

        data_dict.update(dict(zip(zip(inter_M_t.row + n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        # item -> user 反向边（保证图是双向）

        A._update(data_dict)
        # 把所有边更新进邻接矩阵 A

        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # 每个节点的度（有多少邻居）

        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        # 这里算了 D^{-1/2}（通常用于规范化 A）
        # 但注意：下面没有真正做 D*A*D

        L = A 
        # 这里直接 L=A，说明当前实现返回的是未乘D的邻接

        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        # 把 scipy COO 的坐标和值拆出来

        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape))
        # 组装成 PyTorch 稀疏 COO 张量

        return SparseL
        # 返回稀疏邻接图

    def build_user_history_cache(self):
        # 函数：构建用户历史缓存（用于 long/short 兴趣）

        # [Long-short aware connection]
        source_inter_feat = self.dataset.source_domain_dataset.inter_feat
        user_field = self.dataset.source_domain_dataset.uid_field
        item_field = self.dataset.source_domain_dataset.iid_field
        time_field = self.dataset.source_domain_dataset.time_field
        # 拿到 source 域交互表，以及用户/物品/时间字段名

        self.user_history_cache = {}
        self.user_recent_history_cache = {}
        # 两个缓存字典：
        # 全历史缓存、最近窗口历史缓存

        user_ids = source_inter_feat[user_field].numpy()
        item_ids = source_inter_feat[item_field].numpy()
        # 从交互表取出用户序列和物品序列（numpy）

        # [Long-short aware connection]
        # Prefer timestamp ordering when the source interaction feature keeps a timestamp field.
        # Otherwise, fall back to the current interaction row order as an approximation of recency.
        if time_field and time_field in source_inter_feat.columns:
            order = np.argsort(source_inter_feat[time_field].numpy(), kind='stable')
        else:
            order = np.arange(len(source_inter_feat))
        # 如果有时间列：按时间稳定排序（早到晚）
        # 没时间列：按原始行顺序当作“近似时间”

        ordered_user_ids = user_ids[order]
        ordered_item_ids = item_ids[order]
        # 按时间顺序重排用户和物品

        for user_id, item_id in zip(ordered_user_ids, ordered_item_ids):
            history = self.user_history_cache.setdefault(int(user_id), [])
            history.append(int(item_id))
        # 逐条交互累积进“全历史”缓存

        for user_id, history in self.user_history_cache.items():
            self.user_recent_history_cache[user_id] = history[-self.short_window:]
        # 从全历史里切最后 short_window 条，得到“短期历史”
        

     def _mean_history_item_embedding(self, user_ids, history_cache):
        # 函数：给一批用户，按其历史物品向量取平均，得到用户“历史兴趣向量”

        user_ids = user_ids.to(self.device)
        # 把user_ids放到同一设备

        item_weight = self.source_item_embedding.weight
        # source物品embedding矩阵（可训练）

        user_weight = self.source_user_embedding.weight.detach()
        # source用户embedding矩阵（detach防梯度传到这里）

        history_embeddings = []
        # 存每个用户最终的历史向量

        for user_id in user_ids.detach().cpu().tolist():
            item_history = history_cache.get(int(user_id), [])
            # 拿到该用户历史item列表；没有就空列表

            if item_history:
                item_tensor = torch.tensor(item_history, dtype=torch.long, device=self.device)
                history_embeddings.append(item_weight[item_tensor].mean(dim=0))
                # 有历史：取对应item embedding后按行均值
            else:
                history_embeddings.append(user_weight[user_id])
                # 无历史：回退用该用户原始embedding

        return torch.stack(history_embeddings, dim=0)
        # 拼成 [batch_size, latent_dim]

   def compute_long_term_user_embedding(self, user_ids):
        # [Long-short aware connection]
        return self._mean_history_item_embedding(user_ids, self.user_history_cache)
        # 长期兴趣 = 基于“全历史缓存”计算的平均item向量

    def compute_short_term_user_embedding(self, user_ids):
        # [Long-short aware connection]
        return self._mean_history_item_embedding(user_ids, self.user_recent_history_cache)
        # 短期兴趣 = 基于“最近窗口缓存”计算的平均item向量

    def compute_long_short_user_coord_scores(self, user_ids, coord_emb_slice):
        # 函数：计算 用户 -> 当前cood切片 的连接分数（可融合长短期）

        # [Long-short aware connection]
        original_score = F.normalize(self.source_user_embedding(user_ids).detach(), dim=1) @ F.normalize(coord_emb_slice, dim=1).T
        # 原始分数：source用户embedding 与 cood embedding 的归一化点积（余弦相似度）

        if not self.long_short_aware:
            return original_score
        # 如果没开长短期模式，就直接用原始分数

        long_term_user_embedding = self.compute_long_term_user_embedding(user_ids)
        short_term_user_embedding = self.compute_short_term_user_embedding(user_ids)
        # 计算长期/短期用户表示

        long_term_score = F.normalize(long_term_user_embedding, dim=1) @ F.normalize(coord_emb_slice, dim=1).T
        short_term_score = F.normalize(short_term_user_embedding, dim=1) @ F.normalize(coord_emb_slice, dim=1).T
        # 长期分数、短期分数（都是与cood切片做余弦相似度）

        return (
            self.lambda_orig * original_score
            + self.lambda_long * long_term_score
            + self.lambda_short * short_term_score
        )
        # 最终分数 = 原始 + 长期 + 短期 的加权和

    def get_norm_adj_mat_with_cood(self, interaction_matrix, n_users=None, n_items=None):
        # 函数：构建“带 coordinator 节点”的增强邻接图

        interaction_matrix = interaction_matrix.tocoo()
        # 把输入交互矩阵强制转成 COO，便于取 row/col 坐标

        # build adj matrix
        if n_users == None or n_items == None:
            n_users, n_items = interaction_matrix.shape
        # 如果没给用户/物品数量，就从矩阵维度推断

        n_coods = self.total_num_coods
        # coordinator 总节点数（初始化时已算好）

        n_nodes = n_users + n_items+ n_coods
        # 全图总节点数 = 用户 + 物品 + cood
        

        A = sp.dok_matrix((n_users + n_items + n_coods, n_users + n_items + n_coods), dtype=np.float32)
        # 新建全图邻接矩阵（DOK格式，适合增量写边）

        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        # 原交互矩阵和转置矩阵

        data_dict = dict(zip(zip(inter_M.row, inter_M.col + n_users), [1] * inter_M.nnz))
        # 先写 user -> item 边，item 编号偏移 n_users（放在节点后半段）

        data_dict.update(dict(zip(zip(inter_M_t.row + n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        # 再写 item -> user 边，形成双向边
            
            
        A._update(data_dict)
        # 把 user-item 双向边写入邻接矩阵

        A[-n_coods:-n_coods//2, -n_coods//2:] = 1
        A[-n_coods//2:, -n_coods:-n_coods//2] = 1
        # 在 cood 节点内部加两块互连（两组 cood 之间全连接样式）

        self.A = A.tocsr()
        # 缓存一份 CSR 格式（后续若要高效算子可用）

       A = sp.coo_matrix(A)
        row = A.row
        col = A.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(A.data)
        # 把 scipy COO 转成 PyTorch 稀疏坐标格式需要的 index/value

        self.SparseL = torch.sparse_coo_tensor(i, data, torch.Size(A.shape)).to(self.device)
        # 构建基础稀疏邻接张量（包含 user-item 与 cood内部边）
        
        for i in range(1, self.num_graphs+1):
            # 遍历每个子图（1...num_graphs）

            if i < self.num_graphs:
                self.users_domain_list.append(
                    self.dataset.source_domain_dataset.inter_feat["source_user_id"][
                        self.dataset.source_domain_dataset.inter_feat["source_item_type"]==i
                    ].unique().to(self.device)
                )
                # 对 source 的第 i 类 item_type，收集出现过的用户集合

                self.items_domain_list.append(
                    self.dataset.source_domain_dataset.inter_feat["source_item_id"][
                        self.dataset.source_domain_dataset.inter_feat["source_item_type"]==i
                    ].unique().to(self.device)
                )
                # 对 source 的第 i 类 item_type，收集出现过的物品集合

            else:
                self.users_domain_list.append(
                    self.dataset.target_domain_dataset.inter_feat["target_user_id"].unique().to(self.device)
                )
                self.items_domain_list.append(
                    self.dataset.target_domain_dataset.inter_feat["target_item_id"].unique().to(self.device)
                )
                # 最后一个“图”用 target 域全部用户和物品
            

            self.index_list.append(
                torch.cartesian_prod(
                    self.users_domain_list[i-1],
                    torch.arange(
                        n_nodes-(2*self.num_graphs-i)*self.num_coods-self.num_coods,
                        n_nodes - (2*self.num_graphs-i)*self.num_coods,
                        device=self.device
                    )
                ).T
            )
            # 关键1：生成“用户 <-> cood”边坐标
            # cartesian_prod(A, B) = A里每个用户 和 B里每个cood 两两配对
            # .T 后形状是 [2, E]，符合稀疏张量 index 规范

            coord_emb_slice = self.coord_embedding.weight[(i-1)*self.num_coods:i*self.num_coods]
            # 取当前子图对应的一小段 cood embedding

            self.value_list.append(
                self.compute_long_short_user_coord_scores(self.users_domain_list[i-1], coord_emb_slice).flatten()
            )
            # 给上面的“用户<->cood”每条边计算边权（可融合原始/长期/短期）
            self.index_list.append(
                torch.cartesian_prod(
                    self.items_domain_list[i-1],
                    torch.arange(
                        n_nodes-(self.num_graphs-i)*self.num_coods-self.num_coods,
                        n_nodes - (self.num_graphs-i)*self.num_coods,
                        device=self.device
                    )
                ).T
            )
            # 关键2：生成“物品 <-> cood”边坐标（同理，笛卡尔积）

            self.value_list.append(
                (
                    F.normalize(self.source_item_embedding(self.items_domain_list[i-1]).detach())
                    @
                    F.normalize(self.coord_embedding.weight).T[
                        :, self.num_graphs*self.num_coods+(i-1)*self.num_coods:
                           self.num_graphs*self.num_coods+i*self.num_coods
                    ]
                ).flatten()
            )
            # 给“物品<->cood”边计算权重：归一化点积（余弦相似度）

        index_list = torch.cat(self.index_list, dim=1)
        value_list = torch.cat(self.value_list)
        # 把每个子图产生的边索引/边权拼接成一个总集合

        coord_matrix = torch.sparse_coo_tensor(
            index_list,
            torch.where(value_list>0, value_list, 0),
            self.SparseL.shape
        )
        # 构建 cood 边稀疏矩阵，并把负权截断为0（只保留正相关）
            
        return self.SparseL+coord_matrix+coord_matrix.T
        # 返回最终增强图：基础图 + cood边 + 其转置（对称化）
    
        def update_adj_mat_with_cood(self, interaction_matrix, n_users=None, n_items=None):
        # 函数：训练中更新 cood 边权（索引通常复用，重新算值）

        interaction_matrix = interaction_matrix.tocoo()
        # build adj matrix
        if n_users == None or n_items == None:
            n_users, n_items = interaction_matrix.shape
        # 兼容参数（这里要是保持接口一致）

        n_coods = self.total_num_coods
        n_nodes = n_users + n_items+ n_coods
        # 这些变量与上面函数同义

        self.value_list = []
        # 清空旧边权，准备重算

        for i in range(1, self.num_graphs+1):

            coord_emb_slice = self.coord_embedding.weight[(i-1)*self.num_coods:i*self.num_coods]
            # 当前子图对应cood向量切片

            self.value_list.append(
                self.compute_long_short_user_coord_scores(self.users_domain_list[i-1], coord_emb_slice).flatten()
            )
            # 重算“用户<->cood”边权

            self.value_list.append(
                (
                    F.normalize(self.source_item_embedding(self.items_domain_list[i-1]).detach())
                    @ F.normalize(self.coord_embedding.weight).T[:, (i-1)*self.num_coods:i*self.num_coods]
                ).flatten()
            )
            # 重算“物品<->cood”边权

        index_list = torch.cat(self.index_list, dim=1)
        value_list = torch.cat(self.value_list)
        coord_matrix = torch.sparse_coo_tensor(index_list, torch.where(value_list>0, value_list, 0), self.SparseL.shape)
        # 复用已有 index_list，仅替换 value_list，生成新cood边矩阵

        return self.SparseL+coord_matrix+coord_matrix.T
        # 返回更新后的增强图
    
    @staticmethod
    def get_norm_sub_adj_mat(self, A):

        diag = (A > 0).sum(axis=2) + 1e-7
        # 计算每个节点度（避免除零加微小值）

        # add epsilon to avoid divide by zero Warning
        diag = np.power(diag, -0.5)
        D = torch.diag_embed(diag).float()
        # 构造 D^{-1/2} 对角张量

        L = D @ A @ D
        # 规范化邻接：D^{-1/2} A D^{-1/2}

        # covert norm_adj matrix to tensor
        return L
        # 返回规范化子图
    
    def get_ego_embeddings(self, domain='source'):
        if domain == 'source':
            user_embeddings = self.source_user_embedding.weight
            item_embeddings = self.source_item_embedding.weight
            cood_embeddings = self.coord_embedding.weight
            norm_adj_matrix = self.source_norm_adj_matrix_with_cood
            ego_embeddings = torch.cat([user_embeddings, item_embeddings, cood_embeddings], dim=0)
            # source域：拼接 用户+物品+cood 作为图输入

        else:
            user_embeddings = self.target_user_embedding.weight+self.source_user_embedding.weight
            item_embeddings = self.target_item_embedding.weight+self.source_item_embedding.weight
            # target域：用 target embedding + source embedding 的叠加（迁移信息）

            norm_adj_matrix = self.target_norm_adj_matrix 
            ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
            # target域没有cood节点，输入只含 用户+物品

        return ego_embeddings, norm_adj_matrix
        # 返回初始节点向量和对应邻接图
    
    def set_phase(self, phase):
        self.phase = phase
    # 设置训练阶段（SOURCE 或 TARGET）
        
    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = deepcopy(self.online_encoder)
            # 首次调用时，复制一份online encoder当target encoder

            for p in self.target_encoder.parameters():
                p.requires_grad = False
            # target encoder参数不参与梯度更新（只做动量同步）

        return self.target_encoder
        # 返回target encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p
    # 动量更新：target = m*target + (1-m)*online
    # m 通常接近1（这里训练时是0.99）
            
    def forward(self, domain='target'):
        
        all_embeddings, norm_adj_matrix = self.get_ego_embeddings(domain)
        # 先拿对应域的输入节点向量 + 邻接图

        lightgcn_all_embeddings = self.gnn(
            all_embeddings,
            norm_adj_matrix.coalesce().indices(),
            norm_adj_matrix.coalesce().values()
        )
        # 用图编码器传播，得到所有节点新表示

        if domain == "source":
            user_all_embeddings, item_all_embeddings, cood_all_embeddings = torch.split(
                lightgcn_all_embeddings,
                [self.total_num_users, self.total_num_items, self.total_num_coods]
            )
            return user_all_embeddings, item_all_embeddings, cood_all_embeddings
            # source域输出三块：用户、物品、cood

        elif domain == "target":

            user_all_embeddings, item_all_embeddings = torch.split(
                lightgcn_all_embeddings,
                [self.total_num_users, self.total_num_items]
            )

            return user_all_embeddings, item_all_embeddings
            # target域输出两块：用户、物品
    def calculate_loss(self, interaction=None):
        # 训练时调用：根据当前 phase（SOURCE/TARGET）计算损失

        self.init_restore_e()
        # 每次训练前先清理评估缓存，避免用旧表示
        
        if self.pretrain_method == "grace":
                
                all_embeddings, norm_adj_matrix = self.get_ego_embeddings(domain="source")
                # 取 source 域输入节点向量和图

                x1, edge_index1, edge_weight1 = self.aug1(
                    all_embeddings, norm_adj_matrix.coalesce().indices(), norm_adj_matrix.coalesce().values()
                )
                x2, edge_index2, edge_weight2 = self.aug2(
                    all_embeddings, norm_adj_matrix.coalesce().indices(), norm_adj_matrix.coalesce().values()
                )
                # 做两次随机图增强，得到两个视图

                z1 = self.gnn(x1, edge_index1, edge_weight1)[interaction[0]]
                z2 = self.gnn(x2, edge_index2, edge_weight2)[interaction[0]]
                # 两个视图分别编码，并取 interaction[0] 指定的节点表示

                h1, h2 = [self.project(x) for x in [z1, z2]]
                # 过投影头到对比空间

                loss = self.contrast_model(h1, h2)
                # 用 InfoNCE 对比损失
                
            elif self.pretrain_method == "bgrl":
                all_embeddings, norm_adj_matrix = self.get_ego_embeddings(domain="source")
                # 取 source 输入

                x1, edge_index1, edge_weight1 = self.aug1(...)
                x2, edge_index2, edge_weight2 = self.aug2(...)
                # 双视图增强

                self.update_target_encoder(0.99)
                # 先动量更新 target encoder（m=0.99）

                h1 = self.online_encoder(x1, edge_index1, edge_weight1)[interaction[0]]
                h2 = self.online_encoder(x2, edge_index2, edge_weight2)[interaction[0]]
                # online encoder 编码

                h1 = self.batch_norm(h1)
                h2 = self.batch_norm(h2)
                # BN 标准化

                h1_online = self.projection_head(h1)
                h2_online = self.projection_head(h2)
                # 投影

                h1_pred = self.predictor(h1_online)
                h2_pred = self.predictor(h2_online)
                # online 分支预测 target 分支表示

                with torch.no_grad():
                    h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight1)[interaction[0]]
                    h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)[interaction[0]]
                    # target encoder 编码（无梯度）

                    h1_target = self.projection_head(self.batch_norm(h1_target))
                    h2_target = self.projection_head(self.batch_norm(h2_target))
                    # target 分支也做 BN + projection

                loss = 1 - self.contrast_model(
                    h1_pred=h1_pred, h2_pred=h2_pred,
                    h1_target=h1_target.detach(), h2_target=h2_target.detach()
                )
                # bootstrap 目标：希望 online 预测接近 target
            
            elif self.pretrain_method == "costa":
                
                all_embeddings, norm_adj_matrix = self.get_ego_embeddings(domain="source")
                x1, edge_index1, edge_weight1 = self.aug1(...)
                x2, edge_index2, edge_weight2 = self.aug2(...)
                # 同样双增强

                z1 = self.gnn(x1, edge_index1, edge_weight1)[interaction[0]]
                z2 = self.gnn(x2, edge_index2, edge_weight2)[interaction[0]]
                # 编码两个视图

                k = torch.tensor(int(z1.shape[0] * 0.5))
                p = (1/torch.sqrt(k))*torch.randn(k, z1.shape[0]).to(self.device)
                # 构造随机投影矩阵 p（把表示降到约一半维度）

                z1 = p @ z1
                z2 = p @ z2 
                # 随机投影后再对比

                h1, h2 = [self.project(x) for x in [z1, z2]]
                # 再过投影头

                # h1, h2 = z1, z2#[self.project(x) for x in [z1, z2]]
                loss = self.contrast_model(h1, h2)
                # 最终对比损失
                
        elif self.phase == "TARGET":
            target_user_all_embeddings, target_item_all_embeddings = self.forward()
            # 跑 target 域前向，拿全部用户/物品表示

            target_user = interaction[self.TARGET_USER_ID]
            target_pos_item = interaction[self.TARGET_ITEM_ID]
            target_neg_item = interaction[self.TARGET_NEG_ITEM_ID]
            # 从 batch 取用户、正样本物品、负样本物品 id

            target_u_embeddings = target_user_all_embeddings[target_user]
            target_pos_embeddings = target_item_all_embeddings[target_pos_item]
            target_neg_embeddings = target_item_all_embeddings[target_neg_item]
            # 索引出对应 embedding


            # calculate BPR Loss in target domain
            pos_scores = torch.mul(target_u_embeddings, target_pos_embeddings).sum(dim=1)
            neg_scores = torch.mul(target_u_embeddings, target_neg_embeddings).sum(dim=1)
            bpr_loss = self.loss(pos_scores, neg_scores)
            # BPR：希望正样本分数 > 负样本分数

            # calculate Reg Loss in target domain
            u_ego_embeddings = self.target_user_embedding(target_user)
            pos_ego_embeddings = self.target_item_embedding(target_pos_item)
            neg_ego_embeddings = self.target_item_embedding(target_neg_item)
            reg_loss = self.reg_loss(u_ego_embeddings,  pos_ego_embeddings, neg_ego_embeddings)
            # embedding 正则，防止向量无约束变大

            loss = bpr_loss + self.reg_weight * reg_loss
            # 总损失 = 排序损失 + 正则项

        return loss
        # 不管SOURCE还是TARGET，返回对应损失

    def predict(self, interaction):
        result = []
        target_user_all_embeddings, target_item_all_embeddings = self.forward()
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]

        u_embeddings = target_user_all_embeddings[user]
        i_embeddings = target_item_all_embeddings[item]

        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]

        restore_user_e, restore_item_e = self.get_restore_e()
        u_embeddings = restore_user_e[user]
        i_embeddings = restore_item_e[:self.target_num_items]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores.view(-1)

    def init_restore_e(self):
        # clear the storage variable when training
        if self.target_restore_user_e is not None or self.target_restore_item_e is not None:
            self.target_restore_user_e, self.target_restore_item_e = None, None

    def get_restore_e(self):
        if self.target_restore_user_e is None or self.target_restore_item_e is None:
            self.target_restore_user_e, self.target_restore_item_e = self.forward()
        return self.target_restore_user_e, self.target_restore_item_e



