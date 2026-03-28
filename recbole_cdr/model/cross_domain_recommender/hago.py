


import random

import numpy as np
import scipy.sparse as sp
from copy import deepcopy
import math
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d

import GCL.losses as L
import GCL.augmentors as A
from GCL.models import DualBranchContrast, WithinEmbedContrast, BootstrapContrast

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole.model.init import xavier_normal_initialization, xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from torch_sparse import matmul
from torch_geometric.nn import LGConv


class LightGCN(torch.nn.Module):
    def __init__(self, num_layers=2, hidden_dim=64, alpha=None):
        super(LightGCN, self).__init__()
        if alpha is None:
            alpha = 1. / (num_layers + 1)

        if isinstance(alpha, torch.Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))
        self.register_buffer('alpha', alpha)
        
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(LGConv(normalize=True))

    def forward(self, x, edge_index, edge_weight=None):
        z = x 
        out = z * self.alpha[0]
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            out = out + z * self.alpha[i + 1]
        return z



class HAGO(CrossDomainRecommender):

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(HAGO, self).__init__(config, dataset)

        # load dataset info
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field
        self.dataset = dataset

        # load parameters info
        self.device = config['device']

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.domain_lambda_source = config['lambda_source']  # float32 type: the weight of source embedding in transfer function
        self.domain_lambda_target = config['lambda_target']  # float32 type: the weight of target embedding in transfer function
        self.drop_rate = config['drop_rate']  # float32 type: the dropout rate
        self.connect_way = config['connect_way']  # str type: the connect way for all layers
        self.pretrain_method = config['pretrain_method'] #graphcl/simgrace
        self.num_coods = config['num_coods'] # the number of graph coordinators
        self.reconstruct = config["reconstruct"]
        self.pe = config["pe"] # pe
        self.pf = config["pf"] # pf
        self.tau = config["tau"] # tau
        self.time_aware = config["time_aware"]
        self.time_decay_gamma = config["time_decay_gamma"]

        
        # define layers and loss
        
        self.num_graphs = len(dataset.source_domain_dataset.inter_feat["source_item_type"].unique())+1
        
        self.total_num_coods = self.num_graphs*self.num_coods*2
        self.coord_embedding = torch.nn.Embedding(num_embeddings=self.total_num_coods, embedding_dim=self.latent_dim).to(self.device)
        
        

        self.source_user_embedding = torch.nn.Embedding(num_embeddings=self.total_num_users, embedding_dim=self.latent_dim).to(self.device)
        
        self.target_user_embedding = torch.nn.Embedding(num_embeddings=self.total_num_users, embedding_dim=self.latent_dim).to(self.device)
        
        self.source_item_embedding = torch.nn.Embedding(num_embeddings=self.total_num_items, embedding_dim=self.latent_dim).to(self.device)
        
        self.target_item_embedding = torch.nn.Embedding(num_embeddings=self.total_num_items, embedding_dim=self.latent_dim).to(self.device)
        

        self.gnn = LightGCN(self.n_layers, self.latent_dim)

        self.dropout = nn.Dropout(p=self.drop_rate)
        
        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.project = torch.nn.Sequential(torch.nn.Linear(self.latent_dim, self.latent_dim), torch.nn.ELU(), torch.nn.Linear(self.latent_dim, self.latent_dim))
        
        
        # generate intermediate data
        self.source_interaction_matrix = self._get_time_aware_interaction_matrix(domain='source')
        self.target_interaction_matrix = self._get_time_aware_interaction_matrix(domain='target')

        self.source_norm_adj_matrix = self.get_norm_adj_mat(self.source_interaction_matrix, self.total_num_users,
                                                       self.total_num_items).to(self.device)
        self.target_norm_adj_matrix = self.get_norm_adj_mat(self.target_interaction_matrix, self.total_num_users,
                                                       self.total_num_items, domain="target").to(self.device)
        
        self.source_norm_adj_matrix_with_cood = self.get_norm_adj_mat_with_cood(self.source_interaction_matrix+self.target_interaction_matrix, self.total_num_users, self.total_num_items).to(self.device)
        
        
        self.source_user_degree_count = torch.from_numpy(self.source_interaction_matrix.sum(axis=1)).to(self.device)
        self.target_user_degree_count = torch.from_numpy(self.target_interaction_matrix.sum(axis=1)).to(self.device)
        self.source_item_degree_count = torch.from_numpy(self.source_interaction_matrix.sum(axis=0)).transpose(0, 1).to(self.device)
        self.target_item_degree_count = torch.from_numpy(self.target_interaction_matrix.sum(axis=0)).transpose(0, 1).to(self.device)

        # storage variables for full sort evaluation acceleration
        self.target_restore_user_e = None
        self.target_restore_item_e = None
        
        if self.pretrain_method == "grace":
            
            self.aug1 = A.Compose([A.EdgeRemoving(pe=self.pe), A.FeatureMasking(pf=self.pf)])
            self.aug2 = A.Compose([A.EdgeRemoving(pe=self.pe), A.FeatureMasking(pf=self.pf)])
            self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=self.tau), mode='L2L', intraview_negs=True).to(self.device)
        elif self.pretrain_method == "bgrl":
            self.online_encoder = self.gnn
            self.target_encoder = None
            self.projection_head = torch.nn.Sequential(
                        torch.nn.Linear(self.latent_dim, self.latent_dim),
                        torch.nn.BatchNorm1d(self.latent_dim),
                        torch.nn.PReLU(),
                        torch.nn.Dropout(self.drop_rate))
            self.predictor = torch.nn.Sequential(
                        torch.nn.Linear(self.latent_dim, self.latent_dim),
                        torch.nn.BatchNorm1d(self.latent_dim),
                        torch.nn.PReLU(),
                        torch.nn.Dropout(self.drop_rate))
            
            self.batch_norm = torch.nn.BatchNorm1d(self.latent_dim)
            
            self.aug1 = A.Compose([A.EdgeRemoving(pe=self.pe), A.FeatureMasking(pf=self.pf)])
            self.aug2 = A.Compose([A.EdgeRemoving(pe=self.pe), A.FeatureMasking(pf=self.pf)])
            self.contrast_model = BootstrapContrast(loss=L.BootstrapLatent(), mode='L2L').to(self.device)

        elif self.pretrain_method == "costa":
            self.aug1 = A.Compose([A.EdgeRemoving(pe=self.pe), A.FeatureMasking(pf=self.pf)])
            self.aug2 = A.Compose([A.EdgeRemoving(pe=self.pe), A.FeatureMasking(pf=self.pf)])
            self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=self.tau), mode='L2L', intraview_negs=False).to(self.device)
        

        # parameters initialization
        self.apply(xavier_normal_initialization)
        
        with torch.no_grad():
            self.target_user_embedding.weight[self.target_num_users:].fill_(0)
            self.target_item_embedding.weight[self.target_num_items:].fill_(0)

        self.other_parameter_name = ['target_restore_user_e', 'target_restore_item_e']

    def _resolve_time_field(self, domain='source'):
        """Resolve the timestamp field from the prefixed domain interaction feature."""
        domain_dataset = self.dataset.source_domain_dataset if domain == 'source' else self.dataset.target_domain_dataset
        inter_feat = domain_dataset.inter_feat

        # Preferred field in RecBole dataset object.
        if getattr(domain_dataset, "time_field", None) in inter_feat.columns:
            return domain_dataset.time_field

        # Fallback candidates for custom datasets.
        candidates = [
            f"{domain}_timestamp",
            f"{domain}_time",
            f"{domain}_source_time",
            f"{domain}_target_time",
        ]
        for field in candidates:
            if field in inter_feat.columns:
                return field
        return None

    def _compute_time_decay_weights(self, user_ids, timestamps):
        """Compute w_ui_time = exp(-gamma * (t_u_last - t_ui))."""
        user_last_time = np.full(self.total_num_users, -np.inf, dtype=np.float64)
        np.maximum.at(user_last_time, user_ids, timestamps)
        delta_t = user_last_time[user_ids] - timestamps
        delta_t = np.maximum(delta_t, 0.0)
        return np.exp(-self.time_decay_gamma * delta_t).astype(np.float32)

    def _get_time_aware_interaction_matrix(self, domain='source'):
        """Build interaction matrix with optional time-aware edge weights."""
        if not self.time_aware:
            return self.dataset.inter_matrix(form='coo', value_field=None, domain=domain).astype(np.float32)

        domain_dataset = self.dataset.source_domain_dataset if domain == 'source' else self.dataset.target_domain_dataset
        inter_feat = domain_dataset.inter_feat
        user_field = domain_dataset.uid_field
        item_field = domain_dataset.iid_field
        time_field = self._resolve_time_field(domain=domain)

        if time_field is None:
            self.logger.warning(
                f"[HAGO] time_aware=True but no time field found for {domain} domain. "
                f"Fallback to equal edge weights."
            )
            return self.dataset.inter_matrix(form='coo', value_field=None, domain=domain).astype(np.float32)

        user_ids = inter_feat[user_field].to_numpy(dtype=np.int64)
        item_ids = inter_feat[item_field].to_numpy(dtype=np.int64)
        timestamps = inter_feat[time_field].to_numpy(dtype=np.float64)
        weights = self._compute_time_decay_weights(user_ids, timestamps)

        return sp.coo_matrix(
            (weights, (user_ids, item_ids)),
            shape=(self.total_num_users, self.total_num_items),
            dtype=np.float32
        )

    def get_norm_adj_mat(self, interaction_matrix, n_users=None, n_items=None, domain="source"):
        # build adj matrix
        if n_users == None or n_items == None:
            n_users, n_items = interaction_matrix.shape

        inter_M = interaction_matrix.tocoo()
        A = sp.coo_matrix(
            (inter_M.data, (inter_M.row, inter_M.col + n_users)),
            shape=(n_users + n_items, n_users + n_items),
            dtype=np.float32
        )
        A = A + A.transpose()
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = A
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape))
        return SparseL
    
    def get_norm_adj_mat_with_cood(self, interaction_matrix, n_users=None, n_items=None):
        interaction_matrix = interaction_matrix.tocoo()
        # build adj matrix
        if n_users == None or n_items == None:
            n_users, n_items = interaction_matrix.shape
            
        n_coods = self.total_num_coods
        n_nodes = n_users + n_items+ n_coods
        

        A = sp.dok_matrix((n_users + n_items + n_coods, n_users + n_items + n_coods), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose().tocoo()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + n_users), inter_M.data))
        data_dict.update(dict(zip(zip(inter_M_t.row + n_users, inter_M_t.col), inter_M_t.data)))
            
            
        A._update(data_dict)

        A[-n_coods:-n_coods//2, -n_coods//2:] = 1
        A[-n_coods//2:, -n_coods:-n_coods//2] = 1

        self.A = A.tocsr()

        A = sp.coo_matrix(A)
        row = A.row
        col = A.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(A.data)
        self.SparseL = torch.sparse_coo_tensor(i, data, torch.Size(A.shape)).cuda()
        
        self.users_domain_list = []
        self.items_domain_list = []
        self.index_list = []
        self.value_list = []
        
        for i in range(1, self.num_graphs+1):
            if i < self.num_graphs:
                self.users_domain_list.append(self.dataset.source_domain_dataset.inter_feat["source_user_id"][self.dataset.source_domain_dataset.inter_feat["source_item_type"]==i].unique().cuda())
                self.items_domain_list.append(self.dataset.source_domain_dataset.inter_feat["source_item_id"][self.dataset.source_domain_dataset.inter_feat["source_item_type"]==i].unique().cuda())
            else:
                self.users_domain_list.append(self.dataset.target_domain_dataset.inter_feat["target_user_id"].unique().cuda())
                self.items_domain_list.append(self.dataset.target_domain_dataset.inter_feat["target_item_id"].unique().cuda())
            

            self.index_list.append(torch.cartesian_prod(self.users_domain_list[i-1], torch.arange(n_nodes-(2*self.num_graphs-i)*self.num_coods-self.num_coods, n_nodes - (2*self.num_graphs-i)*self.num_coods).cuda()).T)
            self.value_list.append((F.normalize(self.source_user_embedding(self.users_domain_list[i-1]).detach())@F.normalize(self.coord_embedding.weight).T[:, (i-1)*self.num_coods:i*self.num_coods]).flatten())
            
            self.index_list.append(torch.cartesian_prod(self.items_domain_list[i-1], torch.arange(n_nodes-(self.num_graphs-i)*self.num_coods-self.num_coods, n_nodes - (self.num_graphs-i)*self.num_coods).cuda()).T)
            self.value_list.append((F.normalize(self.source_item_embedding(self.items_domain_list[i-1]).detach())@F.normalize(self.coord_embedding.weight).T[:, self.num_graphs*self.num_coods+(i-1)*self.num_coods:self.num_graphs*self.num_coods+i*self.num_coods]).flatten())

        index_list = torch.cat(self.index_list, dim=1)
        value_list = torch.cat(self.value_list)
        coord_matrix = torch.sparse_coo_tensor(index_list, torch.where(value_list>0, value_list, 0), self.SparseL.shape)
            
        return self.SparseL+coord_matrix+coord_matrix.T
    
    def update_adj_mat_with_cood(self, interaction_matrix, n_users=None, n_items=None):
        interaction_matrix = interaction_matrix.tocoo()
        # build adj matrix
        if n_users == None or n_items == None:
            n_users, n_items = interaction_matrix.shape
            
        n_coods = self.total_num_coods
        n_nodes = n_users + n_items+ n_coods
        self.value_list = []
        for i in range(1, self.num_graphs+1):

            self.value_list.append((F.normalize(self.source_user_embedding(self.users_domain_list[i-1]).detach())@F.normalize(self.coord_embedding.weight).T[:, (i-1)*self.num_coods:i*self.num_coods]).flatten())

            self.value_list.append((F.normalize(self.source_item_embedding(self.items_domain_list[i-1]).detach())@F.normalize(self.coord_embedding.weight).T[:, (i-1)*self.num_coods:i*self.num_coods]).flatten())

        index_list = torch.cat(self.index_list, dim=1)
        value_list = torch.cat(self.value_list)
        coord_matrix = torch.sparse_coo_tensor(index_list, torch.where(value_list>0, value_list, 0), self.SparseL.shape)

        return self.SparseL+coord_matrix+coord_matrix.T
    
    @staticmethod
    def corruption(x, edge_index, edge_weight=None):
        return x[torch.randperm(x.size(0))], edge_index, edge_weight
    def get_norm_sub_adj_mat(self, A):

        diag = (A > 0).sum(axis=2) + 1e-7
        # add epsilon to avoid divide by zero Warning
        diag = np.power(diag, -0.5)
        D = torch.diag_embed(diag).float()

        L = D @ A @ D
        # covert norm_adj matrix to tensor
        return L
    
    def get_ego_embeddings(self, domain='source'):
        if domain == 'source':
            user_embeddings = self.source_user_embedding.weight
            item_embeddings = self.source_item_embedding.weight
            cood_embeddings = self.coord_embedding.weight
            norm_adj_matrix = self.source_norm_adj_matrix_with_cood
            ego_embeddings = torch.cat([user_embeddings, item_embeddings, cood_embeddings], dim=0)

        else:
            user_embeddings = self.target_user_embedding.weight+self.source_user_embedding.weight
            item_embeddings = self.target_item_embedding.weight+self.source_item_embedding.weight

            norm_adj_matrix = self.target_norm_adj_matrix 
            ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)

        return ego_embeddings, norm_adj_matrix
    
    def set_phase(self, phase):
        self.phase = phase
        
    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = deepcopy(self.online_encoder)

            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p
            
    def forward(self, domain='target'):
        
        all_embeddings, norm_adj_matrix = self.get_ego_embeddings(domain)

        lightgcn_all_embeddings = self.gnn(all_embeddings, norm_adj_matrix.coalesce().indices(), norm_adj_matrix.coalesce().values())

        if domain == "source":
            user_all_embeddings, item_all_embeddings, cood_all_embeddings = torch.split(lightgcn_all_embeddings,
                                                                   [self.total_num_users, self.total_num_items, self.total_num_coods])
            return user_all_embeddings, item_all_embeddings, cood_all_embeddings
        elif domain == "target":

            user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings,
                                                                    [self.total_num_users, self.total_num_items])

            return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction=None):

        self.init_restore_e()
        
        if self.phase == "SOURCE":
            
            self.source_norm_adj_matrix_with_cood = self.update_adj_mat_with_cood(self.source_interaction_matrix+self.target_interaction_matrix, self.total_num_users, self.total_num_items).to(self.device)
            if self.pretrain_method == "grace":
                
                all_embeddings, norm_adj_matrix = self.get_ego_embeddings(domain="source")
                
                x1, edge_index1, edge_weight1 = self.aug1(all_embeddings, norm_adj_matrix.coalesce().indices(), norm_adj_matrix.coalesce().values())
                x2, edge_index2, edge_weight2 = self.aug2(all_embeddings, norm_adj_matrix.coalesce().indices(), norm_adj_matrix.coalesce().values())
                
                z1 = self.gnn(x1, edge_index1, edge_weight1)[interaction[0]]
                z2 = self.gnn(x2, edge_index2, edge_weight2)[interaction[0]]

                h1, h2 = [self.project(x) for x in [z1, z2]]
                loss = self.contrast_model(h1, h2)
                
            elif self.pretrain_method == "bgrl":
                all_embeddings, norm_adj_matrix = self.get_ego_embeddings(domain="source")
                
                x1, edge_index1, edge_weight1 = self.aug1(all_embeddings, norm_adj_matrix.coalesce().indices(), norm_adj_matrix.coalesce().values())
                x2, edge_index2, edge_weight2 = self.aug2(all_embeddings, norm_adj_matrix.coalesce().indices(), norm_adj_matrix.coalesce().values())
                
                self.update_target_encoder(0.99)
                
                h1 = self.online_encoder(x1, edge_index1, edge_weight1)[interaction[0]]
                h2 = self.online_encoder(x2, edge_index2, edge_weight2)[interaction[0]]
                
                h1 = self.batch_norm(h1)
                h2 = self.batch_norm(h2)
                
                h1_online = self.projection_head(h1)
                h2_online = self.projection_head(h2)

                h1_pred = self.predictor(h1_online)
                h2_pred = self.predictor(h2_online)

                with torch.no_grad():
                    h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight1)[interaction[0]]
                    h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)[interaction[0]]
                    
                    h1_target = self.projection_head(self.batch_norm(h1_target))
                    h2_target = self.projection_head(self.batch_norm(h2_target))

                loss = 1 - self.contrast_model(h1_pred=h1_pred, h2_pred=h2_pred, h1_target=h1_target.detach(), h2_target=h2_target.detach())
            
            elif self.pretrain_method == "costa":
                
                all_embeddings, norm_adj_matrix = self.get_ego_embeddings(domain="source")
                x1, edge_index1, edge_weight1 = self.aug1(all_embeddings, norm_adj_matrix.coalesce().indices(), norm_adj_matrix.coalesce().values())
                x2, edge_index2, edge_weight2 = self.aug2(all_embeddings, norm_adj_matrix.coalesce().indices(), norm_adj_matrix.coalesce().values())

                z1 = self.gnn(x1, edge_index1, edge_weight1)[interaction[0]]
                z2 = self.gnn(x2, edge_index2, edge_weight2)[interaction[0]]
                
                k = torch.tensor(int(z1.shape[0] * 0.5))
                p = (1/torch.sqrt(k))*torch.randn(k, z1.shape[0]).to(self.device)

                z1 = p @ z1
                z2 = p @ z2 
                h1, h2 = [self.project(x) for x in [z1, z2]]

                # h1, h2 = z1, z2#[self.project(x) for x in [z1, z2]]
                loss = self.contrast_model(h1, h2)
                
        elif self.phase == "TARGET":
            target_user_all_embeddings, target_item_all_embeddings = self.forward()

            target_user = interaction[self.TARGET_USER_ID]
            target_pos_item = interaction[self.TARGET_ITEM_ID]
            target_neg_item = interaction[self.TARGET_NEG_ITEM_ID]

            target_u_embeddings = target_user_all_embeddings[target_user]
            target_pos_embeddings = target_item_all_embeddings[target_pos_item]
            target_neg_embeddings = target_item_all_embeddings[target_neg_item]


            # calculate BPR Loss in target domain
            pos_scores = torch.mul(target_u_embeddings, target_pos_embeddings).sum(dim=1)
            neg_scores = torch.mul(target_u_embeddings, target_neg_embeddings).sum(dim=1)
            bpr_loss = self.loss(pos_scores, neg_scores)

            # calculate Reg Loss in target domain
            u_ego_embeddings = self.target_user_embedding(target_user)
            pos_ego_embeddings = self.target_item_embedding(target_pos_item)
            neg_ego_embeddings = self.target_item_embedding(target_neg_item)
            reg_loss = self.reg_loss(u_ego_embeddings,  pos_ego_embeddings, neg_ego_embeddings)

            loss = bpr_loss + self.reg_weight * reg_loss

        return loss

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


