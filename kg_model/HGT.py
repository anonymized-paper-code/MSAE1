import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax

class HGTLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 dropout = 0.5,
                 use_norm = False):
        super(HGTLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.node_dict     = node_dict
        self.edge_dict     = edge_dict
        self.num_types     = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel     = self.num_types * self.num_relations * self.num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None

        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip        = nn.Parameter(torch.ones(self.num_types))
        self.drop        = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v_%d' % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata['t'] = attn_score.unsqueeze(-1)

            G.multi_update_all({etype : (fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't')) \
                                for etype, e_id in edge_dict.items()}, cross_reducer = 'mean')

            new_h = {}
            for ntype in G.ntypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                n_id = node_dict[ntype]
                
                # 如果没有输出的t，则直接使用原本的h
                if 't' in G.nodes[ntype].data:
                    alpha = torch.sigmoid(self.skip[n_id])
                    t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                    trans_out = self.drop(self.a_linears[n_id](t))
                    trans_out = trans_out * alpha + h[ntype] * (1-alpha)
                    if self.use_norm:
                        new_h[ntype] = self.norms[n_id](trans_out)
                    else:
                        new_h[ntype] = trans_out
                else:
                    new_h[ntype] = h[ntype]
            return new_h

class HGT(nn.Module):
    def __init__(self, config):
        # node_dict, edge_dict, n_inp_dict, n_hid, n_out, n_layers, n_heads, use_norm = True
        super(HGT, self).__init__()
        self.node_dict = config['node_dict'] # node_id
        edge_dict = config['edge_dict'] # 
        
        n_inp_dict = config['node_type_to_feature_len_dict']
        n_hid = config['node_hid_len']
        self.class_num = config['class_num']
        self.n_layers = config['layer_num']
        n_heads = config['nhead']
        use_norm = config['use_norm']
        
        self.task_to_node = config['task_to_node']
        self.task_name = config['task_name']
        self.task_type = config['task_type']
        
        self.adapt_ws = nn.ModuleList()
        for tmp_node_type in self.node_dict.keys():
            self.adapt_ws.append(nn.Linear(n_inp_dict[tmp_node_type], n_hid))
        
        self.gcs = nn.ModuleList()
        for _ in range(self.n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, self.node_dict, edge_dict, n_heads, use_norm = use_norm))
        
        self.out = nn.ModuleList()
        for out_dim in self.class_num:
            self.out.append(nn.Linear(n_hid, out_dim))

    def forward(self, G, recorders=None):
        h = {} # dict of all node features
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id](G.nodes[ntype].data['feat']))
        for i in range(self.n_layers):
            h = self.gcs[i](G, h)
            
        # output
        output_all_nodes = {}
        for ind, task in enumerate(self.task_name):
            node_type = self.task_to_node[task]
            output = self.out[ind](h[node_type])
            if self.task_type[task] == 'single-label':
                output = nn.Softmax(1)(output)
            elif self.task_type[task] == 'multi-label':
                output = F.sigmoid(output)
            else:
                raise NameError(f'task type {self.task_type[target_node]} does not exist!')
                
            if self.class_num[ind]<=2:
                output = output[:,0]

            output_all_nodes[task] = output
            
        return output_all_nodes