# -*- coding:utf-8 -*-
"""
Author:
    zanshuxun, zanshuxun@aliyun.com

Reference:
    [1] Tang H, Liu J, Zhao M, et al. Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations[C]//Fourteenth ACM Conference on Recommender Systems. 2020.(https://dl.acm.org/doi/10.1145/3383313.3412236)
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import copy

from IPython.core.debugger import set_trace
from .core import DNN, PredictionLayer

# from .SimpleHGN import SimpleHGN_MTL, SimpleHGN_final


class Our1(nn.Module):
    """Instantiates the multi level of Customized Gate Control of Progressive Layered Extraction architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param shared_expert_num: integer, number of task-shared experts.
    :param specific_expert_num: integer, number of task-specific experts.
    :param num_levels: integer, number of CGC levels.
    :param expert_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of expert DNN.
    :param gate_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of gate DNN.
    :param tower_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of task-specific DNN.
    :param l2_reg_linear: float, L2 regularizer strength applied to linear part.
    :param l2_reg_embedding: float, L2 regularizer strength applied to embedding vector.
    :param l2_reg_dnn: float, L2 regularizer strength applied to DNN.
    :param init_std: float, to use as the initialize std of embedding vector.
    :param seed: integer, to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN.
    :param dnn_use_bn: bool, Whether use BatchNormalization before activation or not in DNN.
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param task_names: list of str, indicating the predict target of each tasks.
    :param device: str, ``"cpu"`` or ``"cuda:0"``.
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.

    :return: A PyTorch model instance.
    """
    def __init__(self, shared_expert_num=1, specific_expert_num=1, num_levels=1,
                 expert_dnn_hidden_units=(256, 128), gate_dnn_hidden_units=(64,), tower_dnn_hidden_units=(64,),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task_types=('binary', 'binary'),
                 task_names=('author', 'paper'), device='cpu', gpus=None, config_dict=None):
        super(Our1, self).__init__()
        self.num_tasks = len(task_names)
        if self.num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1!")
#         if len(task_types) != self.num_tasks:
#             raise ValueError("num_tasks must be equal to the length of task_types")

        self.specific_expert_num = specific_expert_num
        self.shared_expert_num = shared_expert_num
        self.num_levels = num_levels
        self.task_names = task_names
        self.input_dim = config_dict['node_feature_hid_len'] 
        self.no_level0_input_dim = config_dict['GAT_hid_len']* config_dict['nhead'] 
        
        self.expert_dnn_hidden_units = expert_dnn_hidden_units
        self.gate_dnn_hidden_units = gate_dnn_hidden_units
        self.tower_dnn_hidden_units = tower_dnn_hidden_units
        self.edge_type_each_node = config_dict['edge_type_each_node']
        self.disen_layer_num = config_dict['disen_layer_num']
        self.edge_num_node = {} 
        for node in self.edge_type_each_node:
            self.edge_num_node[node] = len(self.edge_type_each_node[node])
         
        self.share_emb_net = True
        
        self.config_dict = config_dict
        
        self.input_dim = [self.input_dim * 2 for i in range(self.disen_layer_num)] + [self.input_dim for i in range(self.num_levels-self.disen_layer_num)] 
        self.no_level0_input_dim = [self.no_level0_input_dim*2 for i in range(self.disen_layer_num)] + [self.no_level0_input_dim for i in range(self.num_levels-self.disen_layer_num)] 
        
        
        if config_dict['try_uncertainty']:
            self.uncertainty_weight = nn.ParameterDict() 
            for task in task_names:
                self.uncertainty_weight[task] = Parameter(torch.tensor(0.5)) 
            
        # 0. embedding net
        self.edge_dict = config_dict['edge_dict']
        self.edge_list = list(self.edge_dict.keys())
        self.node_type_num = len(config_dict['node_dict'])
        node_dict =  config_dict['node_dict']
        node_type_to_feature_len_dict = config_dict['node_type_to_feature_len_dict']
        node_feature_hid_len = config_dict['node_feature_hid_len']
        self.Node_Transform_list = {}
        for tmp_node_type in node_dict:
            tmp_linear = nn.Linear(node_type_to_feature_len_dict[tmp_node_type], node_feature_hid_len)
            self.Node_Transform_list[tmp_node_type] = tmp_linear
            self.add_module('{}_Node_Transform'.format(tmp_node_type), self.Node_Transform_list[tmp_node_type])

        
        def multi_module_list(num_level, num_tasks, expert_num, inputs_dim_level0, inputs_dim_not_level0, hidden_units):
            return nn.ModuleList(
                [nn.ModuleList([nn.ModuleList([DNN(inputs_dim_level0[level_num] if level_num == 0 else inputs_dim_not_level0[level_num],
                                                   hidden_units, activation=dnn_activation,
                                                   l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                                   init_std=init_std, device=device) for _ in
                                               range(expert_num)])
                                for _ in range(num_tasks)]) for level_num in range(num_level)])
        def multi_module_list_han(num_level, num_tasks, expert_num, config_dict, type):
            if config_dict['split_method'] == 'full':
                return nn.ModuleList(
                    [nn.ModuleList([nn.ModuleList([SimpleHGN_MTL(config_dict)  for _ in range(expert_num)])
                                    for _ in range(num_tasks)]) for level_num in range(num_level)])
            elif config_dict['split_method'] == 'layer':
                return nn.ModuleList(
                    [nn.ModuleList([SimpleHGN_MTL_Layer(config_dict, 'share') if type=='share' else SimpleHGN_MTL_Layer(config_dict, self.task_names[i]) 
                                    for _ in range(expert_num)])
                                    for i in range(num_tasks)])
        
        # 1. experts
        # task-specific experts
        self.specific_experts = multi_module_list_han(self.num_levels, self.num_tasks, self.specific_expert_num, config_dict, 'specific')

        # shared experts
        self.shared_experts = multi_module_list_han(self.num_levels, 1, self.shared_expert_num, config_dict, 'share')

        # 2. gates
        if config_dict['mmoe']:
            specific_gate_output_dim = self.num_tasks * self.specific_expert_num + self.shared_expert_num
        shared_gate_output_dim = self.num_tasks * self.specific_expert_num + self.shared_expert_num
        if config_dict['gate_type']=='para':
#             set_trace()
            self.gate_weight_sharing = nn.ParameterList([Parameter(torch.zeros([1, shared_gate_output_dim]).cuda()) for level_num in range(self.num_levels)])
            self.gate_weight_specific = nn.ModuleList([nn.ParameterList([Parameter(torch.zeros([1, specific_gate_output_dim]).cuda()) for _ in range(self.num_tasks)]) for level_num in range(self.num_levels)])
        elif config_dict['gate_type']=='mlp':
            # gates for task-specific experts
            self.specific_gate_dnn = nn.ModuleDict()
            self.specific_gate_dnn_final_layer = nn.ModuleDict()
            for node in node_dict:
                specific_gate_output_dim = [(self.specific_expert_num + self.shared_expert_num) * len(self.edge_type_each_node[node]) for i in range(self.disen_layer_num)]  + [self.specific_expert_num + self.shared_expert_num for i in range(self.num_levels-self.disen_layer_num)]
                if len(gate_dnn_hidden_units) > 0:
                    self.specific_gate_dnn[node] = multi_module_list(self.num_levels, self.num_tasks, 1,
                                                               self.input_dim, self.no_level0_input_dim,
                                                               gate_dnn_hidden_units)
                self.specific_gate_dnn_final_layer[node] = nn.ModuleList(
                [nn.ModuleList([nn.Linear(
                    gate_dnn_hidden_units[-1] if len(gate_dnn_hidden_units) > 0 else self.input_dim[level_num] if level_num == 0 else
                    self.no_level0_input_dim[level_num], specific_gate_output_dim[level_num], bias=False)
                    for _ in range(self.num_tasks)]) for level_num in range(self.num_levels)])
            
            # gates for shared experts
            self.shared_gate_dnn = nn.ModuleDict()
            self.shared_gate_dnn_final_layer = nn.ModuleDict()
            for node in node_dict:
                shared_gate_output_dim = [(self.num_tasks * self.specific_expert_num + self.shared_expert_num) * len(self.edge_type_each_node[node]) for i in range(self.disen_layer_num)] + [self.num_tasks * self.specific_expert_num + self.shared_expert_num for i in range(self.num_levels-self.disen_layer_num)]
                if len(gate_dnn_hidden_units) > 0:
                    self.shared_gate_dnn[node] = nn.ModuleList([DNN(self.input_dim[level_num] if level_num == 0 else self.no_level0_input_dim[level_num],
                                                              gate_dnn_hidden_units, activation=dnn_activation,
                                                              l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                                              init_std=init_std, device=device) for level_num in
                                                          range(self.num_levels)])
#             self.add_regularization_weight(
#                 filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.shared_gate_dnn.named_parameters()),
#                 l2=l2_reg_dnn)
                self.shared_gate_dnn_final_layer[node] = nn.ModuleList(
                    [nn.Linear(
                        gate_dnn_hidden_units[-1] if len(gate_dnn_hidden_units) > 0 else self.input_dim[level_num] if level_num == 0 else
                        expert_dnn_hidden_units[-1], shared_gate_output_dim[level_num], bias=False)
                        for level_num in range(self.num_levels)])
            

            

        # 3. tower dnn (task-specific)
        final_layer_type = {'nc':SimpleHGN_final}
        self.tower_dnn_final_layer = nn.ModuleList([final_layer_type[config_dict['task_graph_type'][i]](config_dict, i)  for i in range(self.num_tasks)])

        if self.config_dict['share_output']:
            if 'nc' in config_dict['task_graph_type']:
                self.share_tower_dnn_final_layer = SimpleHGN_final(config_dict) 
#         regularization_modules = [self.specific_experts, self.shared_experts, self.specific_gate_dnn_final_layer,
#                                   self.shared_gate_dnn_final_layer, self.tower_dnn_final_layer]
#         for module in regularization_modules:
#             self.add_regularization_weight(
#                 filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], module.named_parameters()), l2=l2_reg_dnn)
        self.to(device)

    # a single cgc Layer
    def cgc_net(self, inputs, level_num, recorders, G):
        # inputs: [task1, task2, ... taskn, shared task]
        
        if level_num > 0:
            inputs, all_type_edge_src_node_feature_adj_dict = inputs  
        # 1. experts
        # task-specific experts
        input_target_node_feature_all = []
        specific_expert_outputs = []
        expert_graph = []
        for i in range(self.num_tasks):
            specific_expert_outputs_i, expert_graph_i, input_target_node_feature_all_i = [], [], []
            for j in range(self.specific_expert_num):
                if self.config_dict['split_method'] == 'full':
                    specific_expert_output, specific_input_target_node_feature, specific_graph = self.specific_experts[level_num][i][j](inputs[i])
                elif self.config_dict['split_method'] == 'layer':
                    if level_num == 0:
                        specific_expert_output, specific_input_target_node_feature, specific_graph = self.specific_experts[i][j](inputs[i], level_num, recorders=recorders)
                    else:
                        specific_expert_output, specific_input_target_node_feature, specific_graph = self.specific_experts[i][j](G, level_num, inputs[i], all_type_edge_src_node_feature_adj_dict[i][j], recorders=recorders)
                specific_expert_outputs_i.append(specific_expert_output)
                expert_graph_i.append(specific_graph)
                input_target_node_feature_all_i.append(specific_input_target_node_feature)

            # agg后的特征
            specific_expert_outputs.append(specific_expert_outputs_i)
            # 图结构
            expert_graph.append(expert_graph_i)
            # agg前的节点特征
            input_target_node_feature_all.append(input_target_node_feature_all_i)

        # shared experts
        shared_expert_outputs = []
        expert_graph_i = []
        for k in range(self.shared_expert_num):
            if self.config_dict['split_method'] == 'full':
                shared_expert_output, share_input_target_node_feature, share_graph = self.shared_experts[level_num][0][k](inputs[-1])
            elif self.config_dict['split_method'] == 'layer':
                if level_num == 0:
                    shared_expert_output, share_input_target_node_feature, share_graph = self.shared_experts[0][k](inputs[-1], level_num, recorders=recorders)
                else:
                    shared_expert_output, share_input_target_node_feature, share_graph = self.shared_experts[0][k](G, level_num, inputs[-1], all_type_edge_src_node_feature_adj_dict[-1][k], recorders=recorders)
            shared_expert_outputs.append(shared_expert_output)
            expert_graph_i.append(share_graph)
        
        expert_graph.append(expert_graph_i)
            
        # 2. gates
        # gates for task-specific experts
        if level_num < self.disen_layer_num:
            cgc_outs = []
            for i in range(self.num_tasks):
                cgc_outs_i = {}
                all_gate_output = {}
                for node_type in input_target_node_feature_all[i][0]:
                    input_target_node_feature = input_target_node_feature_all[i][0][node_type]
                    # concat task-specific expert and task-shared expert
                    ## TODO： 目前只实现了experts num=0
                    cur_experts_outputs = [specific_expert_outputs_ij[node_type][0] for specific_expert_outputs_ij in specific_expert_outputs[i]] + [shared_expert_outputs_i[node_type][0] for shared_expert_outputs_i in shared_expert_outputs]

                    # (h_tail[tail_ind, 0, :], (self.a_l, self.a_r), (h_head, head_ind, h_e, tmp_edge, res, (shp1, shp2)), (self.leakyrelu, self.dropout, self,act)), None
                    
                    a_l, a_r = specific_expert_outputs[i][0][node_type][1]
                    gate_input_feature, h_head, edge_list, h_e, tmp_edge, res, shp = shared_expert_outputs[0][node_type][2]
                    head_ind, _ = edge_list
                    leakyrelu_func, dropout_func, act_func = specific_expert_outputs[i][0][node_type][3]
                    
                    if self.config_dict['mmoe']:
                        cur_experts_outputs = [specific_expert_outputs_ij[node_type] for specific_expert_outputs_i in specific_expert_outputs for specific_expert_outputs_ij in specific_expert_outputs_i ] + [shared_expert_outputs_i[node_type] for shared_expert_outputs_i in shared_expert_outputs ]
                    # [N, e, d]
                    cur_experts_outputs = torch.stack(cur_experts_outputs, -2) # stack in column

                    if self.config_dict['gate_type']=='const':
                        gate_dnn_out = (torch.ones([1, cur_experts_outputs.shape[1]])/cur_experts_outputs.shape[1]).cuda()
                        gate_mul_expert = torch.matmul(gate_dnn_out.unsqueeze(1), cur_experts_outputs)[:,0,:] # (bs, 1, dim)
                    elif self.config_dict['gate_type']=='para':
                        gate_dnn_out = self.gate_weight_specific[level_num][i].softmax(1)
                        gate_mul_expert = torch.matmul(gate_dnn_out.unsqueeze(1), cur_experts_outputs)[:,0,:]  # (bs, 1, dim)
                    elif self.config_dict['gate_type']=='mlp':
                        # gate dnn
                        # input_shared_target_node_feature: [N, d]
                        if len(self.gate_dnn_hidden_units) > 0:
                            gate_dnn_out = self.specific_gate_dnn[node_type][level_num][i][0](gate_input_feature)
                            gate_dnn_out = self.specific_gate_dnn_final_layer[node_type][level_num][i](gate_dnn_out)
                        else:
                            gate_dnn_out = self.specific_gate_dnn_final_layer[node_type][level_num][i](gate_input_feature)

                        # gate_dnn_out: # [N_noagg, e_type_all*e_num] -> [N_noagg, e_type_all, e_num]
                        gate_dnn_out = gate_dnn_out.reshape(gate_dnn_out.shape[0], len(self.edge_type_each_node[node_type]), -1)
                        gate_dnn_out = gate_dnn_out.softmax(-1)
                        all_gate_output[node_type] = gate_dnn_out.clone()
#                         import ipdb
#                         ipdb.set_trace() 
                        gate_dnn_out = gate_dnn_out[torch.arange(len(tmp_edge)), tmp_edge-torch.min(tmp_edge), :] # [N_noagg, e_num]
                        # cur_experts_outputs [N_noagg, e_num, d]
                        # [N_noagg, 1, e_num] * [N_noagg, e_num, d] -> [N, 1, d] 
                        gate_mul_expert = torch.matmul(gate_dnn_out.unsqueeze(1), cur_experts_outputs)  # (bs, 1, dim)
                        
                        # Self-attention on the nodes - Shared attention mechanism
                        # [1, head, D]*[N, head, D] -> [N, head] -> [sub_n, head]
                        h_tail = gate_mul_expert
                        h_l = (a_l * h_head).sum(dim=-1)[head_ind]
                        h_r = (a_r * h_tail).sum(dim=-1)
                        edge_attention = dropout_func(leakyrelu_func(h_l + h_r + h_e)) # [sub_n, head]
                        # get aggregatin result by sparse matrix
                        out = []
                        edge_att_mask_all = {}
                        # [sub_n] -> [N_head, N_tail]
                        edge_attention_n = torch.sparse.FloatTensor(edge_list, edge_attention[..., 0], (shp[0], shp[1]))
                        edge_attention_n = torch.sparse.softmax(edge_attention_n, dim=1)
                        edge_attention_softmax = edge_attention_n.coalesce().values() 
                        weight_h_tail = torch.mul(edge_attention_softmax.unsqueeze(-1), h_tail[:,0,:])
                        weight_h_tail_mat = torch.sparse_coo_tensor(edge_list, weight_h_tail, (shp[0], shp[1], weight_h_tail.shape[-1]))
                        out.append(torch.sparse.sum(weight_h_tail_mat, -2).to_dense())
                        out = torch.stack(out, 1) #  [N_head, head, D]
                        if (recorders is not None) and ('att_record' in recorders.recorder_dict):
                            if recorders.recorder_dict['att_record'].if_record:
                                att_weight = {}
                                for e_type in self.edge_type_each_node[node_type]:
                                    if e_type in edge_att_mask_all:
                                        edge_att_mask = edge_att_mask_all[e_type]
                                    else:
                                        edge_mask = torch.sparse.FloatTensor(edge_list, (tmp_edge==e_type).float(), (shp[0], shp[1]))
                                        edge_att_mask = edge_attention_n.mul(edge_mask)
                                    att_weight[self.edge_list[e_type]] =  torch.sparse.sum(edge_att_mask, -1).to_dense().cpu()
                                recorders.recorder_dict['att_record'].record(att_weight, level_num, self.task_names[i])     
                        # node residual
                        gate_mul_expert = out.view(out.shape[0], -1) + res #  [N_head, head*D]
                        # use activation or not
                        if act_func is not None:
                            gate_mul_expert = act_func(gate_mul_expert)
                            
                        

                    
                        # cur_experts_outputs 
                    cgc_outs_i[node_type] = gate_mul_expert
                cgc_outs.append(cgc_outs_i)
 
                if (recorders is not None) and ('gate_weight_record' in recorders.recorder_dict):
                    if recorders.recorder_dict['gate_weight_record'].if_record:
                        gate_save = {}
                        for node_type in all_gate_output:
                            for ind, e in enumerate(self.edge_type_each_node[node_type]):
                                gate_save[self.edge_list[e]] = all_gate_output[node_type][:, ind, :].cpu().data
                        recorders.recorder_dict['gate_weight_record'].record(gate_save, level_num, self.task_names[i])     

            # gates for shared experts
            cgc_outs_i = {}
            all_gate_output = {}
            for node_type in input_target_node_feature_all[0][0]:
                cur_experts_outputs = [specific_expert_outputs_i_j[node_type][0] for specific_expert_outputs_i in specific_expert_outputs 
                                           for specific_expert_outputs_i_j in specific_expert_outputs_i] + [shared_expert_outputs_i[node_type][0] for shared_expert_outputs_i in shared_expert_outputs ]
                
                a_l, a_r = shared_expert_outputs[0][node_type][1]
                gate_input_feature, h_head, edge_list, h_e, tmp_edge, res, shp = shared_expert_outputs[0][node_type][2]
                head_ind, _ = edge_list
                leakyrelu_func, dropout_func, act_func = shared_expert_outputs[0][node_type][3]           
                   
                if self.config_dict['mmoe']:
                    cur_experts_outputs = [specific_expert_outputs_ij[node_type] for specific_expert_outputs_i in specific_expert_outputs for specific_expert_outputs_ij in specific_expert_outputs_i ] + [shared_expert_outputs_i[node_type] for shared_expert_outputs_i in shared_expert_outputs ]
                cur_experts_outputs = torch.stack(cur_experts_outputs, -2)
                input_shared_target_node_feature = share_input_target_node_feature[node_type]
                if self.config_dict['gate_type']=='const':
                    gate_dnn_out = (torch.ones([1, cur_experts_outputs.shape[1]])/cur_experts_outputs.shape[1]).cuda()
                    gate_mul_expert = torch.matmul(gate_dnn_out.unsqueeze(1), cur_experts_outputs)[:,0,:] # (bs, 1, dim)
                elif self.config_dict['gate_type']=='para':
                    gate_dnn_out = self.gate_weight_sharing[level_num]
                    gate_mul_expert = torch.matmul(gate_dnn_out.softmax(1).unsqueeze(1), cur_experts_outputs)[:,0,:]  # (bs, 1, dim)
                elif self.config_dict['gate_type']=='mlp':
                    
                    if len(self.gate_dnn_hidden_units) > 0:
                        gate_dnn_out = self.shared_gate_dnn[node_type][level_num](gate_input_feature)
                        gate_dnn_out = self.shared_gate_dnn_final_layer[node_type][level_num](gate_dnn_out)
                    else:
                        gate_dnn_out = self.shared_gate_dnn_final_layer[node_type][level_num](gate_input_feature)
                    # gate_dnn_out: # [N, e_type_all*e_num] -> [N, e_type_all, e_num]
                    gate_dnn_out = gate_dnn_out.reshape(gate_dnn_out.shape[0], len(self.edge_type_each_node[node_type]), -1)
                    gate_dnn_out = gate_dnn_out.softmax(-1)
                    all_gate_output[node_type] = gate_dnn_out.clone()
                    gate_dnn_out = gate_dnn_out[torch.arange(len(tmp_edge)), tmp_edge-torch.min(tmp_edge), :] # [N_noagg, e_num]
                    # cur_experts_outputs [N_noagg, e_num, d]
                    # [N_noagg, 1, e_num] * [N_noagg, e_num, d] -> [N, 1, d] 
                    gate_mul_expert = torch.matmul(gate_dnn_out.unsqueeze(1), cur_experts_outputs)  # (bs, 1, dim)

                    # Self-attention on the nodes - Shared attention mechanism
                    # [1, head, D]*[N, head, D] -> [N, head] -> [sub_n, head]
                    h_tail = gate_mul_expert
                    h_l = (a_l * h_head).sum(dim=-1)[head_ind]
                    h_r = (a_r * h_tail).sum(dim=-1)
                    edge_attention = dropout_func(leakyrelu_func(h_l + h_r + h_e)) # [sub_n, head]
                    # get aggregatin result by sparse matrix
                    out = []
                    edge_att_mask_all = {}
                    # [sub_n] -> [N_head, N_tail]
                    edge_attention_n = torch.sparse.FloatTensor(edge_list, edge_attention[..., 0], (shp[0], shp[1]))
                    edge_attention_n = torch.sparse.softmax(edge_attention_n, dim=1)
                    edge_attention_softmax = edge_attention_n.coalesce().values() 
                    weight_h_tail = torch.mul(edge_attention_softmax.unsqueeze(-1), h_tail[:,0,:])
                    weight_h_tail_mat = torch.sparse_coo_tensor(edge_list, weight_h_tail, (shp[0], shp[1], weight_h_tail.shape[-1]))
                    out.append(torch.sparse.sum(weight_h_tail_mat, -2).to_dense())
                    out = torch.stack(out, 1) #  [N_head, head, D]
                    if (recorders is not None) and ('att_record' in recorders.recorder_dict):
                        if recorders.recorder_dict['att_record'].if_record:
                            att_weight = {}
                            for e_type in self.edge_type_each_node[node_type]:
                                if e_type in edge_att_mask_all:
                                    edge_att_mask = edge_att_mask_all[e_type]
                                else:
                                    edge_mask = torch.sparse.FloatTensor(edge_list, (tmp_edge==e_type).float(), (shp[0], shp[1]))
                                    edge_att_mask = edge_attention_n.mul(edge_mask)
                                att_weight[self.edge_list[e_type]] =  torch.sparse.sum(edge_att_mask, -1).to_dense().cpu()
                            recorders.recorder_dict['att_record'].record(att_weight, level_num, 'share')     
                    # node residual
                    gate_mul_expert = out.view(out.shape[0], -1) + res #  [N_head, head*D]
                    # use activation or not
                    if act_func is not None:
                        gate_mul_expert = act_func(gate_mul_expert)
                   
                
                cgc_outs_i[node_type] = gate_mul_expert
  

            cgc_outs.append(cgc_outs_i) 
            if (recorders is not None) and ('gate_weight_record' in recorders.recorder_dict):
                if recorders.recorder_dict['gate_weight_record'].if_record:
                    gate_save = {}
                    for node_type in all_gate_output:
                        for ind, e in enumerate(self.edge_type_each_node[node_type]):
                            gate_save[self.edge_list[e]] = all_gate_output[node_type][:, ind, :].cpu().data
                    recorders.recorder_dict['gate_weight_record'].record(gate_save, level_num, 'share')     
                        
                        
        
        else:
            cgc_outs = []
            for i in range(self.num_tasks):
                cgc_outs_i = {}
                all_gate_output = {}
                for node_type in input_target_node_feature_all[i][0]:
                    input_target_node_feature = input_target_node_feature_all[i][0][node_type]
                    # concat task-specific expert and task-shared expert
                    ## TODO： 目前只实现了experts num=0
                    cur_experts_outputs = [specific_expert_outputs_ij[node_type] for specific_expert_outputs_ij in specific_expert_outputs[i]] + [shared_expert_outputs_i[node_type] for shared_expert_outputs_i in shared_expert_outputs]
                    if self.config_dict['mmoe']:
                        cur_experts_outputs = [specific_expert_outputs_ij[node_type] for specific_expert_outputs_i in specific_expert_outputs for specific_expert_outputs_ij in specific_expert_outputs_i ] + [shared_expert_outputs_i[node_type] for shared_expert_outputs_i in shared_expert_outputs ]
                    cur_experts_outputs = torch.stack(cur_experts_outputs, 1) # stack in column

                    if self.config_dict['gate_type']=='const':
                        gate_dnn_out = (torch.ones([1, cur_experts_outputs.shape[1]])/cur_experts_outputs.shape[1]).cuda()
                        gate_mul_expert = torch.matmul(gate_dnn_out.unsqueeze(1), cur_experts_outputs)  # (bs, 1, dim)
                    elif self.config_dict['gate_type']=='para':
                        gate_dnn_out = self.gate_weight_specific[level_num][i].softmax(1)
                        gate_mul_expert = torch.matmul(gate_dnn_out.unsqueeze(1), cur_experts_outputs)  # (bs, 1, dim)
                    elif self.config_dict['gate_type']=='mlp':
                        # gate dnn
                        nt = node_type if self.config_dict['dif_gate'] else '0'
                        if len(self.gate_dnn_hidden_units) > 0:
                            gate_dnn_out = self.specific_gate_dnn[nt][level_num][i][0](input_target_node_feature)
                            gate_dnn_out = self.specific_gate_dnn_final_layer[nt][level_num][i](gate_dnn_out)
                        else:
                            gate_dnn_out = self.specific_gate_dnn_final_layer[nt][level_num][i](input_target_node_feature)

                        gate_dnn_out = gate_dnn_out.softmax(1)
                        gate_mul_expert = torch.matmul(gate_dnn_out.unsqueeze(1), cur_experts_outputs)  # (bs, 1, dim)

                    all_gate_output[node_type] = gate_dnn_out

                        # cur_experts_outputs 
                    cgc_outs_i[node_type] = gate_mul_expert[:,0,:]
                cgc_outs.append(cgc_outs_i)

#                 if (recorders is not None) and ('gate_weight_record' in recorders.recorder_dict):
#                     if recorders.recorder_dict['gate_weight_record'].if_record:
#                         recorders.recorder_dict['gate_weight_record'].record(all_gate_output, level_num, self.task_names[i])     


            # gates for shared experts
            cgc_outs_i = {}
            all_gate_output = {}
            for node_type in input_target_node_feature_all[0][0]:
                cur_experts_outputs = [specific_expert_outputs_i_j[node_type] for specific_expert_outputs_i in specific_expert_outputs 
                                       for specific_expert_outputs_i_j in specific_expert_outputs_i] + [shared_expert_outputs_i[node_type] for shared_expert_outputs_i in shared_expert_outputs ]
                if self.config_dict['mmoe']:
                    cur_experts_outputs = [specific_expert_outputs_ij[node_type] for specific_expert_outputs_i in specific_expert_outputs for specific_expert_outputs_ij in specific_expert_outputs_i ] + [shared_expert_outputs_i[node_type] for shared_expert_outputs_i in shared_expert_outputs ]
                cur_experts_outputs = torch.stack(cur_experts_outputs, 1)
                input_shared_target_node_feature = share_input_target_node_feature[node_type]
                if self.config_dict['gate_type']=='const':
                    gate_dnn_out = (torch.ones([1, cur_experts_outputs.shape[1]])/cur_experts_outputs.shape[1]).cuda()
                    gate_mul_expert = torch.matmul(gate_dnn_out.unsqueeze(1), cur_experts_outputs)  # (bs, 1, dim)
                elif self.config_dict['gate_type']=='para':
                    gate_dnn_out = self.gate_weight_sharing[level_num]
                    gate_mul_expert = torch.matmul(gate_dnn_out.softmax(1).unsqueeze(1), cur_experts_outputs)  # (bs, 1, dim)
                elif self.config_dict['gate_type']=='mlp':
                    nt = node_type if self.config_dict['dif_gate'] else '0'
                    if len(self.gate_dnn_hidden_units) > 0:
                        gate_dnn_out = self.shared_gate_dnn[nt][level_num](input_shared_target_node_feature)
                        gate_dnn_out = self.shared_gate_dnn_final_layer[nt][level_num](gate_dnn_out)
                    else:
                        gate_dnn_out = self.shared_gate_dnn_final_layer[nt][level_num](input_shared_target_node_feature)
                    gate_mul_expert = torch.matmul(gate_dnn_out.softmax(1).unsqueeze(1), cur_experts_outputs)  # (bs, 1, dim)

                all_gate_output[node_type] = gate_dnn_out
                cgc_outs_i[node_type] = gate_mul_expert[:,0,:]

            cgc_outs.append(cgc_outs_i) 
#                 if (recorders is not None) and ('gate_weight_record' in recorders.recorder_dict):
#                         if recorders.recorder_dict['gate_weight_record'].if_record:
#                             recorders.recorder_dict['gate_weight_record'].record(all_gate_output, level_num, 'share')     


        return cgc_outs, expert_graph
    
    
    
    
  
    
    def forward(self, X, recorders=None):
        
        if 'lp' in self.config_dict['task_graph_type']:
            X, tail_nodex_info_each_iter = X
        
        if self.share_emb_net:
            for ntype in X.ntypes:
                X.nodes[ntype].data['feat_emb'] = self.Node_Transform_list[ntype]((X.nodes[ntype].data['feat']))
      
        # repeat `X` for several times to generate cgc input
        ple_inputs = [X] * (self.num_tasks + 1)  # [task1, task2, ... taskn, shared task]
        ple_outputs = []
        for i in range(self.num_levels):
            ple_outputs = self.cgc_net(inputs=ple_inputs, level_num=i, recorders=recorders, G=X)
            ple_inputs = ple_outputs
            

        # tower dnn (task-specific)
        ple_outputs, specific_expert_graph = ple_outputs
        task_outs = {}
        select_expert = 0
        for i in range(self.num_tasks):
#             k = 1
#             output = self.tower_dnn_final_layer[k](ple_outputs[k], specific_expert_graph[k], X, i)
            if 'lp' in self.config_dict['task_graph_type']:
                output = self.tower_dnn_final_layer[i](ple_outputs[i], specific_expert_graph[i][select_expert], X, tail_nodex_info_each_iter[i])
            else:
                output = self.tower_dnn_final_layer[i](ple_outputs[i], specific_expert_graph[i][select_expert], X, i)
            task_outs[self.task_names[i]] = output
        
        if self.config_dict['share_output']:
            for i in range(self.num_tasks):     
                if 'lp' in self.config_dict['task_graph_type']:
                    output = self.share_tower_dnn_final_layer[i](ple_outputs[-1], specific_expert_graph[-1][select_expert], X, tail_nodex_info_each_iter[i])
                else:
                    output = self.share_tower_dnn_final_layer(ple_outputs[-1], specific_expert_graph[-1][select_expert], X, i)
                task_outs['share_'+self.task_names[i]] = output
    #       
        
            
        return task_outs
    
    def get_specific_paramter(self, para_type):
        params = []
        if 'share' in para_type:
            for name , param in self.named_parameters():
                if ('output_linear' not in name) & ('task_metapath_weight' not in name) & ('uncertainty_weight' not in name):
                    params.append(param)
                    
        if 'task_specific' in para_type:
            for name , param in self.named_parameters():
                if 'output_linear' in name:
                    params.append(param)

        if 'uncertainty_weight' in para_type:
            for name , param in self.named_parameters():
                if 'uncertainty_weight' in name:
                    params.append(param)
        if 'not_uncertainty_weight' in para_type:
            for name , param in self.named_parameters():
                if 'uncertainty_weight' not in name:
                    params.append(param)    
 
        return params














import dgl
# from .SimpleHGN import SimpleHGNLayer
from collections import defaultdict


class SimpleHGN_final(nn.Module):
    def __init__(self, Model_Config_dict, task_ind):
        super().__init__()
        
        node_dict =  Model_Config_dict['node_dict']
        edge_dict = Model_Config_dict['edge_dict']
        Target_Node_Type = Model_Config_dict['Target_Node_Type']
        node_type_to_feature_len_dict = Model_Config_dict['node_type_to_feature_len_dict']
        
        self.Target_Node_Type = Target_Node_Type
        if type(self.Target_Node_Type) != list:
            self.Target_Node_Type = [self.Target_Node_Type]
        self.all_relation_list = list(edge_dict.keys())
        self.layer_num = Model_Config_dict['layer_num']
        self.head_num = Model_Config_dict['nhead']
        self.args_cuda = Model_Config_dict['args_cuda']
        self.class_num = Model_Config_dict['class_num'][task_ind]
        self.task_type = Model_Config_dict['task_type']
        self.task_names = Model_Config_dict['task_name']
        # 判断是否为single node&multi label
        if len(self.task_names)>1 and len(self.Target_Node_Type)==1:
            self.single_node_mtl = True
        else:
            self.single_node_mtl = False
        node_feature_hid_len = Model_Config_dict['node_feature_hid_len']
        GAT_hid_len = Model_Config_dict['GAT_hid_len']
        dropout = Model_Config_dict['dropout']
        edge_residual_alpha = Model_Config_dict['edge_residual_alpha']
        
        self.h_dropout = nn.Dropout(dropout)
        
        self.edge_GAT = SimpleHGNLayer(edge_feats_len = Model_Config_dict['edge_feats_len'], 
                                       in_features_len = GAT_hid_len * self.head_num,
                                       out_features_len = self.class_num,
                                       nhead = 1,
                                       edge_dict = self.all_relation_list,
                                       edge_type_each_node = Model_Config_dict['edge_type_each_node'],
                                       node_residual=True,
                                       edge_residual_alpha=edge_residual_alpha,
                                       activation=None,
                                       spilt_edge=False
                                      )
        # 输出预测结果
        self.activation = nn.Softmax(1)#nn.Sigmoid()
        
        self.register_buffer("epsilon", torch.FloatTensor([1e-12]))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.edge_GAT.reset_parameters()

    def forward(self, gate_weighted_feaure, all_type_edge_src_node_feature_adj_dict, G, target_node_index):  
        
        # 清空gate加权之前的tail_featre
        for srctype in all_type_edge_src_node_feature_adj_dict:
            all_type_edge_src_node_feature_adj_dict[srctype]['tail_feature'] = []
            
        # 更新全部节点特征
        for srctype, etype, dsttype in G.canonical_etypes:
            all_type_edge_src_node_feature_adj_dict[srctype]['head_feature'] = gate_weighted_feaure[srctype]
            # 保存该类边对应的尾节点特征
            tail_feature = gate_weighted_feaure[dsttype]
            all_type_edge_src_node_feature_adj_dict[srctype]['tail_feature'].append(tail_feature)

        inter_srctype_feaure = {}
        for srctype in all_type_edge_src_node_feature_adj_dict:
            # 导入GAT获取结果
            inter_srctype_feaure[srctype] = self.edge_GAT(all_type_edge_src_node_feature_adj_dict[srctype], all_type_edge_src_node_feature_adj_dict[srctype]['res_attn'])
#             all_type_edge_src_node_feature_adj_dict[srctype]['tail_feature'] = []
        

        # 取出最后一跳目标节点输出的结果
        if self.single_node_mtl:
            target_node = self.Target_Node_Type[0]
        else:
            target_node = self.Target_Node_Type[target_node_index]
        h_prime = inter_srctype_feaure[target_node][0]
        h_prime = inter_srctype_feaure[target_node][0]

        # L2 Normalization
        h_prime = h_prime / (torch.max(torch.norm(h_prime, dim=1, keepdim=True), self.epsilon))

        # 转化为概率，并返回预测结果
        if self.single_node_mtl:
            task_name = self.task_names[target_node_index]
            task_type = self.task_type[task_name]
        else:
            task_type = self.task_type[target_node]
            
        if task_type == 'single-label':
            output = nn.Softmax(1)(h_prime)
        elif task_type == 'multi-label':
            output = torch.sigmoid(h_prime)
        else:
            raise NameError(f'task type {self.task_type[target_node]} does not exist!')

        if self.class_num<=2:
            output = output[:,0]

        return output
    

    
    
    

class SimpleHGN_MTL_Layer(nn.Module):
    def __init__(self, Model_Config_dict, task_name):
        super().__init__()
        
        node_dict = Model_Config_dict['node_dict']
        edge_dict = Model_Config_dict['edge_dict']
        Target_Node_Type = Model_Config_dict['Target_Node_Type']
        node_type_to_feature_len_dict = Model_Config_dict['node_type_to_feature_len_dict']
        
        self.Target_Node_Type = Target_Node_Type
        if type(self.Target_Node_Type) != list:
            self.Target_Node_Type = [self.Target_Node_Type]
        self.all_relation_list = list(edge_dict.keys())
#         self.all_relation_list.append('self_loop')
        self.layer_num = Model_Config_dict['layer_num']
        self.head_num = Model_Config_dict['nhead']
        self.args_cuda = Model_Config_dict['args_cuda']
        self.model =  Model_Config_dict['model']
        self.task_name = task_name
        self.disen_layer_num = Model_Config_dict['disen_layer_num']
        node_feature_hid_len = Model_Config_dict['node_feature_hid_len']
        GAT_hid_len = Model_Config_dict['GAT_hid_len']
        dropout = Model_Config_dict['dropout']
        edge_residual_alpha = Model_Config_dict['edge_residual_alpha']
        
        self.h_dropout = nn.Dropout(dropout)
        
        # 特征转化
        self.share_emb_net = True
        if not self.share_emb_net:
            self.Node_Transform_list = {}
            for tmp_node_type in node_dict:
                tmp_linear = nn.Linear(node_type_to_feature_len_dict[tmp_node_type], node_feature_hid_len)
                self.Node_Transform_list[tmp_node_type] = tmp_linear
                self.add_module('{}_Node_Transform'.format(tmp_node_type), self.Node_Transform_list[tmp_node_type])

        # 有多少种类型的元路径，每种元路径有多少条，就生成多少个GAT
        self.edge_GAT =  nn.ModuleList()
        # input projection
        ## TODO: 第一层加node_redidual
        if self.disen_layer_num == 0:
            spilt_edge = False
        else:
            spilt_edge = True
        self.edge_GAT.append(SimpleHGNLayer(edge_feats_len = Model_Config_dict['edge_feats_len'], 
                                       in_features_len = node_feature_hid_len,
                                       out_features_len = GAT_hid_len,
                                       nhead = self.head_num,
                                       edge_dict = self.all_relation_list,
                                       edge_type_each_node = Model_Config_dict['edge_type_each_node'],
                                       node_residual=False,
                                       edge_residual_alpha=edge_residual_alpha,
                                       activation=nn.functional.elu,
                                       spilt_edge = spilt_edge
                                      ))
        # middle projection
        for i in range(self.layer_num-1):
            if i < self.disen_layer_num-1:
                spilt_edge = True
            else:
                spilt_edge = False
            self.edge_GAT.append(SimpleHGNLayer(edge_feats_len = Model_Config_dict['edge_feats_len'], 
                                           in_features_len = GAT_hid_len * self.head_num,
                                           out_features_len = GAT_hid_len,
                                           nhead = Model_Config_dict['nhead'],
                                           edge_dict = self.all_relation_list,
                                           edge_type_each_node = Model_Config_dict['edge_type_each_node'],
                                           node_residual=True,
                                           edge_residual_alpha=Model_Config_dict['edge_residual_alpha'],
                                           activation=nn.functional.elu,
                                           spilt_edge = spilt_edge           
                                          ))
        

        
        # 输出预测结果
        self.activation = nn.Softmax(1)#nn.Sigmoid()
        
        self.register_buffer("epsilon", torch.FloatTensor([1e-12]))
        
        self.reset_parameters()

    def reset_parameters(self):
        for edge_GAT_i in self.edge_GAT:
            edge_GAT_i.reset_parameters()

    def forward(self, G, level_num, inter_srctype_feaure=None, all_type_edge_src_node_feature_adj_dict=None, recorders=None):
        
        # inter_srctype_feaure 经过gate加权后的图节点特征
        # all_type_edge_src_node_feature_adj_dict存储了edge residual和节点之间的关系，以及对于权重
        
        if level_num == 0:
            h = {}

            # 先将涉及到的节点对应的特征都进行转化
            if self.share_emb_net:
                for ntype in G.ntypes:
                    h[ntype] = G.nodes[ntype].data['feat_emb'].clone()
                    
            ## TODO: 对于不share输入的情况加入MultiSFS 
            if 'MultiSFS' in self.model:
                for ntype in h:
                    h[ntype] = h[ntype] * G.nodes[ntype].data['feat_mask_'+self.task_name]


            all_type_edge_src_node_feature_adj_dict = {}
            adj_begin_index = {}
            # 存储头节点对应的所有类型边的尾节点特征
            for srctype, etype, dsttype in G.canonical_etypes:

                # 存储头节点对应的所有类型边的尾节点特征
                if srctype not in all_type_edge_src_node_feature_adj_dict:
                    all_type_edge_src_node_feature_adj_dict[srctype] = defaultdict(list)
                    all_type_edge_src_node_feature_adj_dict[srctype]['head_feature'] = h[srctype]
                    all_type_edge_src_node_feature_adj_dict[srctype]['res_attn'] = None
                    adj_begin_index[srctype] = 0

                # 保存该类边对应的尾节点特征
                tail_feature = h[dsttype]
                adj = G.all_edges(etype=etype)  #data_ptr()相同，指向同一地址
                adj = torch.stack(list(adj), dim=0) # stack后生成的adj指向新地址
                all_type_edge_src_node_feature_adj_dict[srctype]['tail_feature'].append(tail_feature)#.append(tail_feature[adj[1,:]])
                # 由于需要stack，对adj的index进行修改
                adj[1,:] = adj[1,:] + adj_begin_index[srctype]
                adj_begin_index[srctype] = adj_begin_index[srctype] + tail_feature.shape[0]
                all_type_edge_src_node_feature_adj_dict[srctype]['Adj'].append(adj)
                # 生成edge_type的对应index
                tmp_edge_index = self.all_relation_list.index(etype)
                tmp_edge_index = torch.tensor([tmp_edge_index]*adj.shape[1])
                if self.args_cuda:
                    tmp_edge_index = tmp_edge_index.cuda()
                all_type_edge_src_node_feature_adj_dict[srctype]['tmp_edge_index'].append(tmp_edge_index)

            
            feature_before_agg = h
        else:

            feature_before_agg = inter_srctype_feaure
             # 清空gate加权之前的tail_featre
#             for srctype in all_type_edge_src_node_feature_adj_dict:
#                 all_type_edge_src_node_feature_adj_dict[srctype]['tail_feature'] = []
            # 用gate加权过的节点特征更新图结构中所有头尾节点
            for srctype, etype, dsttype in G.canonical_etypes:
                all_type_edge_src_node_feature_adj_dict[srctype]['head_feature'] = inter_srctype_feaure[srctype]
                # 保存该类边对应的尾节点特征
                tail_feature = inter_srctype_feaure[dsttype]
                all_type_edge_src_node_feature_adj_dict[srctype]['tail_feature'].append(tail_feature)

        
        inter_srctype_feaure = {}
        # 对每一个节点，进行边聚合

        for srctype in all_type_edge_src_node_feature_adj_dict:
            # 导入GAT获取结果
            
            inter_srctype_feaure[srctype] = self.edge_GAT[level_num](all_type_edge_src_node_feature_adj_dict[srctype], all_type_edge_src_node_feature_adj_dict[srctype]['res_attn'], srctype, level_num, self.task_name, recorders)
            all_type_edge_src_node_feature_adj_dict[srctype]['tail_feature'] = []
        # 更新edge weight
        for srctype, etype, dsttype in G.canonical_etypes:
#             all_type_edge_src_node_feature_adj_dict[srctype]['head_feature'] = inter_srctype_feaure[srctype][0]
            all_type_edge_src_node_feature_adj_dict[srctype]['res_attn'] = inter_srctype_feaure[srctype][1]
            # 保存该类边对应的尾节点特征
#             tail_feature = inter_srctype_feaure[dsttype][0]
#             all_type_edge_src_node_feature_adj_dict[srctype]['tail_feature'].append(tail_feature)

        
        for srctype in inter_srctype_feaure:
            inter_srctype_feaure[srctype] = inter_srctype_feaure[srctype][0]
     

        # 聚合后每个点的特征，聚合前每个点的特征，图关系
        return inter_srctype_feaure, feature_before_agg, all_type_edge_src_node_feature_adj_dict
    
    
import math    
class SimpleHGNLayer(nn.Module):
    """
    implementation of Simple-HGN layer
    source code comes from:  
        https://github.com/THUDM/CogDL/blob/master/examples/simple_hgn/conv.py#L72
    or
        https://github.com/THUDM/HGB/blob/master/NC/benchmark/methods/baseline/conv.py
    """

    def __init__(
        self,
        edge_feats_len,
        in_features_len,
        out_features_len,
        nhead,
        edge_dict,
        edge_type_each_node,
        feat_drop=0.5, 
        attn_drop=0.5,
        negative_slope=0.2, # 0.05
        node_residual=False,
        edge_residual_alpha=0.05,
        activation=None,
        spilt_edge=False
        
    ):
        super(SimpleHGNLayer, self).__init__()
        self.edge_feats_len = edge_feats_len
        self.in_features_len = in_features_len
        self.out_features_len = out_features_len
        self.nhead = nhead
        self.edge_dict = edge_dict
        self.edge_emb = nn.Parameter(torch.zeros(size=(len(edge_dict), edge_feats_len)))  # nn.Embedding(num_etypes, edge_feats)
        self.edge_type_each_node = edge_type_each_node

        self.W = nn.Parameter(torch.FloatTensor(in_features_len, out_features_len * nhead))
        self.W_e = nn.Parameter(torch.FloatTensor(edge_feats_len, edge_feats_len * nhead))

        self.a_l = nn.Parameter(torch.zeros(size=(1, nhead, out_features_len)))
        self.a_r = nn.Parameter(torch.zeros(size=(1, nhead, out_features_len)))
        self.a_e = nn.Parameter(torch.zeros(size=(1, nhead, edge_feats_len)))

        self.feat_drop = nn.Dropout(feat_drop)
        self.dropout = nn.Dropout(attn_drop)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        
        self.act = activation
        #self.act = None if activation is None else get_activation(activation)
        self.spilt_edge = spilt_edge

        if node_residual:
            self.node_residual = nn.Linear(in_features_len, out_features_len * nhead)
        else:
            self.register_buffer("node_residual", None)
        self.reset_parameters()
        self.edge_residual_alpha = edge_residual_alpha  # edge residual weight

    def reset_parameters(self):
        def reset(tensor):
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

        reset(self.a_l)
        reset(self.a_r)
        reset(self.a_e)
        reset(self.W)
        reset(self.W_e)
        reset(self.edge_emb)
    
    def forward(self, all_type_edge_src_node_feature_adj_dict, res_attn, srctype=None, level_num=None, task_name=None, recorders=None):
        
        head_node_feature = all_type_edge_src_node_feature_adj_dict['head_feature']
        all_tail_node_feature = all_type_edge_src_node_feature_adj_dict['tail_feature']  # [n1+n2+.., d]
        all_tail_node_feature = torch.cat(all_tail_node_feature, 0)
        edge_list = all_type_edge_src_node_feature_adj_dict['Adj']  # [2, n1+n2+..]
        edge_list = torch.cat(edge_list, 1)
        tmp_edge = all_type_edge_src_node_feature_adj_dict['tmp_edge_index']
        tmp_edge = torch.cat(tmp_edge)
        all_tail_node_num = all_tail_node_feature.shape[0]
        
        # d:in_features_len  D:out_features_len, de:out_features_len
        x_head = self.feat_drop(head_node_feature)  # [N, d]  
        x_tail = self.feat_drop(all_tail_node_feature)
        #  [N, d]*[d, D*head] -> [N, head, D]
        h_head = torch.matmul(x_head, self.W).view(-1, self.nhead, self.out_features_len) 
        h_tail = torch.matmul(x_tail, self.W).view(-1, self.nhead, self.out_features_len) 
        # [edge_num, de]*[de, de*head] -> [edge_num, head, de]
        e = torch.matmul(self.edge_emb, self.W_e).view(-1, self.nhead, self.edge_feats_len)
        
        head_ind, tail_ind = edge_list
        h_e = (self.a_e * e).sum(dim=-1)[tmp_edge]
        
        if self.spilt_edge:
            if self.node_residual is not None:
                # [N, d]*[d, D*head] -> [N,D*head]
                #res = self.node_residual(x_head)
                res = self.node_residual(head_node_feature)
            else:
                res = torch.zeros_like(x_head)
#             tmp_edge -= torch.min(tmp_edge)
#             if torch.sum(tmp_edge<0) > 0:
#                 import ipdb
#                 ipdb.set_trace() 
            gate_input_feature = torch.cat([x_head[head_ind, :], x_tail[tail_ind, :]], -1)
            #gate_input_feature = torch.cat([head_node_feature[head_ind, :], all_tail_node_feature[tail_ind, :]], -1)
            return (h_tail[tail_ind, 0, :], (self.a_l, self.a_r), (gate_input_feature, h_head, edge_list, h_e, tmp_edge, res,  (x_head.shape[0], x_tail.shape[0])), (self.leakyrelu, self.dropout, self.act)), None
        
        
        # Self-attention on the nodes - Shared attention mechanism
        # [1, head, D]*[N, head, D] -> [N, head] -> [sub_n, head]
        h_l = (self.a_l * h_head).sum(dim=-1)[head_ind]
        h_r = (self.a_r * h_tail).sum(dim=-1)[tail_ind]
        edge_attention = self.leakyrelu(h_l + h_r + h_e) # [sub_n, head]
        # Cannot use dropout on sparse tensor , put dropout operation before sparse
        edge_attention = self.dropout(edge_attention)

        # get aggregatin result by sparse matrix
        out = []
        edge_attention_weight = []
        for n in range(self.nhead):
            # [sub_n] -> [N_head, N_tail]
            edge_attention_n = torch.sparse.FloatTensor(edge_list, edge_attention[..., n], (x_head.shape[0], x_tail.shape[0]))
            edge_attention_n = torch.sparse.softmax(edge_attention_n, dim=1)
            
            edge_att_mask_all = {}
            out.append(torch.sparse.mm(edge_attention_n, h_tail[:,n,:]))
            edge_attention_weight.append(edge_attention_n)
                
        
        out = torch.stack(out, 1) #  [N_head, head, D]

        if (recorders is not None) and ('att_record' in recorders.recorder_dict):
            if recorders.recorder_dict['att_record'].if_record:
                att_weight = {}
                for e_type in self.edge_type_each_node[srctype]:
                    if e_type in edge_att_mask_all:
                        edge_att_mask = edge_att_mask_all[e_type]
                    else:
                        edge_mask = torch.sparse.FloatTensor(edge_list, (tmp_edge==e_type).float(), (x_head.shape[0], x_tail.shape[0]))
                        edge_att_mask = edge_attention_n.mul(edge_mask)
                    att_weight[self.edge_dict[e_type]] =  torch.sparse.sum(edge_att_mask, -1).to_dense().cpu()
                recorders.recorder_dict['att_record'].record(att_weight, level_num, task_name)     


        # node residual
        out = out.view(out.shape[0], -1) #  [N_head, head*D]
        if self.node_residual is not None:
            # [N, d]*[d, D*head] -> [N,D*head]
            res = self.node_residual(x_head)
            out += res
        # use activation or not
        if self.act is not None:
            out = self.act(out)

        return out, edge_attention_weight
    
    
    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features_len) + " -> " + str(self.out_features_len) + ")"

    