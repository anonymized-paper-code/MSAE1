# -*- coding:utf-8 -*-
"""
Author:
    zanshuxun, zanshuxun@aliyun.com

Reference:
    [1] Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts[C] (https://dl.acm.org/doi/10.1145/3219819.3220007)
"""
import torch
import torch.nn as nn

from ..basemodel import BaseModel
from ...inputs import combined_dnn_input
from .core import DNN, PredictionLayer

from ple_simplehgn import SimpleHGN_final, SimpleHGN_MTL_Layer

class MMOE(nn.Module):

    def __init__(self, dnn_feature_columns, num_experts=3, expert_dnn_hidden_units=(256, 128),
                 gate_dnn_hidden_units=(64,), tower_dnn_hidden_units=(64,), l2_reg_linear=0.00001,
                 l2_reg_embedding=0.00001, l2_reg_dnn=0,
                 init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False,
                 task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr'), device='cpu', gpus=None):
        super(MMOE, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
                                   l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, init_std=init_std,
                                   seed=seed, device=device, gpus=gpus)
        self.num_tasks = len(task_names)
        if self.num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1")
        if num_experts <= 1:
            raise ValueError("num_experts must be greater than 1")
        if len(dnn_feature_columns) == 0:
            raise ValueError("dnn_feature_columns is null!")
        if len(task_types) != self.num_tasks:
            raise ValueError("num_tasks must be equal to the length of task_types")

        for task_type in task_types:
            if task_type not in ['binary', 'regression']:
                raise ValueError("task must be binary or regression, {} is illegal".format(task_type))

        self.num_experts = num_experts
        self.task_names = task_names
        self.input_dim = self.compute_input_dim(dnn_feature_columns)
        self.expert_dnn_hidden_units = expert_dnn_hidden_units
        self.gate_dnn_hidden_units = gate_dnn_hidden_units
        self.tower_dnn_hidden_units = tower_dnn_hidden_units

        # expert dnn
        self.expert_dnn = nn.ModuleList([DNN(self.input_dim, expert_dnn_hidden_units, activation=dnn_activation,
                                             l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                             init_std=init_std, device=device) for _ in range(self.num_experts)])

        # gate dnn
        if len(gate_dnn_hidden_units) > 0:
            self.gate_dnn = nn.ModuleList([DNN(self.input_dim, gate_dnn_hidden_units, activation=dnn_activation,
                                               l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                               init_std=init_std, device=device) for _ in range(self.num_tasks)])
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.gate_dnn.named_parameters()),
                l2=l2_reg_dnn)
        self.gate_dnn_final_layer = nn.ModuleList(
            [nn.Linear(gate_dnn_hidden_units[-1] if len(gate_dnn_hidden_units) > 0 else self.input_dim,
                       self.num_experts, bias=False) for _ in range(self.num_tasks)])

        # tower dnn (task-specific)
        if len(tower_dnn_hidden_units) > 0:
            self.tower_dnn = nn.ModuleList(
                [DNN(expert_dnn_hidden_units[-1], tower_dnn_hidden_units, activation=dnn_activation,
                     l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                     init_std=init_std, device=device) for _ in range(self.num_tasks)])
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.tower_dnn.named_parameters()),
                l2=l2_reg_dnn)
        self.tower_dnn_final_layer = nn.ModuleList([nn.Linear(
            tower_dnn_hidden_units[-1] if len(tower_dnn_hidden_units) > 0 else expert_dnn_hidden_units[-1], 1,
            bias=False)
                                                    for _ in range(self.num_tasks)])

        self.out = nn.ModuleList([PredictionLayer(task) for task in task_types])

        regularization_modules = [self.expert_dnn, self.gate_dnn_final_layer, self.tower_dnn_final_layer]
        for module in regularization_modules:
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], module.named_parameters()), l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        # expert dnn
        expert_outs = []
        for i in range(self.num_experts):
            expert_out = self.expert_dnn[i](dnn_input)
            expert_outs.append(expert_out)
        expert_outs = torch.stack(expert_outs, 1)  # (bs, num_experts, dim)

        # gate dnn
        mmoe_outs = []
        for i in range(self.num_tasks):
            if len(self.gate_dnn_hidden_units) > 0:
                gate_dnn_out = self.gate_dnn[i](dnn_input)
                gate_dnn_out = self.gate_dnn_final_layer[i](gate_dnn_out)
            else:
                gate_dnn_out = self.gate_dnn_final_layer[i](dnn_input)
            gate_mul_expert = torch.matmul(gate_dnn_out.softmax(1).unsqueeze(1), expert_outs)  # (bs, 1, dim)
            mmoe_outs.append(gate_mul_expert.squeeze())

        # tower dnn (task-specific)
        task_outs = []
        for i in range(self.num_tasks):
            if len(self.tower_dnn_hidden_units) > 0:
                tower_dnn_out = self.tower_dnn[i](mmoe_outs[i])
                tower_dnn_logit = self.tower_dnn_final_layer[i](tower_dnn_out)
            else:
                tower_dnn_logit = self.tower_dnn_final_layer[i](mmoe_outs[i])
            output = self.out[i](tower_dnn_logit)
            task_outs.append(output)
        task_outs = torch.cat(task_outs, -1)
        return task_outs
