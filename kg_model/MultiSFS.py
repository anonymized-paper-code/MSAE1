import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from torch.autograd import grad

    
    
class MultiSFS(nn.Module):

    def __init__(self):
        super(MultiSFS, self).__init__()
        self.feat_imp_binary = defaultdict(dict)
        
    # TODO: 看不同initialize parameter下，feature selection结果是否会差很多 (会相差很多)
    def feat_select(self, config, mtl_model, dataloader, loss_fun):

        Target_Node_Type = config['Target_Node_Type']

        # load_data
        if config['dataset'] == 'JD':
            month_list = dataloader.KG_time_list['train']
            time_range_str = ('Time_Range:' + str(month_list[0%len(month_list)]))
            _, G = dataloader.load_data(time_range_str, config['train_sample_size'], config['train_pos_sample_percent'])
        elif config['dataset'] == 'Aminer' or config['dataset'] == 'DBLP':
            G = dataloader.hg_dgl

        feat_imp = defaultdict(dict)
        feat_mask = {}
        with G.local_scope():
            # Single-shot Feature importance calculation
            for ntype in G.ntypes:
                feat_mask[ntype] = torch.ones([G.num_nodes(ntype), config['node_feature_hid_len']], requires_grad = True).cuda()
                for task_name in config['task_name'] + ['share']:
                    # create mask with gradient
                    G.nodes[ntype].data['feat_mask_' + task_name] = feat_mask[ntype]

            h_output = mtl_model(G)
            for task_name in config['task_name']:
                if config['dataset'] == 'JD':
                    h_output_squeeze = torch.squeeze(h_output[task_name])

                    tmp_true_label = G.nodes[Target_Node_Type].data[task_name]
                    tmp_true_label_index = (tmp_true_label != -1).nonzero()

                    # 只保留指定index的数据
                    h_output_squeeze_i = h_output_squeeze[tmp_true_label_index]
                    tmp_true_label_i = tmp_true_label[tmp_true_label_index]

                    loss_i = loss_fun(h_output_squeeze_i, tmp_true_label_i)

                elif config['dataset'] == 'Aminer' or config['dataset'] == 'DBLP':
                    output = h_output[task_name]

                    target_node_name = config['task_to_node'][task_name]
                    mask = G.nodes[target_node_name].data['train_label_mask']
                    tmp_true_label = G.nodes[target_node_name].data['train_label'][mask,:]

                    # 只保留指定index的数据
                    output = output[mask]

                    loss_i = loss_fun(output, tmp_true_label)

                for ntype in G.ntypes: 
                    imp = grad(loss_i, feat_mask[ntype], retain_graph=True)[0].mean(0)
                    feat_imp[ntype][task_name] = (imp-imp.min())/(imp.max()-imp.min())
                    
                    

            # Field-wise Task Relation calculation
    #         for task_name in config['task_name']:
    #             if config['dataset'] == 'JD':
    #                 target_node_name = config['task_to_node'][task_name]
    #                 tmp_true_label = G.nodes[Target_Node_Type].data[task_name]
    #                 tmp_true_label_index = (tmp_true_label != -1).nonzero()


    #                 tmp_true_label_i = tmp_true_label[tmp_true_label_index]
    #             elif config['dataset'] == 'Aminer' or config['dataset'] == 'DBLP':
    #                 target_node_name = config['task_to_node'][task_name]
    #                 mask = hg_dgl.nodes[target_node_name].data['train_label_mask']
    #                 tmp_true_label = hg_dgl.nodes[target_node_name].data['train_label'][mask,:]

    #                 G.nodes[target_node_name].data['feat_emb'][tmp_true_label==1]


            # Combination
            k = int(config['topk_ratio']*config['node_feature_hid_len'])
            for ntype in feat_imp:
                if 'PLE_SimpleHGN' in config['model'] or 'Our' in config['model']:
                    self.feat_imp_binary[ntype]['share'] = torch.ones_like(feat_imp[ntype][task_name])
                elif 'MMOE' in config['model']:
                    self.feat_imp_binary[ntype]['share'] = torch.zeros_like(feat_imp[ntype][task_name])
                for task_name in feat_imp[ntype]:
                    # select top k feature
                    _, indices = torch.topk(feat_imp[ntype][task_name], k)
                    self.feat_imp_binary[ntype][task_name] = torch.zeros_like(feat_imp[ntype][task_name])
                    self.feat_imp_binary[ntype][task_name][indices] = 1
                    if 'PLE_SimpleHGN' in config['model'] or 'Our' in config['model']:
                        self.feat_imp_binary[ntype]['share'] *= self.feat_imp_binary[ntype][task_name]
                    elif 'MMOE' in config['model']:
                        self.feat_imp_binary[ntype]['share'] += self.feat_imp_binary[ntype][task_name]
                        self.feat_imp_binary[ntype]['share'][self.feat_imp_binary[ntype]['share']>0] = 1.
                
                    
    
    # TODO: 不reaturn是否会修改G
    def add_feat_imp(self, G):
        for ntype in self.feat_imp_binary:
            for task_name in self.feat_imp_binary[ntype]:
                # create mask with gradient
                G.nodes[ntype].data['feat_mask_' + task_name] = self.feat_imp_binary[ntype][task_name].unsqueeze(0).repeat(G.num_nodes(ntype), 1)


        