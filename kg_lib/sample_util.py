import math
import numpy as np
import torch
import os

# 根据样本数目，按比例随机取样
def sample_random_index_with_portion(Processed_HAN_Data_dict, label_name, sample_size = 3000, positive_percent = 0.2):
    tmp_pos_sample_size = math.ceil(sample_size * positive_percent)
    tmp_neg_sample_size = (sample_size - tmp_pos_sample_size)

    # 获取正样本的序号
    tmp_pos_sample_index_np = np.argwhere(Processed_HAN_Data_dict[label_name] == 1).T[0]

    # 随机选取指定数目的正样本的序号
    tmp_sub_pos_sample_index_np = np.random.choice(tmp_pos_sample_index_np, size = tmp_pos_sample_size, replace = False)

    # 获取负样本的序号
    tmp_neg_sample_index_np = np.argwhere(Processed_HAN_Data_dict[label_name] == 0).T[0]

    # 随机选取指定数目的负样本的序号
    tmp_sub_neg_sample_index_np = np.random.choice(tmp_neg_sample_index_np, size = tmp_neg_sample_size, replace = False)

    # 合并两组序号
    tmp_sampled_label_index = np.concatenate((tmp_sub_pos_sample_index_np, tmp_sub_neg_sample_index_np))
    
    return tmp_sampled_label_index

# sampled_label_index = sample_random_index_with_portion(time_range_to_Processed_HAN_Data_dict[time_range_str])

# 根据取样结果，获取子图，并转Tensor
def sample_sub_graph_with_label_index(Processed_HAN_Data_dict, tmp_head_node_type, tmp_sampled_label_index, label_name, args_cuda):
    sub_graph_data_dict = {}
    sub_graph_data_dict['Feature'] = {}
    sub_graph_data_dict['Feature_Node_Type'] = {}
    sub_graph_data_dict['Adj'] = {}
    
    # 取出对应的标签、转为tensor，并看情况放入cuda
    sub_graph_data_dict['Label'] = torch.FloatTensor(Processed_HAN_Data_dict[label_name][tmp_sampled_label_index])
    if args_cuda:
        sub_graph_data_dict['Label'] = sub_graph_data_dict['Label'].cuda()
        
    # 取出对应的index号
    tmp_sampled_index = Processed_HAN_Data_dict['Label_Index'][tmp_sampled_label_index].astype(int)
    
    # 获取目标点原始特征
    tmp_head_node_feature = Processed_HAN_Data_dict['Feature'][tmp_head_node_type][tmp_sampled_index]
    sub_graph_data_dict['Feature']['src_feat'] = torch.FloatTensor(tmp_head_node_feature)
    sub_graph_data_dict['Feature_Node_Type']['src_feat'] = tmp_head_node_type
    if args_cuda:
        sub_graph_data_dict['Feature']['src_feat'] = sub_graph_data_dict['Feature']['src_feat'].cuda()
        
    # 对各元路径对应的邻接表分批处理
    for tmp_meta_path_name in Processed_HAN_Data_dict['Adj']:
        # print('处理元路径:', tmp_meta_path_name)
        
        # 只保留起始点被采样了的相关边
        tmp_sampled_adj_mask = np.isin(Processed_HAN_Data_dict['Adj'][tmp_meta_path_name]['Adj'][0], tmp_sampled_index)
        tmp_sampled_adj = Processed_HAN_Data_dict['Adj'][tmp_meta_path_name]['Adj'][:, tmp_sampled_adj_mask]
        
        # 如果总边数为0
        if tmp_sampled_adj.shape[1] == 0:
            # 保存空的邻接表（做运算时就会特征自动全给0）
            sub_graph_data_dict['Adj'][tmp_meta_path_name] = torch.LongTensor(np.array([[],[]]))
            
            if args_cuda:
                sub_graph_data_dict['Adj'][tmp_meta_path_name] = sub_graph_data_dict['Adj'][tmp_meta_path_name].cuda()
                
        else:
            # 获取涉及的全部终止点
            tmp_tail_node_index = np.unique(tmp_sampled_adj[1])
            tmp_tail_node_index.sort()

            # 获取终止点对应特征(该关系涉及的全部点)
            tmp_tail_node_type = Processed_HAN_Data_dict['Adj'][tmp_meta_path_name]['tail_type']
            tmp_tail_node_feature = Processed_HAN_Data_dict['Feature'][tmp_tail_node_type][tmp_tail_node_index]

            sub_graph_data_dict['Feature'][tmp_meta_path_name] = torch.FloatTensor(tmp_tail_node_feature)
            sub_graph_data_dict['Feature_Node_Type'][tmp_meta_path_name] = tmp_tail_node_type
            
            # 将起始点序号转化为其在全部采样点中的序号
            tmp_index_trans_dict = dict(zip(tmp_sampled_index, range(len(tmp_sampled_index))))
            tmp_head_new_index = np.vectorize(tmp_index_trans_dict.get)(tmp_sampled_adj[0])

            # 将终止点序号转化为其在全部终止点中的序号
            tmp_index_trans_dict = dict(zip(tmp_tail_node_index, range(len(tmp_tail_node_index))))
            tmp_tail_new_index = np.vectorize(tmp_index_trans_dict.get)(tmp_sampled_adj[1])

            sub_graph_data_dict['Adj'][tmp_meta_path_name] = torch.LongTensor(np.array([tmp_head_new_index, tmp_tail_new_index]))
            
            if args_cuda:
                sub_graph_data_dict['Feature'][tmp_meta_path_name] = sub_graph_data_dict['Feature'][tmp_meta_path_name].cuda()
                sub_graph_data_dict['Adj'][tmp_meta_path_name] = sub_graph_data_dict['Adj'][tmp_meta_path_name].cuda()
                
    return sub_graph_data_dict

# curr_time = datetime.now()

# sampled_label_index = sample_random_index_with_portion(time_range_to_Processed_HAN_Data_dict[time_range_str])
# sub_graph_data_dict = sample_sub_graph_with_label_index(Processed_HAN_Data_dict, Label_Data_Config_dict['Node_Type'], 
#                                                         sampled_label_index)

# curr_time2 = datetime.now()
# print(curr_time2-curr_time)



def get_sample_each_epoch(train_pos, config_dict, dl):
    train_neg = dl.get_train_neg()
    tail_nodex_info_each_iter = []
    all_labels = []
    bs = config_dict['batch_num']
    train_idx = [np.arange(len(train_pos[i][0])) for i in train_neg]
    for i in train_neg:
        np.random.shuffle(train_idx[i])
    for i in range(0, bs):
        result, labels = {}, {}
        for ind in train_pos:
            num_each_bath = len(train_pos[ind][0])//bs
            train_pos_head = np.array(train_pos[ind][0])[train_idx[ind][i:i+num_each_bath]]
            train_neg_head = np.array(train_neg[ind][0])[train_idx[ind][i:i+num_each_bath]]
            train_pos_tail = np.array(train_pos[ind][1])[train_idx[ind][i:i+num_each_bath]]
            train_neg_tail = np.array(train_neg[ind][1])[train_idx[ind][i:i+num_each_bath]]
            left = np.concatenate([train_pos_head, train_neg_head])
            right = np.concatenate([train_pos_tail, train_neg_tail])
            r_id = np.array([ind]*len(train_pos_head))
            mid = np.concatenate([r_id, r_id])
            head_tail_node_type = dl.links['meta'][ind]
            result[ind] = (left, right, mid, head_tail_node_type)
            labels[ind] = torch.FloatTensor(np.concatenate([np.ones(train_pos_head.shape[0]), 
                                                            np.zeros(train_neg_head.shape[0])])).to(config_dict['device'])
            
        tail_nodex_info_each_iter.append(result)
        all_labels.append(labels)
    return tail_nodex_info_each_iter, all_labels

def val_get_sample(train_pos, config_dict, dl):
    train_neg = dl.get_valid_neg()

    train_idx = [np.arange(len(train_pos[i][0])) for i in train_neg]
    for i in train_neg:
        np.random.shuffle(train_idx[i])
    result, labels = {}, {}
    for ind in train_pos:
        train_pos_head = np.array(train_pos[ind][0])
        train_neg_head = np.array(train_neg[ind][0])
        train_pos_tail = np.array(train_pos[ind][1])
        train_neg_tail = np.array(train_neg[ind][1])
        left = np.concatenate([train_pos_head, train_neg_head])
        right = np.concatenate([train_pos_tail, train_neg_tail])
        r_id = np.array([ind]*len(train_pos_head))
        mid = np.concatenate([r_id, r_id])
        head_tail_node_type = dl.links['meta'][ind]
        result[ind] = (left, right, mid, head_tail_node_type)
        labels[ind] = torch.FloatTensor(np.concatenate([np.ones(train_pos_head.shape[0]), 
                                                        np.zeros(train_neg_head.shape[0])])).to(config_dict['device'])
    
    return result, labels

def test_get_sample(config_dict, dl):
    test_neigh, test_label = dl.get_test_neigh()
    left, right, mid, head_tail_node_type, labels = {}, {}, {}, {}, {}
    dataset = config_dict['dataset']
    result, labels = {}, {}
    for ind in test_neigh:
        test_neigh_ind = test_neigh[ind]
        test_label_ind = test_label[ind]
        if os.path.exists(os.path.join(dl.path, f"{dataset}_ini_{ind}_label.txt")):
            save = np.loadtxt(os.path.join(dl.path, f"{dataset}_ini_{ind}_label.txt"), dtype=int)
            test_neigh_ind = [save[0], save[1]]
            if save.shape[0] == 2:
                test_label_ind = np.random.randint(2, size=save[0].shape[0])
            else:
                test_label_ind = save[2]
        left = np.array(test_neigh_ind[0])
        right= np.array(test_neigh_ind[1])
        mid = np.zeros(left.shape[0], dtype=np.int32)
        mid[:] = ind
        head_tail_node_type = dl.links['meta'][ind]
        result[ind] = (left, right, mid, head_tail_node_type)
        labels[ind] = torch.FloatTensor(test_label_ind).to(config_dict['device'])
    return result, labels