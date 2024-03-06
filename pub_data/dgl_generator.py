
# 该文件用于对公开数据集进行预处理，并读取为dgl支持的异构图形式
# 数据集放在http://pan.jd.com/sharedInfo/EEB092600F088546EEC946AFBC5707A7下，可自行下载解压使用
# 公开的异构图数据集整理见 https://docs.google.com/spreadsheets/d/146cSQCa9dgjP0BBy6icjaxN5xJKsQ9er-cKRke7cpVE/edit#gid=1204053846 （还在更新中，欢迎大家随时更新补充
# 对每个数据集详细的分析见 https://docs.google.com/document/d/1nOy0_-YvDQU0WlXFTNdnr2oK5932H11H-icEHy3ic4c/edit （还在更新中，欢迎大家随时更新补充


import dgl
import copy
import numpy as np 
import torch
from scipy import io
import os
import pickle
from collections import defaultdict


def load_dgl_data(config):
    name = config['dataset']
    if name == 'DBLP':
        return DBLP(config)
    elif name == 'Aminer':
        return Aminer(config)
    else:
        raise NameError(f"Do not support the dataset {name}. Please try DBLP/Aminer/Amazon/Yelp/JD.")


# NC
class DBLP:
    def __init__(self, config):
        from pub_data.DBLP_data_loader import DataLoader
        data_path = '../../Data/public_data/DBLP'
        data_loader = DataLoader(data_path)
        # load features
        node_type_ind = ['author', 'paper', 'phase', 'venue']
        feature = {}
        num_nodes = {}
        node_shift = {}
        self.node_type_to_feature_len_dict = {}
        for ind, name in enumerate(node_type_ind):
            feature[name] =  data_loader.nodes['attr'][ind]
            if feature[name] is None:
                feature[name] = torch.eye(data_loader.nodes['count'][ind])
            num_nodes[name] =  data_loader.nodes['count'][ind]
            node_shift[name] =  data_loader.nodes['shift'][ind]
            self.node_type_to_feature_len_dict[name] = feature[name].shape[1]
        
        
        # load adjs
        adj_dict = {}
        node_edge_tuple = [('author','a-p','paper'), ('paper','p-ph','phase'),
                          ('paper','p-v','venue'), ('paper','p-a','author'),
                          ('phase','ph-p','paper'), ('venue','v-p','paper')]
        for ind, name in enumerate(node_edge_tuple):
            head_node_shift = node_shift[name[0]]
            tail_node_shift = node_shift[name[-1]]
            adj_dict[name] = (data_loader.links['data'][ind][:,0]-head_node_shift, data_loader.links['data'][ind][:,1]-tail_node_shift)
    #         print(f'{name}:{ adj_dict[name][0].max()},{ adj_dict[name][1].max()}')
        
        if config['self_loop']:
            adj_dict[('author','a-a','author')] = (torch.arange(0, num_nodes['author']), torch.arange(0, num_nodes['author']))
            adj_dict[('paper','p-p','paper')] =  (torch.arange(0, num_nodes['paper']), torch.arange(0, num_nodes['paper']))
          
       # load label
        ## train，val, test用的都是一张图，区别是标注的label不同
        ##  DBLP在整张图上运行时，每一次都用全部节点和label监督训练
        # # 添加标签及标签对应的节点序号
        ## TODO: 根据正负样本比例对val和train进行拆分
        a_train_data = data_loader.labels_train['data'][node_shift['author']:node_shift['author']+num_nodes['author']]
        a_test_data = data_loader.labels_test['data'][node_shift['author']:node_shift['author']+num_nodes['author']]
        author_label = torch.FloatTensor(a_train_data + a_test_data)
        
        # add paper-venue label based on p-v link
        split_data_path = os.path.join(data_path, 'label_mask_split.npy')
        paper_label = torch.nn.functional.one_hot(adj_dict[('paper','p-v','venue')][1], num_classes=num_nodes['venue']).float()
        paper_label_num = num_nodes['paper']
        del adj_dict[('paper','p-v','venue')]
        del adj_dict[('venue','v-p','paper')]
        del num_nodes['venue']
        del feature['venue']
        
        # load label mask

        # paper mask
        train_ratio = 0.6
        val_ratio = 0.2
        idx = np.arange(0, paper_label_num)
        np.random.shuffle(idx)
        train_idx, val_idx, test_idx = np.array_split(idx, [int(train_ratio*paper_label_num), int((val_ratio+train_ratio)*paper_label_num)])
        mask = torch.zeros(paper_label_num, dtype=torch.bool)
        p_train_mask, p_val_mask, p_test_mask = copy.deepcopy(mask), copy.deepcopy(mask), copy.deepcopy(mask)
        p_train_mask[train_idx] = True
        p_val_mask[val_idx] = True
        p_test_mask[test_idx] = True
            
         
        # author mask
        a_train_mask = data_loader.labels_train['mask'][node_shift['author']:node_shift['author']+num_nodes['author']]
        a_test_mask = data_loader.labels_test['mask'][node_shift['author']:node_shift['author']+num_nodes['author']]
        val_ratio = 0.2
        train_idx = np.nonzero(a_train_mask)[0] #
        np.random.shuffle(train_idx)
        split = int(train_idx.shape[0]*val_ratio)
        a_train_mask_select = copy.deepcopy(a_train_mask)
        a_train_mask_select[train_idx[:split]] = False
        a_val_mask = copy.deepcopy(a_train_mask)
        a_val_mask[train_idx[split:]] = False

        author_mask = [a_train_mask_select, a_val_mask, a_test_mask]
        paper_mask = [p_train_mask, p_val_mask, p_test_mask]
        np.save(split_data_path, [author_mask, paper_mask], allow_pickle=True)

        author_mask  = [torch.BoolTensor(x) for x in author_mask]
        paper_mask  = [torch.BoolTensor(x) for x in paper_mask]

        # build graph
        hg_dgl = dgl.heterograph(adj_dict, num_nodes_dict = num_nodes)
        hg_dgl.ndata['feat'] = feature

        ## TODO: 将train_label和tast_label合并到一起
        for ind, i in enumerate(['train', 'val', 'test']):
            hg_dgl.ndata[i+"_label"] = {'author': author_label, 'paper':paper_label}
            hg_dgl.ndata[i+"_label_mask"] = {'author': author_mask[ind], 'paper': paper_mask[ind]}
            
#         hg_dgl.ndata["train_label_mask"] = {'author': torch.BoolTensor(train_mask_select)}
#         hg_dgl.ndata["val_label"] = {'author': torch.FloatTensor(train_data)}
#         hg_dgl.ndata["val_label_mask"] = {'author': torch.BoolTensor(val_mask)}
#         hg_dgl.ndata["test_label"] = {'author': torch.FloatTensor(test_data)}
#         hg_dgl.ndata["test_label_mask"] = {'author': torch.BoolTensor(test_mask)}
        
        
        self.hg_dgl = hg_dgl
        if config['args_cuda']:
            self.hg_dgl = self.hg_dgl.to(config['device'])
        
        self.node_dict = {}
        self.edge_dict = {}
        for ind, ntype in enumerate(hg_dgl.ntypes):
            self.node_dict[ntype] = ind
        for ind, etype in enumerate(hg_dgl.etypes):
            self.edge_dict[etype] = ind
            
        self.edge_type_each_node = defaultdict(list)
        for srctype, etype, dsttype in hg_dgl.canonical_etypes:
            self.edge_type_each_node[srctype].append(self.edge_dict[etype])
        
#     # 补全反向的边
#     hg_dgl = add_reverse_hetero(hg_dgl)


# NC
class Aminer:
    def __init__(self, config):
        data_path = '../../Data/public_data/'
        split_data_path = os.path.join(data_path, 'Aminer_split.npy')
        full_data_path = os.path.join(data_path, 'Aminer.mat')
        data = io.loadmat(full_data_path)

        # load feature
        feature = {}
        num_nodes = {}
        self.node_type_to_feature_len_dict = {}
        ## TODO: 数据标准化
        feature['paper'] = torch.FloatTensor(normalization(data["PvsF"]))
        feature['author'] = torch.FloatTensor(normalization(data["AvsF"]))
    #     import ipdb
    #     ipdb.set_trace() 
        for node in feature:
            self.node_type_to_feature_len_dict[node] =  feature[node].shape[1]
        num_nodes['paper'] = data["PvsF"].shape[0]
        num_nodes['author'] = data["AvsF"].shape[0]

        # load adj
        adj_dict = {}
        adj_dict[('paper','p-a','author')] = convert_csc_to_value(data["PvsA"])
        adj_dict[('author','a-p','paper')] = convert_csc_to_value(data["PvsA"].transpose())
        adj_dict[('paper','p-p','paper')] = convert_csc_to_value(data["PvsP"], config['self_loop'])
        adj_dict[('author','a-a','author')] = convert_csc_to_value(data["AvsA"], config['self_loop'])
        
            
        author_label = torch.FloatTensor(data["AvsC"].toarray())
        paper_label = torch.FloatTensor(data["PvsC"].toarray())


        def sample_train_test_val(num, val_ratio=0.1, test_ratio=0.1):
            train_ratio = 1 - val_ratio - test_ratio
            idx = np.arange(0, num)
            np.random.shuffle(idx)
            train_idx, val_idx, test_idx = np.array_split(idx, [int(train_ratio*num), int((val_ratio+train_ratio)*num)])

            mask = torch.zeros(num, dtype=torch.bool)
            train_mask, val_mask, test_mask = copy.deepcopy(mask), copy.deepcopy(mask), copy.deepcopy(mask)
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True
            return train_mask, val_mask, test_mask


        if os.path.exists(split_data_path):
            split_data = np.load(split_data_path, allow_pickle=True)
            split_data = split_data.tolist()
            author_mask, paper_mask = split_data[0], split_data[1]
            author_mask  = [torch.tensor(x) for x in author_mask]
            paper_mask  = [torch.tensor(x) for x in paper_mask]
        else:
            # load label 
            val_ratio, test_ratio = 0.1, 0.1
            author_mask = sample_train_test_val(author_label.shape[0], val_ratio, test_ratio)
            paper_mask = sample_train_test_val(paper_label.shape[0], val_ratio, test_ratio)

    #     from IPython.core.debugger import set_trace
    #     set_trace() 
            author_mask = [x.numpy() for x in author_mask]
            paper_mask = [x.numpy() for x in paper_mask]
            save_path = '../../Data/public_data/Aminer_split.npy'
            np.save(save_path, [author_mask, paper_mask], allow_pickle=True)
#             aa = np.load(save_path, allow_pickle=True)
#             aa = aa.tolist()


    #     ap = adj_dict[('author','a-p','paper')]
    #     author_train_mask = np.where(author_mask[0])[0]
    #     paper_label_mask = data["PvsC"].toarray()
    #     paper_label_mask[paper_mask[0], :] = [0,0,0,0]
    #     dif = []
    #     for author_id in author_train_mask:
    #         ind = np.where(ap[0]==author_id)[0]
    #         paper_id = ap[1][ind]
    #         author_label = data["AvsC"][author_id,:].toarray()
    #         paper_label = np.any(paper_label_mask[paper_id,:], 0)
    #         dif.append(np.sum(np.abs(author_label-paper_label)))
    #     print(np.sum(dif))

        # build graph
        hg_dgl = dgl.heterograph(adj_dict, num_nodes_dict = num_nodes)
        hg_dgl.ndata['feat'] = feature

        hg_dgl.ndata["train_label"] = {'author': author_label, 'paper': paper_label}
        hg_dgl.ndata["val_label"] = {'author': author_label, 'paper': paper_label}
        hg_dgl.ndata["test_label"] = {'author': author_label, 'paper': paper_label}
        hg_dgl.ndata["train_label_mask"] = {'author': author_mask[0], 'paper': paper_mask[0]}
        hg_dgl.ndata["val_label_mask"] = {'author': author_mask[1], 'paper': paper_mask[1]}
        hg_dgl.ndata["test_label_mask"] = {'author': author_mask[2], 'paper': paper_mask[2]}
        
        
         
        self.hg_dgl = hg_dgl
        if config['args_cuda']:
            self.hg_dgl = self.hg_dgl.to(config['device'])
        
        
        self.node_dict = {}
        self.edge_dict = {}
        for ind, ntype in enumerate(hg_dgl.ntypes):
            self.node_dict[ntype] = ind
        for ind, etype in enumerate(hg_dgl.etypes):
            self.edge_dict[etype] = ind
            
        self.edge_type_each_node = defaultdict(list)
        for srctype, etype, dsttype in hg_dgl.canonical_etypes:
            self.edge_type_each_node[srctype].append(self.edge_dict[etype])
        




    
def convert_csc_to_value(csc_matrix, self_loop=False):
    coo_matrix = csc_matrix.tocoo()
    if self_loop:
        coo_matrix.setdiag(1)
    row, col = coo_matrix.row, coo_matrix.col
    return (row, col)

def normalization(data):
    mean = np.mean(data, 0, keepdims=True)
    std = np.std(data, 0, keepdims=True)
    norm_data = (data - mean) / std
    return norm_data

# 对部分关系，加入反转head和tail后的边
def add_reverse_hetero(g):
    relations = {}
    num_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
    
    for metapath in g.canonical_etypes:
        # Original edges
        src, dst = g.all_edges(etype=metapath[1])
        relations[metapath] = (src, dst)

        reverse_metapath = (metapath[2], metapath[1] + '_by', metapath[0])
        assert reverse_metapath not in relations
        relations[reverse_metapath] = (dst, src)           # Reverse edges

    new_g = dgl.heterograph(relations, num_nodes_dict = num_nodes_dict)

    # copy_ndata:
    for ntype in g.ntypes:
        for k, v in g.nodes[ntype].data.items():
            new_g.nodes[ntype].data[k] = v.detach().clone()
    
    
    return new_g



    
if __name__ == '__main__':
#     load_Aminer('../../../Data/public_data/')