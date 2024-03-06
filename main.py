#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import pandas as pd
from collections import defaultdict
import math
import os
import io
import time
import copy
import re
import gc
import sys
# sys.path.append("..")

import random
from random import sample
import numpy as np
import pandas as pd

import scipy.sparse as sp
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.stats import norm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
import math

import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import normalize

from tqdm import tqdm

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt


from kg_lib.args import load_parameter
import argparse


from kg_lib.utils import mkdir
import logging
from kg_lib.logging_util import init_logger

from pub_data.dgl_generator import load_dgl_data
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth',100)

plt.rcParams["font.sans-serif"]=["SimHei"] 
plt.rcParams["axes.unicode_minus"]=False 



#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "0"



# 固定随机值
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed = 4200
setup_seed(seed)



# In[6]:




parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
args = parser.parse_args()
config_name = args.config


config = load_parameter(f'kg_config/{config_name}.yaml')



loss_weight_decay = False
if 'loss_weight_decay' in config and config['loss_weight_decay']:
    loss_weight_decay = True
decay_value = 0.00
intial_value = 0.99


# # 处理数据

# In[7]:


# In[8]:


# In[9]:



localtime = time.strftime("%m-%d-%H:%M:%S", time.localtime())


tmp_model_parameter_output_dir = "../../Model_Parameter/" + config['model'] + "_" + config['dataset'] + "/" + localtime
mkdir('../../Model_Parameter')
mkdir("../../Model_Parameter/" + config['model'] + "_" + config['dataset'])
mkdir(tmp_model_parameter_output_dir)
print(tmp_model_parameter_output_dir)

record_result_save_path = os.path.join(tmp_model_parameter_output_dir, 'record')

log_filename = os.path.join(tmp_model_parameter_output_dir, 'log.txt')
init_logger('', log_filename, logging.INFO, False)
logging.info(config)

metric_list_dict = {}
metric_list_dict['train'] = defaultdict(lambda: defaultdict(list))
metric_list_dict['val'] = defaultdict(lambda: defaultdict(list))
metric_list_dict['test'] = defaultdict(lambda: defaultdict(list))

best_result_save_file_name = os.path.join(tmp_model_parameter_output_dir, 'best_result.json')

BCE_loss = torch.nn.BCELoss()


# In[10]:


def get_model(config):
    if config['model'] == 'SimpleHGN':
        from kg_model.SimpleHGN import SimpleHGN
        return SimpleHGN(config)
    elif config['model'] == 'HGT':
        from kg_model.HGT import HGT
        return HGT(config)
    elif 'PLE_SimpleHGN' in config['model'] or 'MMOE' in config['model']:
        from kg_model.ple_simplehgn import PLE
        return PLE(shared_expert_num=config['shared_expert_num'], specific_expert_num=config['specific_expert_num'], 
                                  num_levels=config['num_levels'],
                             expert_dnn_hidden_units=(256, 128), gate_dnn_hidden_units=(64,), tower_dnn_hidden_units=(),
                             l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                             dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, 
                             task_names=config['task_name'], device='cuda:0', gpus=None, config_dict=config)

    elif 'Our' in config['model']:
        from kg_model.our import Our1
        return Our1(shared_expert_num=config['shared_expert_num'], specific_expert_num=config['specific_expert_num'], 
                                  num_levels=config['num_levels'],
                             expert_dnn_hidden_units=(256, 128), gate_dnn_hidden_units=(64,), tower_dnn_hidden_units=(),
                             l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                             dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False,
                             task_names=config['task_name'], device='cuda:0', gpus=None, config_dict=config)

    
# In[11]:

from kg_lib.visual_util import Recoders
from kg_lib.evaluation_util import evaluation
from kg_model.MultiSFS import MultiSFS
Target_Node_Type = config['Target_Node_Type']
args_cuda = config['args_cuda']

evaluation_func = evaluation(config['dataset'])

def train(multi_train, tt):
    
    if loss_weight_decay:
        loss_weight = {}
        loss_weight['paper'] = intial_value
        loss_weight['author'] = 1- loss_weight['paper']
    else:
        loss_weight = None
        

    # 建立模型
    model = get_model(config)
    if args_cuda:
        model.cuda()
        
    if 'MultiSFS' in config['model']:
        multisfs = MultiSFS()
        multisfs.feat_select(config, model, dataloader, BCE_loss)
        if config['dataset'] == 'Aminer' or config['dataset'] == 'DBLP':
            hg_dgl = dataloader.hg_dgl  
            if 'MultiSFS' in config['model']:
                multisfs.add_feat_imp(hg_dgl)
        torch.cuda.empty_cache()
    else:
        multisfs = None

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr = config['learning_rate'],
                                weight_decay = config['weight_decay'])
    
    # 最优roc_auc
    if config['save_metric'] == 'loss':
        best_roc_auc = [999] * len(config['task_name'])
    else:
        best_roc_auc = [0] * len(config['task_name'])
    # 累计未优化次数
    early_stop_count = 0
    print_result_best = ''

    metric_list_dict_single_ep = defaultdict(lambda: defaultdict(dict))
    metric_list_dict_single_ep_best = None
    
    ## 生成中间参数记录器
    recorders = Recoders(config, tt, record_result_save_path) if len(config['recorder_type'])>0 else None


    for epoch in range(config['train_epoch']):
        
        model.train()
    
        train_per_epoch(config, model, dataloader, optimizer, multi_train, recorders, multisfs, loss_weight)
        
        if loss_weight_decay:
            if epoch%10 == 0:
                loss_weight['paper'] = np.max([0.5, loss_weight['paper']-decay_value])
                loss_weight['author'] = 1-loss_weight['paper']
                print('paper:', loss_weight['paper'])

        if 'course_learning' in config and config['course_learning']:
            if epoch>0 and epoch%config['course_learning_ep'] == 0:
                if model.course_learning_weight<0:
                    model.course_learning_weight = 0
                else:
                    model.course_learning_weight += config['course_learning_add']
                model.course_learning_weight = np.min([model.course_learning_weight, 1])
        
        
        if epoch%config['eval_inter_ep'] == 0:
            # 查看效果
            print('Epoch:', epoch)
            print_result = ''
            for data_type in ['train', 'val', 'test']:
                result = evaluation_func(model, dataloader, data_type, config)
                print_result += f'\n{data_type} - '
                for label_name in result:
                    for metric in result[label_name]:
                        metric_value = result[label_name][metric]
                        metric_list_dict[data_type][label_name][metric].append(metric_value)
                        metric_list_dict_single_ep[data_type][label_name][metric] = metric_value
                        if (type(metric_value)!=dict) and (type(metric_value)!=list):
                            print_result += f'{label_name}_{metric}:{metric_value:.4f}, '
                        else:
                            print_result += f'{label_name}_{metric}:{metric_value}, '
                    print_result += '\n'


            logging.info(f'Epoch:{epoch}')
            logging.info(print_result)
            if not multi_train:
                print(print_result)


            # 达到最优效果时存储一次模型
            val_roc_auc = [metric_list_dict['val'][label_name][config['save_metric']][-1] for label_name in config['task_name']]
            
            if config['save_metric'] == 'loss':
                dif = np.array(best_roc_auc)-np.array(val_roc_auc)
            else:
                dif = np.array(val_roc_auc)-np.array(best_roc_auc)
            if np.all(dif>=0):
                early_stop_count = 0
                best_roc_auc = val_roc_auc
                print_result_best = print_result
                metric_list_dict_single_ep_best = copy.deepcopy(metric_list_dict_single_ep)
                model_save_name = 'model_parameter_best_roc_auc_'
                for ind, label_name in enumerate(config['task_name']):
                    model_save_name += f'_{label_name}-{best_roc_auc[ind]:.4f}'
                if multi_train:
                    torch.save(model.state_dict(), os.path.join(tmp_model_parameter_output_dir, f'best_model_{tt}.pt'))
                else:
                    torch.save(model.state_dict(), os.path.join(tmp_model_parameter_output_dir, model_save_name+'.pt'))
            else:
                early_stop_count = early_stop_count + 1
                if not multi_train:
                    print("Early Stop Count:", early_stop_count)
                logging.info(f"Early Stop Count:{early_stop_count}")

                if early_stop_count >= config['early_stop']:
                    return print_result_best, metric_list_dict_single_ep_best, metric_list_dict
                
    return print_result_best, metric_list_dict_single_ep_best, metric_list_dict


def train_per_epoch(config, model, dataloader, optimizer, multi_train, recorders, multisfs, loss_weight):
    if config['dataset'] == 'Aminer' or config['dataset'] == 'DBLP':
        
        hg_dgl = dataloader.hg_dgl

        if multi_train:
            pbar = range(config['iter_num'])
        else:
            pbar = tqdm(range(config['iter_num']))
 
        for sample_index in pbar:
            # 再训练模型
            loss = 0

            h_output = model(hg_dgl, recorders)
            result_pbar = {}
            
            if recorders is not None: recorders.update() 
                
            # 提取label和mask
            ## TODO: 把target_node_name改为task name
            for ind, task_name in enumerate(h_output):
                output = h_output[task_name]
                
                target_node_name = config['task_to_node'][task_name]
                mask = hg_dgl.nodes[target_node_name].data['train_label_mask']
                tmp_true_label = hg_dgl.nodes[target_node_name].data['train_label'][mask,:]

                # 只保留指定index的数据
                output = output[mask]

                # 正则化
                loss_i = BCE_loss(output, tmp_true_label)
                if ('try_uncertainty' in config) and (config['try_uncertainty']):
                    weight_i = model.uncertainty_weight[task_name]
                    loss += torch.exp(-weight_i)*loss_i + weight_i
                else:
                    if loss_weight_decay:
                        loss += loss_weight[task_name]*loss_i
                    else:
                        loss += loss_i

                # 查看其他指标
                if args_cuda:
                    source_data_label_np = tmp_true_label.data.cpu().numpy()
                    h_output_squeeze_np = output.data.cpu().numpy()
                else:
                    source_data_label_np = tmp_true_label.data.numpy() 
                    h_output_squeeze_np = output.data.numpy()

                task_type = config['task_type'][task_name]
                if task_type == 'single-label':
                    h_output_squeeze_np = h_output_squeeze_np.argmax(axis=1)
                    source_data_label_np = source_data_label_np.argmax(axis=1)
                elif task_type == 'multi-label':
                    h_output_squeeze_np = (h_output_squeeze_np>0.5).astype(int)
                else:
                    raise NameError(f'task type {task_type} does not exist!')

                micro = f1_score(source_data_label_np, h_output_squeeze_np, average='micro')
                macro = f1_score(source_data_label_np, h_output_squeeze_np, average='macro')

                result_pbar[task_name + '_loss'] = loss_i.item()
                result_pbar[task_name + '_micro'] = micro
                result_pbar[task_name + '_macro'] = macro

            if not multi_train:
                pbar.set_postfix(result_pbar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
     
        


# In[ ]:


import copy
import json
rerun_num = config['rerun_num']
multi_train = True if rerun_num>1 else False
metric_list_dict_best = []
for tt in range(rerun_num):
    print('Loading data......')
    dataloader = load_dgl_data(config)
    config['edge_dict'] = dataloader.edge_dict
    config['node_dict'] = dataloader.node_dict
    config['node_type_to_feature_len_dict'] = dataloader.node_type_to_feature_len_dict
    config['edge_type_each_node'] = dataloader.edge_type_each_node

    print_result_best, metric_list_dict_single_ep, metric_list_dict = train(multi_train, tt)
    logging.info("\n\n\n\n-----------------------------------------------------------------\n\n\n\n")
    logging.info("\n----------------------------Result-------------------------------------")
    logging.info(print_result_best)
    metric_list_dict_best.append(copy.deepcopy(metric_list_dict_single_ep))
    with open(best_result_save_file_name, 'w') as outfile:
        json.dump(metric_list_dict_best, outfile)
    with open(os.path.join(tmp_model_parameter_output_dir, f'metric_record_{tt}.json'), 'w') as outfile:
        json.dump(metric_list_dict, outfile)    
        
    logging.info("----------------------------------------------------------------")
                 


# In[ ]:






# In[ ]:




