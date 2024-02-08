from collections import defaultdict
import os
import torch
import pickle
import matplotlib.pyplot as plt
import copy

from .utils import mkdir


class Recoders():
    def __init__(self, config, tt, save_path):
        self.recorder_dict = {}
        recorder_type = config['recorder_type']
        for i in recorder_type:
            if i == 'gate':
                self.recorder_dict['gate_weight_record'] = RecordGate(save_path, tt, 5, 10000)
            elif i == 'att':
                self.recorder_dict['att_record'] = RecordAtt(save_path, tt, 5, 10000)
    
    def update(self):
        for name in self.recorder_dict:
            self.recorder_dict[name].save()
        
        
class RecordGate():
    def __init__(self, save_path, run_num_th, inter=5, max_time=10000):
        self.gate_outs_dict = defaultdict(dict)
        self.inter = inter
        self.max_time = max_time
        self.time_count = 0
        self.if_record = True
        self.label_name = None
        self.run_num_th = run_num_th
        self.clear_inter = 5
        self.record_time = 0
        self.reset_time = 0 
                  
        self.save_path = os.path.join(save_path, 'gate_weight')
        mkdir(self.save_path)
    
#     def check_record(self, epoch, label_name):
#         if (epoch%self.inter == 0) & (epoch<=self.max_ep):
#             self.if_record = True
#             self.label_name = label_name
#         else:
#             self.if_record = False
        
    ## TOOD: 可能会出现存储的数据过大的情况，考虑改为分批保存      
    def record(self, gate_outs, level_num, task):  
        if task not in self.gate_outs_dict[level_num]:
            self.gate_outs_dict[level_num][task] = []
        self.gate_outs_dict[level_num][task].append(gate_outs)
            
        # 对于第一层，gate的格式为:
        # mlp, transformer: for task [N, 2],  for share [N, task_num+1]
        # para: for task [2], for share [task_num+1]
        # graph: []
        # 对于第二层，gate的格式为: 
        # graph: []

    def save(self):
        if (self.time_count%self.inter == 0) & (self.time_count<=self.max_time):
            self.record_time += 1
            self.if_record = True
            with open(os.path.join(self.save_path, f"val_gate_outs_list_all_{self.run_num_th}_{self.reset_time}"), "wb") as fp:
                pickle.dump(self.gate_outs_dict, fp)
            # reset recorder
            if self.record_time%self.clear_inter == 0:
                self.gate_outs_dict = defaultdict(dict)
                self.reset_time += 1
        else:
            self.if_record = False
        self.time_count += 1
        
        
class RecordAtt():
    def __init__(self, save_path, run_num_th, inter=5, max_time=10000):
        self.gate_outs_dict = defaultdict(dict)
        self.inter = inter
        self.max_time = max_time
        self.time_count = 0
        self.if_record = True
        self.label_name = None
        self.run_num_th = run_num_th
        self.clear_inter = 5
        self.record_time = 0
        self.reset_time = 0 
                  
        self.save_path = os.path.join(save_path, 'att_weight')
        mkdir(self.save_path)
    
#     def check_record(self, epoch, label_name):
#         if (epoch%self.inter == 0) & (epoch<=self.max_ep):
#             self.if_record = True
#             self.label_name = label_name
#         else:
#             self.if_record = False
        
    ## TOOD: 可能会出现存储的数据过大的情况，考虑改为分批保存      
    def record(self, gate_outs, level_num, task):  
        if task not in self.gate_outs_dict[level_num]:
            self.gate_outs_dict[level_num][task] = []
        self.gate_outs_dict[level_num][task].append(gate_outs)
            
        # 对于第一层，gate的格式为:
        # mlp, transformer: for task [N, 2],  for share [N, task_num+1]
        # para: for task [2], for share [task_num+1]
        # graph: []
        # 对于第二层，gate的格式为: 
        # graph: []

    def save(self):
        if (self.time_count%self.inter == 0) & (self.time_count<=self.max_time):
            self.record_time += 1
            self.if_record = True
            with open(os.path.join(self.save_path, f"val_gate_outs_list_all_{self.run_num_th}_{self.reset_time}"), "wb") as fp:
                pickle.dump(self.gate_outs_dict, fp)
            # reset recorder
            if self.record_time%self.clear_inter == 0:
                self.gate_outs_dict = defaultdict(dict)
                self.reset_time += 1
        else:
            self.if_record = False
        self.time_count += 1
        
        

class RecordLossWeight():
    def __init__(self, save_path, inter=10, save_inter=5):
        self.weight_record = defaultdict(list)
        self.inter = inter
        self.save_inter = save_inter
                  
        self.save_path = os.path.join(save_path, 'loss_weight')
        mkdir(self.save_path)
                  
    def record(self, epoch, label_name, weight_i):                  
        self.weight_record[label_name].append(weight_i.item())
        if epoch%self.save_inter == 0:          
            with open(os.path.join(self.save_path, f"loss_weight"), "wb") as fp:
                pickle.dump(self.weight_record, fp)
                      
                    
class RecordGrad():
    def __init__(self, config_dict, save_path, inter=10, save_inter=5):
        self.grad_save_all = []
        self.save_dict = {}
        self.save_dict['size_dif'] = defaultdict(list)
        self.save_dict['sign_save'] = defaultdict(list)
        self.save_dict['ht_save'] = defaultdict(list)
        self.save_dict['direct_dif'] = defaultdict(list)
        self.inter = inter
        self.save_inter = save_inter
    
        self.metapath_grad_save_all = defaultdict(list)
        
        self.try_metapath_weight = config_dict['try_metapath_weight']
        
        self.save_path = os.path.join(save_path, 'grad_record')
        mkdir(self.save_path)
        self.direction_dif_path = os.path.join(save_path, 'direction_dif')
        self.size_dif_path = os.path.join(save_path, 'size_dif')
        mkdir(self.direction_dif_path)
        mkdir(self.size_dif_path)
    
    def record_per_task(self, model, optimizer, loss, loss_sum, sample_index):
        if (sample_index%self.inter == 0) & (sample_index>0):
            grad_save = {}
            for name, parameter in model.named_parameters():
                # 只关注共享层
                if ('label' not in name) and ('uncertainty' not in name):
                    grad_save[name] = torch.autograd.grad(loss, parameter, retain_graph=True)[0].clone()
                if self.try_metapath_weight:
                    if 'task_metapath_weight.'+label_name in name:
                        self.metapath_grad_save_all[label_name].append(torch.autograd.grad(loss_sum, parameter, retain_graph=True)[0].clone())
                optimizer.zero_grad()   
                self.grad_save_all.append(grad_save)
    
    ### 分析不同task之间grad的区别
    def task_grad_diff(self, sample_index, epoch):
        if (sample_index%self.inter == 0) & (sample_index>0):
            for name in self.grad_save_all[0].keys():
                sign = self.grad_save_all[0][name]
                ht = self.grad_save_all[1][name]
                dif1 = torch.mean(torch.abs(torch.abs(sign) - torch.abs(ht)))
                dif2 = F.cosine_similarity(sign.view(1,-1), ht.view(1,-1))[0]
                self.save_dict['size_dif'][name].append(dif1.cpu().data)
                self.save_dict['sign_save'][name].append(torch.mean(torch.abs(sign)).cpu().data)
                self.save_dict['ht_save'][name].append(torch.mean(torch.abs(ht)).cpu().data)
                self.save_dict['direct_dif'][name].append(dif2.cpu().data)
            self.grad_save_all = []
        if epoch%self.save_inter == 0:
            with open(self.save_path, "wb") as fp:
                pickle.dump(self.save_dict, fp)
    
    def load_result_and_show(self, save_path=None):
        if save_path is not None:
            with open(os.path.join(save_path, "rb")) as fp:
                self.save_dict = pickle.load(fp)
        size_dif_new = self.save_dict['size_dif']
        direct_dif_new = self.save_dict['sign_save']
        sign_save_new = self.save_dict['ht_save']
        ht_save__new = self.save_dict['direct_dif']
                  

        for ind, name in enumerate(direct_dif_new.keys()):

            fig = plt.figure(figsize = (10,4))
            plt.subplot(1, 2, 1)
            if 'weight' in  direct_dif_new[name].keys():
                length = range(len(direct_dif_new[name]['weight']))
                plt.plot(length, direct_dif_new[name]['weight'])
                plt.legend(['dif','sign','ht'])
            plt.subplot(1, 2, 2)
            if 'bias' in  direct_dif_new[name].keys():
                length = range(len(direct_dif_new[name]['bias']))
                plt.plot(length, direct_dif_new[name]['bias'])
            plt.title(name)
            plt.savefig(os.path.join(self.direction_dif_path, name+'.png'))
            plt.show()
            plt.close()
                  
        for ind, name in enumerate(size_dif_new.keys()):

            fig = plt.figure(figsize = (10,4))
            plt.subplot(1, 2, 1)
            if 'weight' in  size_dif_new[name].keys():
                length = range(len(size_dif_new[name]['weight']))
                plt.plot(length, size_dif_new[name]['weight'])
                plt.plot(length, sign_save_new[name]['weight'])
                plt.plot(length, ht_save__new[name]['weight'])
                plt.legend(['dif','sign','ht'])
            if 'bias' in  size_dif_new[name].keys():
                plt.subplot(1, 2, 2)
                length = range(len(size_dif_new[name]['bias']))
                plt.plot(length, size_dif_new[name]['bias'])
                plt.plot(length, sign_save_new[name]['bias'])
                plt.plot(length, ht_save__new[name]['bias'])
            plt.title(name[:-4])
            plt.savefig(os.path.join(self.size_dif_path, name[:-4]+'.png'))
            plt.show()
            plt.close()
        
        
def resort_dict(dict_input):
    direct_dif_new = defaultdict(dict)
    for name in dict_input.keys():
        name2,typ = name.rsplit('.',1)
        direct_dif_new[name2][typ] = dict_input[name]
    return direct_dif_new

                  
                  
                  
class RecordMetapathWeight():
    def __init__(self, save_path, inter=10, save_inter=5):
        self.intial_weight = defaultdict(list)   
        self.save_inter = save_inter
                  
        self.save_path = os.path.join(save_path, 'metapath_weight')
        mkdir(self.save_path)
                  
    def record(self, model, epoch):
        for i in model.task_metapath_weight:
            self.intial_weight[i].append(model.task_metapath_weight[i].cpu().data)
        if epoch%self.save_inter == 0:
            with open(os.path.join(self.save_path, 'metapath_weight'), "wb") as fp:
                pickle.dump(self.intial_weight, fp)
             
    def visual(self, save_path=None):
        if save_path is not None:
            with open(save_path, "rb") as fp:
                self.intial_weight = pickle.load(fp)              
        meta_path_weight = defaultdict(list)
        for i in self.intial_weight:
            for j in all_meta_path_list:
                meta_path_weight[i].append([])
        for i in self.intial_weight:
            for j in self.intial_weight[i]:
                for ind, k in enumerate(j[0]):
                    meta_path_weight[i][ind].append(torch.sigmoid(k).numpy())
        for j in range(len(meta_path_weight[i])):
            for i in meta_path_weight:
                plt.plot(range(len(meta_path_weight[i][j])), meta_path_weight[i][j])
            plt.legend(['sign','hot_clue'])
            plt.show()          
                      
# 定义recorder类的优点是不需要将需要输出的参数进行层层传递
class RecordMetapathAttention():
    def __init__(self, save_path, inter=10, save_inter=5):
        self.semantic_attention_all = []
        self.save_inter = save_inter
                  
        self.save_path = os.path.join(save_path, 'metapath_attention')
        mkdir(self.save_path)
                  
    def record(self, semantic_attention):
        self.semantic_attention_all.append(semantic_attention[...,0])
    
    def end_record(self):
        results = torch.cat(self.semantic_attention_all)
        self.semantic_attention_all = []
        return results
                  

        
       

