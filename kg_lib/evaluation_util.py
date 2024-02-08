import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve, roc_curve, f1_score
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

from .sample_util import sample_sub_graph_with_label_index

BCE_loss = torch.nn.BCELoss()


def evaluation(dataset):
    if dataset == 'Aminer' or dataset == 'DBLP':
        return evaluate_Aminer
    else:
        raise NameError(f"Do not support the dataset {name}. Please try DBLP/Aminer/Amazon/Yelp/JD.")

    



def top_k_accuracy_score(y_true, y_score, k):
    sorted_pred = np.argsort(y_score)
    sorted_pred = sorted_pred[::-1]
    sorted_pred = sorted_pred[:k]
    
    hits = y_true[sorted_pred]
    
    return np.sum(hits)/k


    
def evaluate_Aminer(model, dataloader, data_type, config, print_figure = False):
    model.eval()
    result = {}
    hg_dgl = dataloader.hg_dgl
    with torch.no_grad():    
        h_output = model(hg_dgl)
        loss = 0
        for target_node_name in h_output:
            h_output_squeeze = h_output[target_node_name]
            mask = hg_dgl.nodes[target_node_name].data[f'{data_type}_label_mask']
            source_data_label = hg_dgl.nodes[target_node_name].data[f'{data_type}_label'][mask]

            h_output_squeeze = h_output_squeeze[mask]
            
            loss_i = BCE_loss(h_output_squeeze, source_data_label)
#             loss_i = CE_loss(output, tmp_true_label)
            loss = loss +  loss_i
    
            # 计算roc-auc值和AVG_Pre值
            if config['args_cuda']:
                source_data_label_np = source_data_label.data.cpu().numpy()
                h_output_squeeze_np = h_output_squeeze.data.cpu().numpy()
            else:
                source_data_label_np = source_data_label.data.numpy()
                h_output_squeeze_np = h_output_squeeze.data.numpy()
            
            task_type = config['task_type'][target_node_name]
            if  task_type == 'single-label':
                h_output_squeeze_np = h_output_squeeze_np.argmax(axis=1)
                source_data_label_np = source_data_label_np.argmax(axis=1)
            elif task_type == 'multi-label':
                h_output_squeeze_np = (h_output_squeeze_np>0.5).astype(int)
            else:
                raise NameError(f'task type {task_type} does not exist!')
            micro = f1_score(source_data_label_np, h_output_squeeze_np, average='micro')
            macro = f1_score(source_data_label_np, h_output_squeeze_np, average='macro')
            
            
            result[target_node_name] = {'loss':loss_i.item(), 'micro': micro, 'macro': macro}
    
    if print_figure:
        plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

        plt.plot(precision, recall, label = 'Val AVG_PRECISION = %0.3f' % average_precision)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('recall')
        plt.xlabel('precision')
        plt.show()
    
#     top_k_acc_dict = {}
#     for tmp_aim_k in [100, 500, 1000, 5000, 10000]:
#         top_k_acc_dict[tmp_aim_k] = top_k_accuracy_score(source_data_label_np, h_output_squeeze_np, k = tmp_aim_k)
    
    return result

\