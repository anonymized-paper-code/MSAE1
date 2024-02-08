from .utils import load_config
import torch


def load_parameter(config_path):
    # load config 
    config = load_config(config_path)
    
    # default setting
    if 'recorder_type' not in config:
        config['recorder_type'] = []
    
    # check cuda
    if torch.cuda.is_available():
        print('cuda')
        args_cuda = True
    else:
        print('cpu')
        args_cuda = False
    config['args_cuda'] = args_cuda
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    
    # for PLE
    #if (config['model']=='PLE_SimpleHGN') and ('split_method' in config) and (config['split_method'] == 'layer'):
    if (config['model']=='PLE_SimpleHGN') or (config['model']=='MMOE'):
        config['num_levels'] = config['layer_num']
        
    if 'try_pcgrad' in config:
        if config['try_pcgrad']:
            import sys
            sys.path.append("/media/cfs/fengxinyue8/Pytorch-PCGrad-master")
            from pcgrad import PCGrad
            
    # for Our
#     if config['model'] == 'Our_old':
#         config['layer_num'] = config['layer_num'] - 1 #第一层被disentanglement module替代
    if config['model'] == 'Our_old':
        if config['layer_num'] - config['disen_layer_num'] < 0:
            config['disen_layer_num'] = config['layer_num']
            config['layer_num'] = 0 # 用PD layer逐步取代original PLE layer
        else:
            config['layer_num'] = config['layer_num'] - config['disen_layer_num']
        
        config['num_levels'] = config['layer_num']  # 
#         if Model_Config_dict['num_levels'] ==0:
#             Model_Config_dict['layer_num'] = 1
    if config['model']=='Our':
        config['layer_num'] = config['num_levels']

    return config