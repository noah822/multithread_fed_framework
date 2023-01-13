import torch

def fedAvg(params, weights):
    res_state_dict = {}
    # init result state dict
    for _param_name, _param_v in params[0].items():
        if _param_name.endswith('num_batches_tracked'):
            res_state_dict[_param_name] = _param_v
        else:
            res_state_dict[_param_name] = torch.zeros_like(_param_v)
    
    for label, param in enumerate(params):
        for _param_name, _param_v in param.items():
            if _param_name.endswith('num_batches_tracked'): continue
            else:
                res_state_dict[_param_name] += weights[label] * _param_v
    return res_state_dict