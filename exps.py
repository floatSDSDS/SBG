from persearch.config import get_cfg_model
from persearch.args import arg


"""
design different test here, return a series of configs (dict) of models, takes 
prefix_outp to log these config
def <experiment name>()
    :return: [config: dict]
"""

eval_keys = [
    'uq_topk_mix1000',
]

def model_default():
    """check models, use their default config"""
    model_keys = [
        'ZAM',
        'SBG',
    ]
    cfgs_model = [get_cfg_model(key, arg) for key in model_keys]
    return eval_keys, cfgs_model
