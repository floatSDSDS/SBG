from datetime import datetime
from pathlib import Path

import torch
import numpy as np

from persearch.config import get_info_data
from persearch.utils import Evaluator, save_dict, print_title, timing
from persearch.data import DataLoader, Tokenizer, DataPicker
import persearch.model as models


@timing
def initialization(arg):
    device = torch.device("cuda" if torch.cuda.is_available() & arg.use_gpu else "cpu")
    dataset_key = arg.dataset
    data_name, data_ver = dataset_key.split('@')
    prefix = Path(arg.dir_log, data_name, data_ver, arg.caption,
                  datetime.now().strftime('%b%d%H%M%S').lower())
    prefix_cfg = Path(prefix, 'cfg')
    prefix_rst = Path(prefix, 'rst')
    if arg.output:
        prefix.mkdir(parents=True, exist_ok=True)
        prefix_cfg.mkdir(parents=True, exist_ok=True)
        prefix_rst.mkdir(parents=True, exist_ok=True)
        save_dict(vars(arg), prefix_cfg, 'args.json')
    return device, prefix, prefix_cfg, prefix_rst


@timing
def load_data(arg, prefix_outp):

    info = get_info_data(arg.dataset)
    data_path, name = info['data_path'], info['name']
    data = DataLoader(data_path, name)
    tokenizer = Tokenizer(mode='dict')
    tokenizer.build_dictionary(data.corpus)
    data.tokenize(tokenizer)
    picker = DataPicker(strategy=arg.strategy, use_p=arg.prop)
    data.set_picker(picker)
    if arg.output:
        save_dict(info, prefix_outp, 'data_info.json')
    return data


@timing
def runs(arg, data, cfgs_model, eval_keys, prefix, device):

    for i_rep in range(arg.repeat):
        print_title('ROUND {}'.format(i_rep + 1), token='#')
        seed = arg.seed + i_rep
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        data.pick()
        evaluator = Evaluator(data, eval_keys, prefix=prefix, is_test=arg.is_test)
        for cfg_model in cfgs_model:
            model = getattr(models, cfg_model['model'])(data, cfg_model)
            model.move_to(device)
            while model.i_epoch < cfg_model['epoch']:
                model.fit(step=arg.step)
                evaluator.evaluate(model)
        if arg.output:
            evaluator.write(prefix)
        else:
            evaluator.print()
