import exps as exps
from persearch.args import arg
from persearch.utils import Summary, save_dict
from persearch.process import initialization, load_data, runs


"""
train once, test on all the given evaluation cases
* pipeline(dataset(data_ver, tokenize, pick), model_keys, evals):
    * input cfg, cfg.save, make log dir
    * load and tokenize data
    * initialize picker for data subset and split
    for i in n_rep:
        * set seed
        * pick data
        * build evaluator  # select eval settings and prepare test data
        for model in model_keys:
            * get cfg_model and modified cfg_model
            * build model(cfg_model)
            * model.fit()
            * evaluator.evaluate(model)  # evaluate and log summary
        evaluator.summary(path_outp)
"""


if __name__ == '__main__':

    device, prefix, prefix_cfg, prefix_rst = initialization(arg)
    data = load_data(arg, prefix_cfg)
    eval_keys, cfgs_model = getattr(exps, arg.test)()

    runs(arg, data, cfgs_model, eval_keys, prefix, device)
    if arg.output:
        [save_dict(cfg_model, prefix_cfg, '{}.json'.format(cfg_model['name']))
         for cfg_model in cfgs_model]
        report = Summary()
        report.summary(dir_inp=prefix, dir_outp=prefix_rst, keywords=eval_keys)
