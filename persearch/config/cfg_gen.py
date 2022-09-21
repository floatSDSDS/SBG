"""store config of data generators"""

def get_cfg_gen(key, arg=None):

    cfg_gen = dict()

    cfg_gen['amazon_uqii'] = dict(
        name='amazon_uqii',
        gen='AmazonUqii',
        param=dict(use_rand=True, n_neg_rand=2),
    )

    cfg_gen['amazon_uqtiil'] = dict(
        name='amazon_uqtiil',
        gen='GetAmazonUqiil',
        param=dict(
            use_real=False, use_rand=True, n_neg_rand=2, max_clk_seq=10),
    )

    if arg is not None:
        for key_var in cfg_gen[key].keys():
            if key_var in arg.keys():
                cfg_gen[key][key_var] = arg[key_var]

    return cfg_gen[key]
