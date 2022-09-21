"""store default model config"""


def get_cfg_model(key, arg=None):
    cfg_model = dict()
    cfg_model['Base'] = dict(
        name='Base',
        model='Base',
        generator='amazon_uqii',
        batch_size=1024,
        lr=1e-3,
        d=64,
    )

    cfg_model['SBG'] = cfg_model['Base'].copy()
    cfg_model['SBG'].update((dict(
        name='SBG',
        model='SBG',
        user_as_item=True,  # (use query/item as user's textual representation)
        share_term_emb=True,  # (whether share/separate term embedding for user and item)
        neg_ratio_i=2,
        neg_ratio_w=5,
        a_bce=1,
        a_lm=1,
        lambda_uq=0.5,
        d_attn=8,
        generator='amazon_uqtiil',
        conv_pv='e',  # ['conv_e', 'e', 'i'],
        conv_i='e',  # ['conv_e', 'e', 'i'],
        conv_u='conv_e',  # ['conv_e', 'e', 'i'],
        a_self=0.1,
        drop_gcn=0.5,
        k=2,
        duration=604800,  # 86400, 604800, 2592000, 7776000
    )))

    cfg_model['ZAM'] = cfg_model['Base'].copy()
    cfg_model['ZAM'].update(dict(
        name='ZAM',
        model='ZAM',
        user_as_item=True,  # (use query/item as user's textual representation)
        share_term_emb=True,  # fixed attribute
        neg_ratio_i=2,
        neg_ratio_w=5,
        a_bce=1,
        a_lm=1,
        lambda_uq=0.5,
        conv_pv='e',  # ['conv_e', 'e', 'i'],
        conv_i='e',  # ['conv_e', 'e', 'i'],
        conv_u='conv_e',  # ['conv_e', 'e', 'i'],
        d_attn=8,
        generator='amazon_uqtiil',
    ))

    if arg is not None:
        for key_var in cfg_model[key].keys():
            if key_var in vars(arg).keys():
                cfg_model[key][key_var] = vars(arg)[key_var]

    cfg_model[key]['epoch'] = arg.epoch

    if arg.fast:
        cfg_model[key]['epoch'] = 1

    return cfg_model[key]
