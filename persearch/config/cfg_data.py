"""
store dataset information, static through test
assume default root as PerSearch/<one-level-path>, like PerSearch/tests
"""


def get_info_data(key: str) -> dict:
    info = dict()

    info['amazon_magazine'] = dict(
        name='Magazine_Subscriptions',
        data_path='data/amazon/mm/Magazine_Subscriptions/',
        col_label='click',
    )

    info['amazon_software'] = dict(
        name='Software',
        data_path='data/amazon/mm/Software/',
        col_label='click',
    )

    dataset_default = list(info.keys())
    for dataset in dataset_default:
        if dataset in key:
            info[key] = info[dataset].copy()
            info[key]['data_path'] += key.split('@')[1]

    return info[key]
