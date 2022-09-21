"""store pre-defined evaluations"""


def get_eval(key):
    evals = dict()
    evals['uq_topk_mix1000'] = dict(
        eval_name='uq_topk_mix1000',
        eval_key='uq_topk_mix',
        params=dict(topk=1000),
        metrics=metrics_topk,
    )
    evals['uqi_frequent0_mix'] = dict(
        eval_name='uqi_frequent0_mix',
        eval_key='uqi_frequent_mix',
        params=dict(topk=100, lim_freq=0),
        metrics=metrics_topk,
    )
    evals['uqi_frequent1_mix'] = dict(
        eval_name='uqi_frequent1_mix',
        eval_key='uqi_frequent_mix',
        params=dict(topk=100, lim_freq=1),
        metrics=metrics_topk,
    )
    evals['uqi_frequent2_mix'] = dict(
        eval_name='uqi_frequent2_mix',
        eval_key='uqi_frequent_mix',
        params=dict(topk=100, lim_freq=2),
        metrics=metrics_topk,
    )
    evals['uqi_frequent3_mix'] = dict(
        eval_name='uqi_frequent3_mix',
        eval_key='uqi_frequent_mix',
        params=dict(topk=100, lim_freq=3),
        metrics=metrics_topk,
    )
    evals['uqi_frequent4_mix'] = dict(
        eval_name='uqi_frequent4_mix',
        eval_key='uqi_frequent_mix',
        params=dict(topk=100, lim_freq=4),
        metrics=metrics_topk,
    )
    evals['uqi_frequent16_mix'] = dict(
        eval_name='uqi_frequent16_mix',
        eval_key='uqi_frequent_mix',
        params=dict(topk=100, lim_freq=16),
        metrics=metrics_topk,
    )
    return evals[key]


metrics_topk = dict(
    hr10=dict(name='hr', k=10),
    hr20=dict(name='hr', k=20),
    hr50=dict(name='hr', k=50),
    hr100=dict(name='hr', k=100),

    mrr10=dict(name='mrr', k=10),
    mrr20=dict(name='mrr', k=20),
    mrr50=dict(name='mrr', k=50),
    mrr100=dict(name='mrr', k=100),

    ndcg10=dict(name='ndcg', k=10),
    ndcg20=dict(name='ndcg', k=20),
    ndcg50=dict(name='ndcg', k=50),
    ndcg100=dict(name='ndcg', k=100),
)
