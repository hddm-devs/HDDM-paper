from collections import OrderedDict
from copy import deepcopy, copy
import time
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from IPython import parallel
from IPython.parallel.client.asyncresult import AsyncResult
import estimate as est


def singleMAP_fixed_n_samples(n_subjs=8, n_samples=200):

    #include params
    params = {'include': ('v','t','a')}

    #kwards for gen_rand_data
    subj_noise = {'v':0.1, 'a':0.1, 't':0.05}
    data = {'subjs': n_subjs, 'subj_noise': subj_noise, 'size': n_samples}

    #kwargs for initialize estimation
    init = {}

    #kwargs for estimation
    estimate = {'runs': 3}

    #creat kw_dict
    kw_dict = {'params': params, 'data': data, 'init': init, 'estimate': estimate}

    #create filename
    now = time.strftime("%Y%m%dT%H%M%S", time.localtime())
    filename = 'singleMAP.fixed_n_samples.simple.%d_subjs.%d_samples.%s.pickle' % (n_subjs, n_samples, now)

    #run analysis
    recover = est.multi_recovery_fixed_n_samples
    results = recover(est.EstimationSingleMAP, seed_data=1, seed_params=1, n_runs=3, mpi=False,
            kw_dict=kw_dict, path=filename)

    return results

def run_experiments(n_subjs=(12,), n_trials=(10, 40, 100), n_runs=5, view=None):
    if not isinstance(n_subjs, (tuple, list)):
        n_subjs = (n_subjs,)
    if not isinstance(n_trials, (tuple, list)):
        n_trials = (n_trials,)

    #include params
    params = {'include': ('v','t','a')}
    recover = est.multi_recovery_fixed_n_trials

    n_subjs_results = {}
    for cur_subjs in n_subjs:
        n_trials_results = {}
        for cur_trials in n_trials:
            #kwards for gen_rand_data
            subj_noise = {'v':0.1, 'a':0.1, 't':0.05}
            data = {'subjs': cur_subjs, 'subj_noise': subj_noise, 'size': cur_trials}

            #kwargs for initialize estimation
            init = {}

            #kwargs for estimation
            estimate = {'runs': 3}

            #creat kw_dict
            kw_dict = {'params': params, 'data': data, 'init': init, 'estimate': estimate}

            models_params = OrderedDict()
            models_params['SingleMAP'] = {'estimator': est.EstimationSingleMAP, 'params': {'runs': 3}}
            models_params['HDDM'] = {'estimator': est.EstimationHDDM, 'params': {'samples': 20000, 'burn': 15000}}

            models_results = {}
            for model_name, descr in models_params.iteritems():
                kw_dict_model = deepcopy(kw_dict)
                kw_dict_model['estimate'] = descr['params']
                #run analysis
                models_results[model_name] = recover(descr['estimator'], seed_data=1, seed_params=1, n_runs=n_runs,
                                                     kw_dict=kw_dict_model, view=view)


            n_trials_results[cur_trials] = models_results

        n_subjs_results[cur_subjs] = n_trials_results

    return n_subjs_results

def plot_trial_exp(data):
    grouped = data.MSE.dropna().groupby(level=('n_trials', 'estimation', 'params')).mean()

    fig = plt.figure()
    for i, (param_name, param_data) in enumerate(grouped.groupby(level=('params',))):
        ax = fig.add_subplot(3, 1, i+1)
        ax.set_title(param_name)
        for est_name, est_data in param_data.groupby(level=['estimation']):
            ax.plot(est_data.index.get_level_values('n_trials'), est_data, label=est_name)

        ax.set_xlabel('trials')
        ax.set_ylabel('MSE')

    plt.legend()

def plot_subj_exp(data):
    grouped = data.MSE.dropna().groupby(level=('n_subjs', 'estimation', 'params')).mean()

    fig = plt.figure()
    for i, (param_name, param_data) in enumerate(grouped.groupby(level=('params',))):
        ax = fig.add_subplot(3, 1, i+1)
        ax.set_title(param_name)
        for est_name, est_data in param_data.groupby(level=['estimation']):
            ax.plot(est_data.index.get_level_values('n_subjs'), est_data, label=est_name)

        ax.set_xlabel('subjs')
        ax.set_ylabel('MSE')

    plt.legend()

def plot_recovery_exp(data):
    for i, (est_name, est_data) in enumerate(data.dropna().groupby(level=['estimation'])):
        for j, (param_name, param_data) in enumerate(est_data.groupby(level=('params',))):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title(param_name)
            ax.plot(param_data.truth, param_data.estimate, 'x', label=est_name)

            mini = min(param_data.truth.min(), param_data.estimate.min())
            maxi = max(param_data.truth.max(), param_data.estimate.max())
            ax.set_xlim((mini, maxi))
            ax.set_ylim((mini, maxi))

            ax.set_xlabel('truth')
            ax.set_ylabel('estimated')

            plt.legend()

            plt.savefig('recovery_exp_%s_%s.png'%(est_name, param_name))
            plt.savefig('recovery_exp_%s_%s.svg'%(est_name, param_name))


def concat_dicts(d, names=()):
    name = names.pop(0) if len(names) != 0 else None

    if isinstance(d.values()[0], pd.DataFrame):
        return pd.concat(d, names=[name])
    elif isinstance(d.values()[0], AsyncResult):
        d_get = {k: v.get() for k, v in d.iteritems()}
        return pd.concat(d_get, names=[name])
    else:
        sublevel_d = {}
        for k, v in d.iteritems():
            sublevel_d[k] = concat_dicts(v, names=copy(names))
        return pd.concat(sublevel_d, names=[name])


def merge_and_extract(data):
    results = concat_dicts(data, names=['n_subjs', 'n_trials', 'estimation', 'name_seed', 'param_seed'])
    results = est.select_param(results, ['v', 'a', 't'])

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('profile', type=str,
                        help='Which IPython environment to use.')

    result = parser.parse_args()

    c = parallel.Client(profile=result.profile)
    view = c.load_balanced_view()

    trial_exp = run_experiments(n_subjs=(12,), n_trials=np.linspace(10, 100, 10), n_runs=10, view=view)
    subj_exp = run_experiments(n_subjs=np.linspace(4, 30, 2), n_trials=(20), n_runs=10, view=view)
    recovery_exp = run_experiments(n_subjs=(12), n_trials=(30), n_runs=20, view=view)

    trial_data = merge_and_extract(trial_exp)
    trial_data.save('trial_data.dat')
    plot_trial_exp(trial_data)
    plt.savefig('trial_exp.png')
    plt.savefig('trial_exp.svg')

    subj_data = merge_and_extract(subj_exp)
    subj_data.save('subj_data.dat')
    plot_subj_exp(subj_data)
    plt.savefig('subj_exp.png')
    plt.savefig('subj_exp.svg')

    recovery_data = merge_and_extract(recovery_exp)
    recovery_data.save('recovery_data.dat')
    plot_recovery_exp(recovery_data)
    plt.savefig('recovery_exp.png')
    plt.savefig('recovery_exp.svg')

    sys.exit(0)
    #a = view.apply_async(lambda x: x**2, 3)
    #print a.get()
