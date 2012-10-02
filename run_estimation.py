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
    if not isinstance(n_subjs, (tuple, list, np.ndarray)):
        n_subjs = (n_subjs,)
    if not isinstance(n_trials, (tuple, list, np.ndarray)):
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
    grouped = data.MSE.dropna().groupby(level=('n_trials', 'estimation', 'params')).agg((np.mean, np.std))

    fig = plt.figure()
    plt.subplots_adjust(hspace=.5)
    for i, (param_name, param_data) in enumerate(grouped.groupby(level=('params',))):
        ax = fig.add_subplot(3, 1, i+1)
        ax.set_title(param_name)
        ax.set_xlim(5, 95)
        for est_name, est_data in param_data.groupby(level=['estimation']):
            ax.errorbar(est_data.index.get_level_values('n_trials'),
                        est_data['mean'], label=est_name, lw=2.,
                        marker='o')

        ax.set_ylabel('MSE')

    ax.set_xlabel('trials')
    plt.legend(loc=0)

def plot_subj_exp(data):
    grouped = data.MSE.dropna().groupby(level=('n_subjs', 'estimation', 'params')).agg((np.mean, np.std))

    fig = plt.figure()
    plt.subplots_adjust(hspace=.5)
    for i, (param_name, param_data) in enumerate(grouped.groupby(level=('params',))):
        ax = fig.add_subplot(3, 1, i+1)
        ax.set_title(param_name)
        ax.set_xlim(2, 30)
        for est_name, est_data in param_data.groupby(level=['estimation']):
            ax.errorbar(est_data.index.get_level_values('n_subjs'),
                        est_data['mean'], label=est_name, lw=2.,
                        marker='o')

        ax.set_ylabel('MSE')
    ax.set_xlabel('subjs')
    plt.legend(loc=0)

def plot_recovery_exp(data):
    for i, (est_name, est_data) in enumerate(data.dropna().groupby(level=['estimation'])):
        for j, (param_name, param_data) in enumerate(est_data.groupby(level=('params',))):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title(param_name)
            #ax.plot(param_data.truth, param_data.estimate, 'x', label=est_name)

            mini = min(param_data.truth.min(), param_data.estimate.min())
            maxi = max(param_data.truth.max(), param_data.estimate.max())
            ax.set_xlim((mini, maxi))
            ax.set_ylim((mini, maxi))
            ax.set_xlabel('truth')
            ax.set_ylabel('estimated')
            kwargs = {'gridsize': 100, 'bins': 'log', 'extent': (mini, maxi, mini, maxi)}
            ax.hexbin(param_data.truth, param_data.estimate, label='post pred lb', **kwargs)

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
    parser = argparse.ArgumentParser(description='Run HDDM experiments.')
    parser.add_argument('--profile', action='store', dest='profile', type=str, default='mpi',
                        help='Which IPython environment to use.')
    parser.add_argument('-r', action='store_true', dest='run', default=False,
                        help='Whether to run simulations.')
    parser.add_argument('-a', action='store_true', dest='analyze', default=False,
                        help='Whether to analyze and plot results.')
    parser.add_argument('-l', action='store_true', dest='load', default=False,
                        help='Whether to load results from file.')
    parser.add_argument('--parallel', action='store_true', dest='parallel', default=False,
                        help='Whether to use IPython parallelize.')

    result = parser.parse_args()

    if result.parallel:
        c = parallel.Client(profile=result.profile)
        view = c.load_balanced_view()
    else:
        view = None

    if result.run:
        trial_exp = run_experiments(n_subjs=(12,), n_trials=list(np.arange(10, 100, 10)), n_runs=10, view=view)
        subj_exp = run_experiments(n_subjs=list(np.arange(4, 30, 2)), n_trials=(20), n_runs=10, view=view)
        recovery_exp = run_experiments(n_subjs=(12), n_trials=(30), n_runs=20, view=view)

        trial_data = merge_and_extract(trial_exp)
        subj_data = merge_and_extract(subj_exp)
        recovery_data = merge_and_extract(recovery_exp)

        trial_data.save('trial.dat')
        subj_data.save('subj.dat')
        recovery_data.save('recovery.dat')

    if result.load:
        trial_data = pd.load('trial.dat')
        subj_data = pd.load('subj.dat')
        recovery_data = pd.load('recovery.dat')

    if result.analyze:
        plot_trial_exp(trial_data)
        plt.savefig('trial_exp.png')
        plt.savefig('trial_exp.svg')

        plot_subj_exp(subj_data)
        plt.savefig('subj_exp.png')
        plt.savefig('subj_exp.svg')

        plot_recovery_exp(recovery_data)

    sys.exit(0)
    #a = view.apply_async(lambda x: x**2, 3)
    #print a.get()
