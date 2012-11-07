from collections import OrderedDict
from copy import deepcopy, copy
import time
import argparse
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid

import numpy as np
import pandas as pd

from IPython import parallel
from IPython.parallel.client.asyncresult import AsyncResult
import estimate as est

PARAM_NAMES = {'a': 'threshold',
               'v': 'drift-rate',
               't': 'non-decision time',
               'z': 'bias',
               'st': 'non-decision time var.',
               'sz': 'bias var.'
}

def run_experiments(n_subjs=(12,), n_trials=(10, 40, 100), n_params=5, n_datasets=5, include=('v','t','a'),
                    estimators=None, n_outliers=(0,), view=None):
    if not isinstance(n_subjs, (tuple, list, np.ndarray)):
        n_subjs = (n_subjs,)
    if not isinstance(n_trials, (tuple, list, np.ndarray)):
        n_trials = (n_trials,)

    #include params
    params = {'include': include}
    recover = est.multi_recovery_fixed_n_trials

    #set estimator_dict
    if estimators is None:
        estimators = ['SingleMAP', 'HDDM', 'Quantiles_subj', 'Quantiles_group']

    estimator_dict = OrderedDict()
    if 'SingleMAP' in estimators:
        estimator_dict['SingleMAP'] = {'estimator': est.EstimationSingleMAP, 'params': {'runs': 3}}
    if 'SingleMAPoutliers' in estimators:
        estimator_dict['SingleMAPoutliers'] = {'estimator': est.EstimationSingleMAPoutliers, 'params': {'runs': 3}}
    if 'HDDM' in estimators:
        estimator_dict['HDDM'] = {'estimator': est.EstimationHDDM, 'params': {'samples': 35000, 'burn': 30000, 'map': True}}
    if 'Quantiles_subj' in estimators:
        estimator_dict['Quantiles_subj'] = {'estimator': est.EstimationSingleOptimization,
                                           'params': {'method': 'chisquare', 'quantiles': (0.1, 0.3, 0.5, 0.7, 0.9)}}
    if 'Quantiles_group' in estimators:
        estimator_dict['Quantiles_group'] = {'estimator': est.EstimationGroupOptimization,
                                            'params': {'method': 'chisquare', 'quantiles': (0.1, 0.3, 0.5, 0.7, 0.9)}}


    n_subjs_results = {}
    for cur_subjs in n_subjs:
        n_trials_results = {}
        for cur_trials in n_trials:

            #kwards for gen_rand_data
            subj_noise = {'v':0.1, 'a':0.1, 't':0.05}
            if 'z' in include:
                subj_noise['z'] = .1
            if 'sz' in include:
                subj_noise['sz'] = .05
            if 'st' in include:
                subj_noise['st'] = .05
            if 'sv' in include:
                subj_noise['sv'] = .05

            #kwargs for initialize estimation
            init = {'include': include}

            #kwargs for estimation
            estimate = {'runs': 3}

            n_outliers_results = {}
            for cur_outliers in n_outliers:

                n_fast_outliers = cur_outliers // 2;
                n_slow_outliers = cur_outliers- n_fast_outliers
                data = {'subjs': cur_subjs, 'subj_noise': subj_noise, 'size': cur_trials,
                        'n_fast_outliers': n_fast_outliers, 'n_slow_outliers': n_slow_outliers}
                #creat kw_dict
                kw_dict = {'params': params, 'data': data, 'init': init, 'estimate': estimate}

                models_results = {}
                for model_name, descr in estimator_dict.iteritems():
                    kw_dict_model = deepcopy(kw_dict)
                    kw_dict_model['estimate'] = descr['params']
                    #run analysis
                    models_results[model_name] = recover(descr['estimator'], seed_data=1, seed_params=1, n_params=n_params,
                                                         n_datasets=n_datasets, kw_dict=kw_dict_model, view=view)

                n_outliers_results[cur_outliers] = models_results
            #end of (for cur_outliers in n_outliers)

            n_trials_results[cur_trials] = n_outliers_results
        #end of (for cur_trials in n_trials)

        n_subjs_results[cur_subjs] = n_trials_results
    #end of (for cur_subjs in n_subjs)

    return n_subjs_results

def plot_trial_exp(data):
    grouped = data.MSE.dropna().groupby(level=('n_trials', 'estimation', 'params')).agg((np.mean, np.std))
    n_params = len(grouped.groupby(level=('params',)).groups.keys())

    fig = plt.figure(figsize=(8, n_params*3))
    grid = Grid(fig, 111, nrows_ncols=(n_params, 1), add_all=True, share_all=False, label_mode='L', share_x=True, share_y=False, axes_pad=.25)

    for i, (param_name, param_data) in enumerate(grouped.groupby(level=('params',))):
        ax = grid[i]
        ax.set_ylabel(PARAM_NAMES[param_name])
        ax.set_xlim(5, 95)
        ax.set_yscale('log')
        for est_name, est_data in param_data.groupby(level=['estimation']):
            ax.errorbar(est_data.index.get_level_values('n_trials'),
                        est_data['mean'], label=est_name, lw=2.,
                        marker='o')
            #ax.yaxis.set_major_locator(MaxNLocator(20))


        #ax.set_ylabel('MSE')

    ax.set_xlabel('trials')
    plt.legend(loc=0)

def plot_subj_exp(data):
    grouped = data.MSE.dropna().groupby(level=('n_subjs', 'estimation', 'params')).agg((np.mean, np.std))
    n_params = len(grouped.groupby(level=('params',)).groups.keys())

    fig = plt.figure(figsize=(8, n_params*3))
    grid = Grid(fig, 111, nrows_ncols=(n_params, 1), add_all=True, share_all=False, label_mode='L', share_x=True, share_y=False, axes_pad=.25)

    for i, (param_name, param_data) in enumerate(grouped.groupby(level=('params',))):
        ax = grid[i]
        ax.set_ylabel(PARAM_NAMES[param_name])
        ax.set_xlim(2, 30)
        ax.set_yscale('log')
        for est_name, est_data in param_data.groupby(level=['estimation']):
            ax.errorbar(est_data.index.get_level_values('n_subjs'),
                        est_data['mean'], label=est_name, lw=2.,
                        marker='o')
            #ax.yaxis.set_major_locator(MaxNLocator(7))

        #ax.set_ylabel('MSE')
    ax.set_xlabel('subjs')
    plt.legend(loc=0)

def plot_recovery_exp(data, tag='', abs_min=-5, abs_max=5, gridsize=100, save=True):
    ni = len(data.dropna().groupby(level=['estimation']))
    nj = len(data.dropna().groupby(level=('params',)))

    data = data.dropna()
    data = data[(data['estimate'] > abs_min) & (data['estimate'] < abs_max)]
    data_params = data.groupby(level=('params',))[['truth', 'estimate']]
    mini = data_params.min().min(axis=1)
    maxi = data_params.max().max(axis=1)
    print mini
    print maxi

    fig = plt.figure(figsize=(9, 3*nj))
    grid = Grid(fig, 111, nrows_ncols=(nj, ni), add_all=True, share_all=False, label_mode='L', share_x=False, share_y=False)
    for i, (est_name, est_data) in enumerate(data.dropna().groupby(level=['estimation'])):
        nj = len(est_data.groupby(level=('params',)))
        for j, (param_name, param_data) in enumerate(est_data.groupby(level=('params',))):
            ax = grid.axes_column[i][j] #plt.subplot2grid((nj, ni), (j, i))
            # if i == 0:
            #     ax.set_title(est_name)
            #ax.plot(param_data.truth, param_data.estimate, 'x', label=est_name)
            minimaxi = (mini[param_name], maxi[param_name])
            ax.set_xlim(minimaxi)
            ax.set_ylim(minimaxi)
            ax.set_xlabel(est_name)
            ax.set_ylabel(PARAM_NAMES[param_name])
            kwargs = {'gridsize': gridsize, 'bins': 'log', 'extent': (mini[param_name], maxi[param_name], mini[param_name], maxi[param_name])}
            ax.hexbin(param_data.truth, param_data.estimate, label='post pred lb', **kwargs)

            plt.legend()

    if save:
        plt.savefig('recovery_exp_%s.png'%(tag), dpi=600)
        plt.savefig('recovery_exp_%s.svg'%(tag))

def plot_outliers_exp(data, tag='', gridsize=100, save=True):

    data = data.dropna()
    data_params = data.groupby(level=('params',))[['truth', 'estimate']]
    mini = data_params.min().min(axis=1)
    maxi = data_params.max().max(axis=1)
    print mini
    print maxi

    grouped_data = data.dropna().groupby(level=['estimation'])
    MAPoutliers_data = grouped_data.get_group('SingleMAPoutliers')
    ni = len(grouped_data) - 1
    nj = len(data.groupby(level=('params',)))
    fig = plt.figure()#figsize=(9, 3*nj))
    grid = Grid(fig, 111, nrows_ncols=(nj, ni), add_all=True, share_all=False, label_mode='L', share_x=False, share_y=False)
    for i, (est_name, est_data) in enumerate(grouped_data):
        if est_name == 'SingleMAPoutliers':
            continue
        for j, (param_name, param_data) in enumerate(est_data.groupby(level=('params',))):
            ax = grid.axes_column[i][j]
            minimaxi = (mini[param_name], maxi[param_name])
#            ax.set_xlim(minimaxi)
#            ax.set_ylim(minimaxi)
            ax.set_xlabel(est_name)
            ax.set_ylabel(PARAM_NAMES[param_name])
            kwargs = {'gridsize': gridsize, 'bins': 'log'}
#                      'extent': (mini[param_name], maxi[param_name], mini[param_name], maxi[param_name])}
#            ax.hexbin(np.abs(param_data.relErr), MAPoutliers_data.ix[param_name].relErr, **kwargs)
            ax.scatter(np.abs(param_data.relErr), np.abs(MAPoutliers_data.ix[param_name].relErr))
            lb = min(ax.get_xlim()[0], ax.get_ylim()[0])
            ub = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([lb, ub], [lb, ub])
#            ax.axis('equal')
            ax.plot()

#            plt.legend()

    if save:
        plt.savefig('recovery_exp_%s.png'%(tag), dpi=600)
        plt.savefig('recovery_exp_%s.svg'%(tag))


def concat_dicts(d, names=()):
    name = names.pop(0) if len(names) != 0 else None

    if isinstance(d.values()[0], pd.DataFrame):
        return pd.concat(d, names=[name])
    elif isinstance(d.values()[0], AsyncResult):
        d_get = {}
        for k, v in d.iteritems():
            d_get[k] = v.get()
        return pd.concat(d_get, names=[name])
    else:
        sublevel_d = {}
        for k, v in d.iteritems():
            sublevel_d[k] = concat_dicts(v, names=copy(names))
        return pd.concat(sublevel_d, names=[name])


def merge(data):
    results = concat_dicts(data, names=['n_subjs', 'n_trials', 'n_outliers', 'estimation', 'param_seed', 'data_seed'])
    return results

def select(stats, param_names, subj=True):
    if isinstance(param_names, str):
        param_names = [param_names]

    if subj:
        estimators = ['SingleMAP', 'HDDM', 'Quantiles_subj', 'SingleMAPoutliers']
    else:
        estimators = ['HDDM', 'Quantiles_group']

    extracted = {}
    index = stats.index
    for name in param_names:
        select = []
        for ix in index:
            if subj:
                if ix[-4] in estimators and ix[-1].startswith(name) and 'subj' in ix[-1]:
                    select.append(ix)
            else:
                if ix[-4] in estimators and ix[-1] == name:
                    select.append(ix)

        extracted[name] = stats.ix[select]

    return pd.concat(extracted, names=['params'])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run HDDM experiments.', add_help=True)
    parser.add_argument('--profile', action='store', dest='profile', type=str, default='mpi',
                        help='IPython environment to use.')
    parser.add_argument('-r', action='store_true', dest='run', default=False,
                        help='Run simulations.')
    parser.add_argument('-a', action='store_true', dest='analyze', default=False,
                        help='Analyze and plot results.')
    parser.add_argument('-l', action='store_true', dest='load', default=False,
                        help='Load results from file.')
    parser.add_argument('--parallel', action='store_true', dest='parallel', default=False,
                        help='Use IPython parallel.')
    parser.add_argument('--trials', action='store_true', dest='trials', default=False,
                        help='Run trial experiment.')
    parser.add_argument('--subjs', action='store_true', dest='subjs', default=False,
                        help='Run subject experiment.')
    parser.add_argument('--recovery', action='store_true', dest='recovery', default=False,
                        help='Run recovery experiment.')
    parser.add_argument('--outliers', action='store_true', dest='outliers', default=False,
                        help='Run outliers experiment')
    parser.add_argument('--all', action='store_true', dest='all', default=False,
                        help='Run all of the above experiments.')
    parser.add_argument('--group', action='store_true', dest='group', default=False,
                        help='Run only group estimations.')
    parser.add_argument('-st', action='store_true', dest='st', default=False)
    parser.add_argument('-sv', action='store_true', dest='sv', default=False)
    parser.add_argument('-sz', action='store_true', dest='sz', default=False)
    parser.add_argument('-z', action='store_true', dest='z', default=False)
    parser.add_argument('--full', action='store_true', dest='full', default=False)
    parser.add_argument('--debug', action='store_true', dest='debug', default=False)

    include=['v','t','a']

    result = parser.parse_args()

    if result.st or result.full:
        include.append('st')
    if result.sv or result.full:
        include.append('sv')
    if result.sz or result.full:
        include.append('sz')
    if result.z or result.full:
        include.append('z')

    run_trials, run_subjs, run_recovery, run_outliers = result.trials, result.subjs, result.recovery, result.outliers

    if result.all:
        run_trials, run_subjs, run_recovery = True, True, True

    if result.parallel:
        c = parallel.Client(profile=result.profile)
        view = c.load_balanced_view()
    else:
        view = None

    if result.run:
        if result.group:
            estimators = ['HDDM','Quantiles_group']
        else:
            estimators = None

        if result.debug:
            if run_trials:
                trial_exp = run_experiments(n_subjs=12, n_trials=100, estimators=estimators, n_params=3, n_datasets=1, include=include, view=view)
            if run_subjs:
                subj_exp = run_experiments(n_subjs=2, n_trials=20, n_params=1, n_datasets=1, include=include, view=view)
            if run_recovery:
                recovery_exp = run_experiments(n_subjs=12, n_trials=30, n_params=2, n_datasets=1, include=include, view=view)
            if run_outliers:
                outliers_estimators = ['SingleMAP', 'SingleMAPoutliers', 'Quantiles_subj']
                outliers_exp = run_experiments(n_subjs=2, n_trials=250, n_params=3, n_datasets=1, include=include,
                                              estimators=outliers_estimators, view=view, n_outliers=[0,10,50])

        else:
            if run_trials:
                trial_exp = run_experiments(n_subjs=12, n_trials=list(np.arange(10, 100, 10)), n_params=5, n_datasets=5, include=include, view=view)
            if run_subjs:
                subj_exp = run_experiments(n_subjs=list(np.arange(2, 30, 2)), n_trials=20, n_params=5, n_datasets=5, include=include, view=view)
            if run_recovery:
                recovery_exp = run_experiments(n_subjs=12, n_trials=30, n_params=200, n_datasets=1, include=include, view=view)
            if run_outliers:
                outliers_exp = run_experiments(n_subjs=12, n_trials=250, n_params=2, n_datasets=1, include=include, view=view)

        if run_trials:
            trial_data = merge(trial_exp)
            trial_data.save('trial'+str(include)+'.dat')
        if run_subjs:
            subj_data = merge(subj_exp)
            subj_data.save('subj'+str(include)+'.dat')
        if run_recovery:
            recovery_data = merge(recovery_exp)
            recovery_data.save('recovery'+str(include)+'.dat')
        if run_outliers:
            outliers_data = merge(outliers_exp)
            outliers_data.save('outliers'+str(include)+'.dat')

    if result.load:
        if run_trials:
            trial_data = pd.load('trial'+str(include)+'.dat')
            trial_data['estimate'] = np.float64(trial_data['estimate'])
        if run_subjs:
            subj_data = pd.load('subj'+str(include)+'.dat')
            subj_data['estimate'] = np.float64(subj_data['estimate'])
        if run_recovery:
            recovery_data = pd.load('recovery'+str(include)+'.dat')
            recovery_data['estimate'] = np.float64(recovery_data['estimate'])
        if run_outliers:
            outliers_data = pd.load('outliers'+str(include)+'.dat')
            outliers_data['estimate'] = np.float64(outliers_data['estimate'])


    if result.analyze:
        if run_trials:
            plot_trial_exp(select(trial_data, include, subj=True))
            plt.savefig('trial_exp_subj'+str(include)+'.png')
            plt.savefig('trial_exp_subj'+str(include)+'.svg')

            plot_trial_exp(select(trial_data, include, subj=False))
            plt.savefig('trial_exp_group'+str(include)+'.png')
            plt.savefig('trial_exp_group'+str(include)+'.svg')

        if run_subjs:
            plot_subj_exp(select(subj_data, include, subj=True))
            plt.savefig('subj_exp_subj'+str(include)+'.png')
            plt.savefig('subj_exp_subj'+str(include)+'.svg')

            plot_subj_exp(select(subj_data, include, subj=False))
            plt.savefig('subj_exp_group'+str(include)+'.png')
            plt.savefig('subj_exp_group'+str(include)+'.svg')

        if run_recovery:
            plot_recovery_exp(select(recovery_data, include, subj=True), tag='subj'+str(include))
            plot_recovery_exp(select(recovery_data, include, subj=False), tag='group'+str(include), gridsize=50)

        if run_outliers:
            plot_outliers_exp(select(outliers_data, include, subj=True), tag='subj'+str(include), save=False)

    sys.exit(0)
