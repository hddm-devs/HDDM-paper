from collections import OrderedDict
from copy import deepcopy, copy
import time
import argparse
import sys
import kabuki

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid

import numpy as np
import pandas as pd

from IPython import parallel
from IPython.parallel.client.asyncresult import AsyncResult
import estimate as est

#COLORS = {'HDDMsharedVar': 'Blue', 'HDDMTruncted': 'Brown', 'Quantiles_group': 'BurlyWood', 'Quantiles_subj': 'CadetBlue',
#          'SingleMAP': 'Chartreuse', 'SingleMAPOutliers': 'red'}

#PARAM_NAMES = {'a': 'threshold',
#               'v': 'drift-rate',
#               't': 'non-decision time',
#               'z': 'bias',
#               'st': 'non-decision time var.',
#               'sz': 'bias var.'}

PARAM_NAMES = {'a': 'a',
               'v': 'v',
               't': 't',
               'z': 'z',
               'st': 'st',
               'sz': 'sz',
               'sv': 'sv'}

def run_experiments(n_subjs=(12,), n_trials=(10, 40, 100), n_params=5, n_datasets=5, include=('v','t','a'),
                    estimators=None, p_outliers=(0,), view=None, depends_on = None, n_conds=4, **kwargs):
    if not isinstance(n_subjs, (tuple, list, np.ndarray)):
        n_subjs = (n_subjs,)
    if not isinstance(n_trials, (tuple, list, np.ndarray)):
        n_trials = (n_trials,)
    if depends_on is None:
        depends_on = {}

    effects = kwargs.get('effects', None)
    factor3_vals = p_outliers;
    if effects is None:
        is_regress = False
    else:
        is_regress = True
        factor3_vals = effects

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
    if is_regress:
        subj_noise.update({'v_slope':0.2, 'v_inter':0.2})

    #kwargs for initialize estimation
    init = {'include': include, 'depends_on': depends_on}

    #kwargs for estimation
    estimate = {'runs': 3}

    #include params
    params = {'include': include, 'n_conds': n_conds}
    recover = est.multi_recovery_fixed_n_trials

    estimator_dict = OrderedDict()
    if 'SingleMAP' in estimators:
        estimator_dict['SingleMAP'] = {'estimator': est.EstimationSingleMAP, 'params': {'runs': 50}}
    if 'SingleMAPoutliers' in estimators:
        estimator_dict['SingleMAPoutliers'] = {'estimator': est.EstimationSingleMAPoutliers, 'params': {'runs': 50}}

    if 'HDDMsharedVar' in estimators:
        estimator_dict['HDDMsharedVar'] = {'estimator': est.EstimationHDDMsharedVar, 'params': {'samples': 35000, 'burn': 30000, 'map': True}}

    if 'HDDMGamma' in estimators:
        estimator_dict['HDDMGamma'] = {'estimator': est.EstimationHDDMGamma, 'params': {'samples': 35000, 'burn': 30000, 'map': True}}

    if 'HDDMOutliers' in estimators:
        estimator_dict['HDDMOutliers'] = {'estimator': est.EstimationHDDMOutliers, 'params': {'samples': 35000, 'burn': 30000, 'map': True}}

    if 'HDDMRegressor' in estimators:
        estimator_dict['HDDMRegressor'] = {'estimator': est.EstimationHDDMRegressor, 'params': {'samples': 35000, 'burn': 30000, 'map': False}}

    if 'SingleRegressor' in estimators:
        estimator_dict['SingleRegressor'] = {'estimator': est.SingleRegressor, 'params': {'samples': 35000, 'burn': 30000, 'map': False}}

    if 'HDDMTruncated' in estimators:
        estimator_dict['HDDMTruncated'] = {'estimator': est.EstimationHDDMTruncated, 'params': {'samples': 35000, 'burn': 30000, 'map': True}}

    if 'Quantiles_subj' in estimators:
        estimator_dict['Quantiles_subj'] = {'estimator': est.EstimationSingleOptimization,
                                           'params': {'method': 'chisquare', 'quantiles': (0.1, 0.3, 0.5, 0.7, 0.9), 'n_runs': 50}}
    if 'Quantiles_group' in estimators:
        estimator_dict['Quantiles_group'] = {'estimator': est.EstimationGroupOptimization,
                                            'params': {'method': 'chisquare', 'quantiles': (0.1, 0.3, 0.5, 0.7, 0.9), 'n_runs': 50}}


    n_subjs_results = {}
    for cur_subjs in n_subjs:
        n_trials_results = {}
        for cur_trials in n_trials:

            factor3_results = {}
            for cur_value in factor3_vals:

                #create kw_dict
                kw_dict = {'params': params, 'init': init, 'estimate': estimate}

                #if this is not a full model we should add exclude params
                if (set(['sv','st','sz','z','a','v','t']) != set(include)) or is_regress:
                    if is_regress:
                        exclude = set(['sv','st','sz','z', 'reg_outcomes']) - set(include)
                    else:
                        exclude = set(['sv','st','sz','z']) - set(include)


                #create kw_dict['data']
                if is_regress:
                    reg_func = lambda args, cols: args[0]*cols[:,0]+args[1]
                    reg = {'func': reg_func, 'args':['v_slope','v_inter'], 'covariates': 'cov', 'outcome':'v'}
                    init['regressor'] = reg
                    kw_dict['init'] = init

                    reg_data = {'subjs': cur_subjs, 'subj_noise': subj_noise, 'size': cur_trials,
                                'effect': cur_value, 'exclude_params': exclude}
                    kw_dict['reg_data'] = reg_data

                else:
                    cur_outliers = cur_value
                    n_outliers = int(cur_trials * cur_outliers)
                    n_fast_outliers = (n_outliers // 2)
                    n_slow_outliers = n_outliers - n_fast_outliers
                    data = {'subjs': cur_subjs, 'subj_noise': subj_noise, 'size': cur_trials - n_outliers,
                            'n_fast_outliers': n_fast_outliers, 'n_slow_outliers': n_slow_outliers, 'exclude_params': exclude}

                    #creat kw_dict
                    kw_dict['data'] = data


                models_results = {}
                for model_name, descr in estimator_dict.iteritems():
                    kw_dict_model = deepcopy(kw_dict)
                    kw_dict_model['estimate'] = descr['params']
                    #run analysis
                    models_results[model_name] = recover(descr['estimator'], seed_data=1, seed_params=1, n_params=n_params,
                                                         n_datasets=n_datasets, kw_dict=kw_dict_model, view=view)

                factor3_results[cur_value] = models_results
            #end of (for cur_outliers in factor3_vals)

            n_trials_results[cur_trials] = factor3_results
        #end of (for cur_trials in n_trials)

        n_subjs_results[cur_subjs] = n_trials_results
    #end of (for cur_subjs in n_subjs)

    return n_subjs_results


def plot_exp(data, stat, plot_type, figname, savefig):
    if plot_type == 'subjs':
        level_name = 'n_subjs'
        xlabel = 'subjs'
    elif plot_type == 'trials':
        level_name = 'n_trials'
        xlabel = 'trials'
    else:
        raise ValueError('unknown plot_type')
    grouped = data.Err.dropna().groupby(level=(level_name, 'estimation', 'knode')).agg((stat, np.std))
    n_params = len(grouped.groupby(level=('knode',)).groups.keys())

    fig = plt.figure(figsize=(8, n_params*3))
    grid = Grid(fig, 111, nrows_ncols=(n_params, 1), add_all=True, share_all=False,
                label_mode='L', share_x=True, share_y=False, axes_pad=.25)

    for i, (param_name, param_data) in enumerate(grouped.groupby(level=('knode',))):
        ax = grid[i]
        ax.set_ylabel(param_name)
#        ax.set_ylabel(PARAM_NAMES[param_name])
#        ax.set_xlim(2, 30)
#        ax.set_yscale('log')
        for est_name, est_data in param_data.groupby(level=['estimation']):
            ax.errorbar(est_data.index.get_level_values(level_name),
                        est_data[stat.__name__], label=est_name, lw=2.,
                        marker='o')

    ax.set_xlabel(xlabel)
    plt.legend(loc=0)
    title = '%s_exp_%s'%(plot_type, figname)
    plt.suptitle(title)

    if savefig:
        plt.savefig(title + '.png')
        plt.savefig(title + '.svg')


def plot_recovery_exp(data, tag='', abs_min=-5, abs_max=5, gridsize=100, save=True):

    data = data[['truth', 'estimate','Err']].dropna()
    ni = len(data.dropna().groupby(level=['estimation']))
    nj = len(data.dropna().groupby(level=('knode',)))

    data = data[(data['estimate'] > abs_min) & (data['estimate'] < abs_max)]
    data_params = data.groupby(level=('knode',))[['truth', 'estimate']]
    mini = data_params.min().min(axis=1)
    maxi = data_params.max().max(axis=1)
    print mini
    print maxi

    fig = plt.figure(figsize=(9, 3*nj))
    grid = Grid(fig, 111, nrows_ncols=(nj, ni), add_all=True, share_all=False, label_mode='L', share_x=False, share_y=False)
    for i, (est_name, est_data) in enumerate(data.dropna().groupby(level=['estimation'])):
        nj = len(est_data.groupby(level=('knode',)))
        for j, (param_name, param_data) in enumerate(est_data.groupby(level=('knode',))):
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
#            kabuki.debug_here()

#            plt.legend()

    if save:
        plt.savefig('recovery_exp_%s.png'%(tag), dpi=600)
        plt.savefig('recovery_exp_%s.svg'%(tag))

def one_vs_others(data, main_estimator, tag='', gridsize=100, save=True):

    data = data[['truth', 'estimate','Err']].dropna()
    data_params = data.groupby(level=('knode',))[['truth', 'estimate']]
    mini = data_params.min().min(axis=1)
    maxi = data_params.max().max(axis=1)
    print mini
    print maxi

    grouped_data = data.groupby(level=['estimation'])
    main_data = grouped_data.get_group(main_estimator)
    ni = len(grouped_data) - 1
    nj = len(data.groupby(level=('knode',)))
    fig = plt.figure()#figsize=(9, 3*nj))
    counter = 0
    for j, (param_name, param_data) in enumerate(data.groupby(level=('knode',))):
        for i, (est_name, est_data) in enumerate(param_data.groupby(level=('estimation',))):
            if est_name == main_estimator:
                continue
            counter = counter + 1
            ax = fig.add_subplot(nj, ni, counter)
            minimaxi = (mini[param_name], maxi[param_name])
#            ax.set_xlim(minimaxi)
#            ax.set_ylim(minimaxi)
            ax.set_xlabel(est_name)
            ax.set_ylabel(PARAM_NAMES[param_name])
            kwargs = {'gridsize': gridsize, 'bins': 'log'}
#                      'extent': (mini[param_name], maxi[param_name], mini[param_name], maxi[param_name])}
#            ax.hexbin(np.abs(est_data.relErr), main_data.ix[param_name].relErr, **kwargs)
#            ax.scatter(np.abs(est_data.relErr), np.abs(main_data.ix[param_name].relErr))
            ax.scatter(est_data.Err, main_data.ix[param_name].Err)
            lb = min(ax.get_xlim()[0], ax.get_ylim()[0])
            ub = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([lb, ub], [lb, ub])
#            ax.axis('equal')
            ax.plot()


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
    results = concat_dicts(data, names=['n_subjs', 'n_trials', 'p_outliers', 'estimation', 'param_seed', 'data_seed', 'param'])
    return results

def select(stats, param_names, depends_on, subj=True):

    if isinstance(param_names, str):
        param_names = [param_names]

    if subj:
        estimators = ['SingleMAP', 'HDDMsharedVar', 'HDDMTruncated', 'Quantiles_subj', 'SingleMAPoutliers', 'HDDMOutliers']
    else:
        estimators = ['HDDMTruncated', 'Quantiles_group', 'HDDMsharedVar', 'HDDMOutliers', 'Quantiles_subj']

    extracted = {}
    index = stats.index
    for name in param_names:
        for cond in depends_on.get(name, [None]):
            if (cond is not None):
                fullname = "%s(%s)" % (name, cond)
            else:
                fullname = name

            selected = []
            for ix in index:
                if subj:
                    if ix[-4] in estimators and ix[-1].startswith(name) and 'subj' in ix[-1]:
                        if (cond is None) or (('(%s)' % cond) in ix[-1]):
                            selected.append(ix)
                else:
                    if ix[-4] in estimators and ((ix[-1] == name) or (ix[-1].startswith(name + '('))):
                        if (cond is None) or (('(%s)' % cond) in ix[-1]):
                            selected.append(ix)

            extracted[fullname] = stats.ix[selected]

    return pd.concat(extracted, names=['knode'])



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
    parser.add_argument('--regress', action='store_true', dest='regress', default=False,
                        help='Run only regression estimations.')
    parser.add_argument('-st', action='store_true', dest='st', default=False)
    parser.add_argument('-sv', action='store_true', dest='sv', default=False)
    parser.add_argument('-sz', action='store_true', dest='sz', default=False)
    parser.add_argument('-z', action='store_true', dest='z', default=False)
    parser.add_argument('--full', action='store_true', dest='full', default=False)
    parser.add_argument('--debug', action='store_true', dest='debug', default=False)
    parser.add_argument('--discardfig', action='store_true', dest='discardfig', default=False)

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
    run_regress = result.regress
    savefig = not result.discardfig

    if result.all:
        run_trials, run_subjs, run_recovery = True, True, True

    if result.parallel:
        c = parallel.Client(profile=result.profile)
        view = c.load_balanced_view()
    else:
        view = None

    if result.run:
        if result.group:
            estimators = ['HDDMTruncated','Quantiles_group', 'HDDMsharedVar']
        else:
            estimators = set(['SingleMAP', 'HDDMTruncated', 'Quantiles_subj',
            'Quantiles_group','HDDMsharedVar', 'HDDMGamma'])
            if not result.full:
                estimators.remove('HDDMTruncated')

        if result.debug:
            if run_trials:
                estimators = ['HDDMGamma']
                trial_exp = run_experiments(n_subjs=6, n_trials=[50,100], estimators=estimators, n_params=2, n_datasets=1,
                                            include=include, view=view, depends_on = {'v':'condition'})
            if run_subjs:
                subj_exp = run_experiments(n_subjs=[6,7], n_trials=20, n_params=2, n_datasets=1, include=include,
                                           view=view, estimators=estimators, depends_on = {'v':'condition'})
            if run_recovery:
                recovery_exp = run_experiments(n_subjs=5, n_trials=30, estimators=estimators, n_params=2, n_datasets=1,
                                               include=include, view=view, depends_on = {'v':'condition'})
            if run_outliers:
                outliers_estimators = ['SingleMAP', 'SingleMAPoutliers', 'Quantiles_subj','HDDMOutliers']
                outliers_exp = run_experiments(n_subjs=[4], n_trials=(100), n_params=2, n_datasets=1, include=include,
                                              estimators=outliers_estimators, view=view, p_outliers=[0.06])
            if run_regress:
                regress_estimators = ['SingleRegressor', 'HDDMRegressor']
                include = ('a','v','t','sv')
                regress_exp = run_experiments(n_subjs=10, n_trials=30, n_params=1, n_datasets=1, include=include,
                                              estimators=regress_estimators, view=view, effects=[0.1, 0.3])

        else:
            if run_trials:
                trial_exp = run_experiments(n_subjs=12, n_trials=list(np.arange(10, 100, 10)) + [150,250], n_params=5, n_datasets=5,
                                            include=include, view=view, depends_on = {'v':'condition'}, estimators=estimators)
            if run_subjs:
                subj_exp = run_experiments(n_subjs=list(np.arange(5, 31, 5)), n_trials=20, n_params=10, n_datasets=1,
                                           include=include, view=view, depends_on = {'v':'condition'}, estimators=estimators)
            if run_recovery:
                recovery_exp = run_experiments(n_subjs=12, n_trials=30, n_params=200, n_datasets=1, include=include,
                                               view=view, depends_on = {'v':'condition'}, estimators=estimators)
            if run_outliers:
                outliers_estimators = ['HDDMsharedVar', 'HDDMOutliers', 'Quantiles_subj', 'Quantiles_group']
                outliers_exp = run_experiments(n_subjs=(5,10,15), n_trials=[250,], n_params=25, n_datasets=1, include=include,
                                              estimators=outliers_estimators, depends_on = {'v':'condition'}, view=view, p_outliers=[0.06])
            if run_regress:
                regress_estimators = ['SingleRegressor', 'HDDMRegressor']
                include = ('a','v','t','sv')
                regress_exp = run_experiments(n_subjs=12, n_trials=np.arange(30,250,30), n_params=25, n_datasets=1, include=include,
                                              estimators=regress_estimators, view=view, effects=[0.1, 0.3, 0.5])

        if run_trials:
            trials_data = merge(trial_exp)
            trials_data = est.add_group_stat_to_SingleOptimation(trials_data, np.mean)
            trials_data = est.use_group_truth_value_for_subjects_in_HDDMsharedVar(trials_data)
            trials_data.save('trial'+str(include)+'.dat')
        if run_subjs:
            subj_data = merge(subj_exp)
            subj_data = est.add_group_stat_to_SingleOptimation(subj_data, np.mean)
            subj_data = est.use_group_truth_value_for_subjects_in_HDDMsharedVar(subj_data)
            subj_data.save('subj'+str(include)+'.dat')
        if run_recovery:
            recovery_data = merge(recovery_exp)
            recovery_data = est.add_group_stat_to_SingleOptimation(recovery_data, np.mean)
            recovery_data = est.use_group_truth_value_for_subjects_in_HDDMsharedVar(recovery_data)
            recovery_data.save('recovery'+str(include)+'.dat')
        if run_outliers:
            outliers_data = merge(outliers_exp)
            outliers_data = est.add_group_stat_to_SingleOptimation(outliers_data, np.mean)
            outliers_data = est.use_group_truth_value_for_subjects_in_HDDMsharedVar(outliers_data)
            outliers_data.save('outliers'+str(include)+'.dat')
        if run_regress:
            regress_data = merge(regress_exp)
            regress_data.save('regress'+str(include)+'.dat')

    if result.load:
        if run_trials:
            trials_data = pd.load('trial'+str(include)+'.dat')
            trials_data['estimate'] = np.float64(trials_data['estimate'])
        if run_subjs:
            subj_data = pd.load('subj'+str(include)+'.dat')
            subj_data['estimate'] = np.float64(subj_data['estimate'])
        if run_recovery:
            recovery_data = pd.load('recovery'+str(include)+'.dat')
            recovery_data['estimate'] = np.float64(recovery_data['estimate'])
        if run_outliers:
            outliers_data = pd.load('outliers'+str(include)+'.dat')
            outliers_data['estimate'] = np.float64(outliers_data['estimate'])
        if run_regress:
            regress_data = pd.load('regress'+str(include)+'.dat')
            regress_data['estimate'] = np.float64(regress_data['estimate'])



    if result.analyze:
        if run_subjs or run_trials:
            include = ['v','a']
            depends_on= {'v': ['c0', 'c1', 'c2', 'c3']}
            stat=np.median

            #create figname
            figname = ''
            if result.full:
                figname += 'full'
            else:
                figname += 'simple'
            figname += ('_' + stat.__name__)

        if run_subjs:

            plot_exp(select(subj_data, include, depends_on=depends_on, subj=True) , stat=stat, plot_type='subjs', figname='single_' + figname, savefig=savefig)
            plot_exp(select(subj_data, include, depends_on=depends_on, subj=False), stat=stat, plot_type='subjs', figname='group_' + figname, savefig=savefig)
        if run_trials:
            plot_exp(select(trials_data, include, depends_on=depends_on, subj=True) , stat=stat, plot_type='trials', figname='single_' + figname, savefig=savefig)
            plot_exp(select(trials_data, include, depends_on=depends_on, subj=False), stat=stat, plot_type='trials', figname='group_' + figname, savefig=savefig)



        if run_recovery:
#            one_vs_others(select(recovery_data, include, subj=False), main_estimator='HDDMTruncated', tag='group'+str(include), save=False)
            plot_recovery_exp(select(recovery_data, include, subj=True), tag='subj'+str(include))
            plot_recovery_exp(select(recovery_data, include, subj=False), tag='group'+str(include), gridsize=50)

        if run_outliers:
            depends_on= {'v': ['c0', 'c1', 'c2', 'c3']}
            stat=np.median

            #create figname
            figname = ''
            if result.full:
                figname += 'full'
            else:
                figname += 'simple'
            figname += ('_' + stat.__name__)

            plot_exp(select(outliers_data, include, depends_on=depends_on, subj=True) , stat=stat, plot_type='subjs', figname='single_outliers_' + figname, savefig=savefig)
            plot_exp(select(outliers_data, include, depends_on=depends_on, subj=False), stat=stat, plot_type='subjs', figname='group_outliers_' + figname, savefig=savefig)

#            one_vs_others(select(outliers_data, include, depends_on={},subj=True), main_estimator='SingleMAPoutliers', tag='subj'+str(include), save=savefig)

    sys.exit(0)
