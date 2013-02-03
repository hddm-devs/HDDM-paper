from collections import OrderedDict
from copy import deepcopy, copy
import time
import argparse
import sys
import kabuki

import plots_utils as utils
import numpy as np
import pandas as pd

from plots_utils import select
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
                    estimators=None, p_outliers=(0,), view=None, depends_on = None, n_conds=4, equal_seeds=True, **kwargs):
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
                else:
                    exclude = None


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
                            'n_fast_outliers': n_fast_outliers, 'n_slow_outliers': n_slow_outliers}
                    if exclude is not None:
                        data['exclude_params'] = exclude

                    #creat kw_dict
                    kw_dict['data'] = data


                models_results = {}
                for model_name, descr in estimator_dict.iteritems():
                    kw_dict_model = deepcopy(kw_dict)
                    kw_dict_model['estimate'] = descr['params']
                    #run analysis
                    models_results[model_name] = recover(descr['estimator'], seed_data=1, seed_params=1, n_params=n_params,
                                                         n_datasets=n_datasets, kw_dict=kw_dict_model, view=view, equal_seeds=equal_seeds)

                factor3_results[cur_value] = models_results
            #end of (for cur_outliers in factor3_vals)

            n_trials_results[cur_trials] = factor3_results
        #end of (for cur_trials in n_trials)

        n_subjs_results[cur_subjs] = n_trials_results
    #end of (for cur_subjs in n_subjs)

    return n_subjs_results


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


    if run_regress:
        run_type = 'regress'
    elif run_trials:
        run_type = 'trials'
    elif run_subjs:
        run_type = 'subjs'
    elif run_outliers:
        run_type = 'outliers'
    elif run_recovery:
        run_type = 'recovery'

    if result.debug:
        fname = run_type + 'debug' + '.dat'
    else:
        fname = run_type + str(include) + '.dat'

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
                estimators = ['SingleMAP', 'Quantiles_subj']
                exp = run_experiments(n_subjs=12, n_trials=[5000], n_params=25, n_datasets=1, equal_seeds=True,
                                            include=include, view=view, depends_on = {'v':'condition'}, estimators=estimators)
            if run_subjs:
                exp = run_experiments(n_subjs=[6,7], n_trials=20, n_params=2, n_datasets=1, include=include,
                                           view=view, estimators=estimators, depends_on = {'v':'condition'})
            if run_recovery:
                exp = run_experiments(n_subjs=5, n_trials=30, estimators=estimators, n_params=2, n_datasets=1,
                                               include=include, view=view, depends_on = {'v':'condition'})
            if run_outliers:
                outliers_estimators = ['SingleMAP', 'SingleMAPoutliers', 'Quantiles_subj','HDDMOutliers']
                exp = run_experiments(n_subjs=[4], n_trials=(100), n_params=2, n_datasets=1, include=include,
                                              estimators=outliers_estimators, view=view, p_outliers=[0.06])
            if run_regress:
                regress_estimators = ['SingleRegressor', 'HDDMRegressor']
                include = ('a','v','t','sv')
                exp = run_experiments(n_subjs=10, n_trials=30, n_params=1, n_datasets=1, include=include,
                                              estimators=regress_estimators, view=view, effects=[0.1, 0.3])

        else:
            if run_trials:
                exp = run_experiments(n_subjs=12, n_trials=[30,40,50,75,100,150,250], n_params=25, n_datasets=1, equal_seeds=True,
                                            include=include, view=view, depends_on = {'v':'condition'}, estimators=estimators)
            if run_subjs:
                exp = run_experiments(n_subjs=list(np.arange(5, 31, 5)), n_trials=20, n_params=10, n_datasets=1, equal_seeds=True,
                                           include=include, view=view, depends_on = {'v':'condition'}, estimators=estimators)
            if run_recovery:
                exp = run_experiments(n_subjs=12, n_trials=30, n_params=200, n_datasets=1, include=include, equal_seeds=True,
                                               view=view, depends_on = {'v':'condition'}, estimators=estimators)
            if run_outliers:
                outliers_estimators = ['HDDMsharedVar', 'HDDMOutliers', 'Quantiles_subj', 'Quantiles_group']
                exp = run_experiments(n_subjs=(5,10,15), n_trials=[250,], n_params=25, n_datasets=1, include=include, equal_seeds=True,
                                              estimators=outliers_estimators, depends_on = {'v':'condition'}, view=view, p_outliers=[0.06])
            if run_regress:
                regress_estimators = ['SingleRegressor', 'HDDMRegressor']
                include = ('a','v','t','sv')
                exp = run_experiments(n_subjs=12, n_trials=[30,40,50,75,100,150,250], n_params=25, n_datasets=1, include=include,
                                              estimators=regress_estimators, view=view, effects=[0.1, 0.3, 0.5])

        data = merge(exp)
        if not run_regress:
            data = est.add_group_stat_to_SingleOptimation(data, np.mean)
            data = est.use_group_truth_value_for_subjects_in_HDDMsharedVar(data)
        data.save(fname)

    if result.load:
        data = pd.load(fname)
        if run_regress:
            data['estimate'] = np.float64(data['estimate'])
            data = est.add_group_stat_to_SingleRegressor(data)

        data['estimate'] = np.float64(data['estimate'])

        try:
            bad = data[(data.estimate < 1e-5) & (data.estimate > 0) & (data['std'] < 1e-10)]
            print "Found %d problematic experiments" % len(bad)
            print len(data)
            for i in bad.index:
                print i
                t_bad = data.select(lambda x: x[:-1] == i[:-1]).index
                data = data.drop(labels=t_bad)
            print len(data)
        except KeyError:
            print "cound not run problems detection"



    if result.analyze:
        if run_subjs or run_trials:
            # include = ['v','a']
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
                plot_type = 'subjs'
            else:
                plot_type = 'trials'

            utils.plot_exp(select(data, include, depends_on=depends_on, subj=True) , stat=stat, plot_type=plot_type, figname='single_' + figname, savefig=savefig)
            utils.plot_exp(select(data, include, depends_on=depends_on, subj=False), stat=stat, plot_type=plot_type, figname='group_' + figname, savefig=savefig)

        if run_regress:
            utils.likelihood_of_detection(data, subj=False, savefig=savefig)
            utils.likelihood_of_detection(data, subj=True, savefig=savefig)

        if run_recovery:
#            one_vs_others(select(recovery_data, include, subj=False), main_estimator='HDDMTruncated', tag='group'+str(include), save=False)
            utils.plot_recovery_exp(select(data, include, subj=True), tag='subj'+str(include))
            utils.plot_recovery_exp(select(data, include, subj=False), tag='group'+str(include), gridsize=50)

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

            utils.plot_exp(select(data, include, depends_on=depends_on, subj=True) , stat=stat, plot_type='subjs', figname='single_outliers_' + figname, savefig=savefig)
            utils.plot_exp(select(data, include, depends_on=depends_on, subj=False), stat=stat, plot_type='subjs', figname='group_outliers_' + figname, savefig=savefig)

#            one_vs_others(select(outliers_data, include, depends_on={},subj=True), main_estimator='SingleMAPoutliers', tag='subj'+str(include), save=savefig)

    sys.exit(0)
