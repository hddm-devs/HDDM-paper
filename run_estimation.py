from collections import OrderedDict
from copy import deepcopy, copy
import pprint
import time
import argparse
import sys
import kabuki
import os
import plots_utils as utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
                    estimators=None, view=None, depends_on = None, equal_seeds=True, run_type=None,
                    factor3_vals = None, action='run', single_runs_folder='.', subj_noise=None,
                    seed_data=1, seed_params=1):
    if not isinstance(n_subjs, (tuple, list, np.ndarray)):
        n_subjs = (n_subjs,)
    if not isinstance(n_trials, (tuple, list, np.ndarray)):
        n_trials = (n_trials,)
    if depends_on is None:
        depends_on = {}

    #kwargs for initialize estimation
    init = OrderedDict([('include', include), ('depends_on', depends_on)])

    #kwargs for estimation
    estimate = OrderedDict([('runs', 3)])

    #include params
    params = OrderedDict([('include', include)])
    recover = est.multi_recovery_fixed_n_trials

    estimator_dict = OrderedDict()
    hddm_sampling_params = OrderedDict([('samples', 1500), ('burn', 500), ('map', False)])
    optimizations_params = OrderedDict([('method', 'ML'), ('quantiles', (0.1, 0.3, 0.5, 0.7, 0.9)), ('n_runs', 50)])
    if 'SingleMAP' in estimators:
        estimator_dict['SingleMAP'] = OrderedDict([('estimator', est.EstimationSingleMAP), ('params', {'runs': 50})])

    if 'SingleMAPoutliers' in estimators:
        estimator_dict['SingleMAPoutliers'] = OrderedDict([('estimator', est.EstimationSingleMAPoutliers), ('params', {'runs': 50})])

    if 'HDDMsharedVar' in estimators:
        estimator_dict['HDDMsharedVar'] = OrderedDict([('estimator', est.EstimationHDDMsharedVar), ('params', hddm_sampling_params)])

    if 'HDDMGamma' in estimators:
        estimator_dict['HDDMGamma'] = OrderedDict([('estimator', est.EstimationHDDMGamma), ('params', hddm_sampling_params)])

    if 'noninformHDDM' in estimators:
        estimator_dict['noninformHDDM'] = OrderedDict([('estimator', est.EstimationNoninformHDDM), ('params', hddm_sampling_params)])

    if 'HDDMOutliers' in estimators:
        estimator_dict['HDDMOutliers'] = OrderedDict([('estimator', est.EstimationHDDMOutliers), ('params', hddm_sampling_params)])

    if 'HDDMRegressor' in estimators:
        estimator_dict['HDDMRegressor'] = OrderedDict([('estimator', est.EstimationHDDMRegressor), ('params', hddm_sampling_params)])

    if 'HDDM2' in estimators:
        estimator_dict['HDDM2'] = OrderedDict([('estimator', est.EstimationHDDM2), ('params', hddm_sampling_params)])

    if 'SingleRegressor' in estimators:
        estimator_dict['SingleRegressor'] = OrderedDict([('estimator', est.SingleRegressor), ('params', hddm_sampling_params)])

    if 'HDDM2Single' in estimators:
        estimator_dict['HDDM2Single'] = OrderedDict([('estimator', est.EstimationHDDM2Single), ('params', hddm_sampling_params)])

    if 'HDDMTruncated' in estimators:
        estimator_dict['HDDMTruncated'] = OrderedDict([('estimator', est.EstimationHDDMTruncated), ('params', hddm_sampling_params)])

    if 'Quantiles_subj' in estimators:
        opt_params = deepcopy(optimizations_params)
        opt_params['method'] = 'chisquare'
        estimator_dict['Quantiles_subj'] = OrderedDict([('estimator', est.EstimationSingleOptimization), ('params', opt_params)])

    if 'ML' in estimators:
        opt_params = deepcopy(optimizations_params)
        opt_params['method'] = 'ML'
        estimator_dict['ML'] = OrderedDict([('estimator', est.EstimationSingleOptimization), ('params', opt_params)])

    if 'Quantiles_group' in estimators:
        opt_params = deepcopy(optimizations_params)
        opt_params['method'] = 'chisquare'
        estimator_dict['Quantiles_group'] = OrderedDict([('estimator', est.EstimationGroupOptimization), ('params', opt_params)])

    if 'MLRegressor' in estimators:
        opt_params = deepcopy(optimizations_params)
        opt_params['method'] = 'ML'
        estimator_dict['MLRegressor'] = OrderedDict([('estimator', est.SingleRegOptimization), ('params', opt_params)])

    n_subjs_results = {}
    for cur_subjs in n_subjs:
        n_trials_results = {}
        for cur_trials in n_trials:

            factor3_results = {}
            for cur_value in factor3_vals:

                #if regress experiments then we add an effect
                if run_type == 'regress':
                    params['effect'] = cur_value

                if run_type == 'priors':
                    n_conds = cur_value
                else:
                    n_conds = 2

                #create kw_dict
                kw_dict = OrderedDict([('params', params), ('init', init), ('estimate', estimate), ('n_conds', n_conds)])

                #exclude params
                if run_type == 'regress':
                    exclude = set(['sv','st','sz','z', 'reg_outcomes'])
                else:
                    exclude = set(['sv','st','sz','z']) - set(include)

                #create kw_dict['data']
                if run_type == 'outliers':
                    cur_outliers = cur_value
                else:
                    cur_outliers = 0
                n_outliers = int(cur_trials * cur_outliers)
                n_fast_outliers = (n_outliers // 2)
                n_slow_outliers = n_outliers - n_fast_outliers
                data = OrderedDict([('subjs', cur_subjs), ('subj_noise', subj_noise), ('size', cur_trials - n_outliers),
                        ('exclude_params', exclude)])
                if run_type != 'regress':
                        data['n_fast_outliers'] = n_fast_outliers
                        data['n_slow_outliers'] = n_slow_outliers

                #creat kw_dict
                kw_dict['data'] = data


                models_results = {}
                for model_name, descr in estimator_dict.iteritems():

                    #create kw_dict
                    kw_dict_model = deepcopy(kw_dict)
                    kw_dict_model['estimate'] = descr['params']

                    #update it with regressor information if needed
                    if model_name in est.MODELS_WITH_REGRESSORS:
                        reg_func = lambda args, cols: args[0]*cols[:,0]+args[1]
                        if run_type == 'regress':
                            reg = {'func': reg_func, 'args':['v_slope','v_inter'], 'covariates': 'cov', 'outcome':'v'}
                        else:
                            reg = {'func': reg_func, 'args':['v_shift','v(c0)'], 'covariates': 'condition', 'outcome':'v'}
                        reg = OrderedDict(sorted(reg.items(), key=lambda t: t[0]))
                        kw_dict_model['init']['regressor'] = reg
                        kw_dict_model['init']['depends_on'] = {}

                    #run analysis
                    models_results[model_name] = recover(descr['estimator'], seed_data=seed_data, seed_params=seed_params, n_params=n_params,
                                                         n_datasets=n_datasets, kw_dict=kw_dict_model, view=view, run_type=run_type,
                                                         equal_seeds=equal_seeds, action=action, single_runs_folder=single_runs_folder)

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
    parser.add_argument('--regress', action='store_true', dest='regress', default=False,
                        help='Run only regression estimations.')
    parser.add_argument('--priors', action='store_true', dest='priors', default=False,
                        help='Run only priors experiment.')
    parser.add_argument('--debug', action='store_true', dest='debug', default=False)
    parser.add_argument('--savefig', action='store_true', dest='savefig', default=False)
    parser.add_argument('--action', action='store', dest='action', default='run',
                        help='Which action to do: run/collect/delete')
    parser.add_argument('--folder', action='store', dest='folder', default='current',
                        help='Which folder are the simulations going to be saved to/loaded from')



    result = parser.parse_args()


    run_trials, run_subjs, run_recovery, run_outliers = result.trials, result.subjs, result.recovery, result.outliers
    run_priors = result.priors
    run_regress = result.regress
    savefig = result.savefig
    action = result.action
    folder = result.folder

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
    elif run_priors:
        run_type = 'priors'
    else:
        raise ValueError("run_type was not found")

    main_folder = os.path.join('simulations',run_type, folder)
    single_runs_folder = os.path.join(main_folder, 'single_runs')
    summary_folder = os.path.join(main_folder, 'summary')

    if result.debug:
        fname = os.path.join(summary_folder, run_type + '_debug' + '.dat')
    else:
        fname = os.path.join(summary_folder, run_type + '.dat')

    if result.parallel:
        c = parallel.Client(profile=result.profile)
        view = c.load_balanced_view()
    else:
        view = None


    #load arguments
    sys.path.insert(0, main_folder)
    try:
        del sys.modules['args']
    except KeyError:
        pass
    import args
    exp_kwargs = args.args()


    #run
    if result.run:

        #run experiment
        pprint.pprint(exp_kwargs)
        exp = run_experiments(view=view, action=action, single_runs_folder=single_runs_folder,
                              **exp_kwargs)

        #collect data
        if action == 'collect':
            data = merge(exp)
            if not run_regress:
                estimators=('HDDM2Single', 'Quantiles_subj', 'ML')
                data = est.add_group_stat_to_SingleOptimation(data, np.mean, estimators=estimators)
                data = est.add_var_to_SingleOptimation(data, estimators=estimators)
                data['err'] = np.asarray((data['estimate'] - data['truth']), dtype=np.float32)
                data['abserr'] = np.abs(data['err'])

            data.save(fname)

    if result.load:
        data = pd.load(fname)
        data.index.names[-1] = 'param'
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
            # idx = ~np.isnan(data['50q'])
            # data['estimate'][idx] = data['50q'][idx]
            # include = ['v','a']
            params = set(['a','v','t']).union(args.args()['include'])
            depends_on= {'v': ['c0', 'c1']}
            # stat = np.mean
            stat = utils.trimmed_mean

            #create figname
            figname = stat.__name__

            if run_subjs:
                plot_type = 'subjs'
            else:
                plot_type = 'trials'

            estimators = ['HDDM2', 'HDDM2Single', 'Quantiles_subj', 'ML']
            utils.plot_exp(select(data, params, depends_on=depends_on, subj=True, estimators=estimators),
                           stat=stat, plot_type=plot_type,
                           figname='single_' + figname, savefig=savefig)

            estimators += ['Quantiles_group']
            utils.plot_exp(select(data, params, depends_on=depends_on, subj=False, estimators=estimators),
                           stat=stat, plot_type=plot_type,
                           figname='group_' + figname, savefig=savefig)

            utils.likelihood_of_detection(data, plot_type=plot_type, savefig=savefig)

            var_params = ['v_var', 'a_var', 't_var']
            stat = utils.trimmed_mean
            utils.plot_exp(select(data, var_params, depends_on=depends_on, subj=False, estimators=estimators),
                           stat=stat, plot_type=plot_type, col='err',
                           figname='variance_err_' + stat.__name__, savefig=savefig)

        if run_priors:
            # idx = ~np.isnan(data['50q'])
            # data['estimate'][idx] = data['50q'][idx]

            stat=utils.upper_trimmed_mean
            estimators = ['HDDMGamma', 'ML', 'Quantiles_subj']
            include = ['a','v','t','z']
            # include = ['sz','st', 'sv']

            #create figname
            figname = stat.__name__

            for i in [2, 3]:
                depends_on= {'v': ['c0', 'c1', 'c2'][:i]}
                selected_data = select(data, include, depends_on=depends_on, subj=False, estimators=estimators)
                utils.plot_exp(selected_data.xs(i, level='p_outliers'), stat=stat, plot_type=run_type,
                                figname='_' + figname, savefig=savefig)

        if run_regress:
            utils.likelihood_of_detection(data, plot_type='regress', savefig=savefig)

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

    plt.show()
    sys.exit(0)
