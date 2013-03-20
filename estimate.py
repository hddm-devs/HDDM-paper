import pprint
import hashlib
import cPickle
import hddm
import my_hddm
import numpy as np
import pandas as pd
import copy
import os
import os.path
import time
import glob
import generate_regression as genreg
import scipy
import pymc as pm
import traceback

from my_hddm_regression import HDDMRegressor
from scipy.optimize import fmin_powell
from multiprocessing import Pool
from pandas import DataFrame

MODELS_WITH_REGRESSORS = ['HDDMRegressor', 'SingleRegressor', 'HDDM2', 'HDDM2Single', 'MLRegressor']
ESTIMATTIONS_WITH_REGRESSORS = ['EstimationHDDMRegressor', 'EstimationHDDM2', 'EstimationHDDM2Single',
                                'SingleRegressor', 'SingleRegOptimization']


# For each method that you would like to check you need do create a ass that inherits from
# the Estimation class and implement the estimate and get_stats attributes.


class Estimation(object):
    def __init__(self, data, include=(), pool_size=1, **kwargs):

        #save args
        self.data = data
        self.include = include
        self.stats = {}

#HDDM Estimation
class EstimationHDDMBase(Estimation):

    def __init__(self, data, **kwargs):
        super(EstimationHDDMBase, self).__init__(data, **kwargs)
        self.init_kwargs = kwargs.copy()
        self.init_model(data)

    def init_model(self, data):
            pass

    def estimate(self, **kwargs):
        samples = kwargs.pop('samples', 10000)

        if kwargs.pop('map', False):
            try:
                self.model.approximate_map(fall_to_simplex=False)
            except FloatingPointError: #in case of error we have to reinit the model
                self.init_model(self.model.data)

        self.model.sample(samples, **kwargs)
        self.geweke_problem = geweke_test_problem(self.model)

    def get_stats(self):
        stats = self.model.gen_stats()

        return stats



#HDDM Estimation
class EstimationHDDMTruncated(EstimationHDDMBase):

    def __init__(self, data, **kwargs):
        super(EstimationHDDMTruncated, self).__init__(data, **kwargs)

    def init_model(self, data):
            self.model = hddm.HDDMTruncated(data, **self.init_kwargs)


#HDDM Estimation
class EstimationHDDMsharedVar(EstimationHDDMBase):

    def __init__(self, data, **kwargs):
        super(EstimationHDDMsharedVar, self).__init__(data, **kwargs)

    def init_model(self, data):
            self.model = hddm.HDDMTruncated(data, group_only_nodes = ['sz','st','sv'], **self.init_kwargs)


#HDDMGamma Estimation
class EstimationHDDMGamma(EstimationHDDMBase):

    def __init__(self, data, **kwargs):
        super(EstimationHDDMGamma, self).__init__(data, **kwargs)

    def init_model(self, data):
            self.model = my_hddm.HDDMGamma(data, group_only_nodes = ['sz','st','sv'], **self.init_kwargs)

#noniformative HDDM
class EstimationNoninformHDDM(EstimationHDDMBase):

    def __init__(self, data, **kwargs):
        super(EstimationNoninformHDDM, self).__init__(data, **kwargs)

    def init_model(self, data):
            self.model = my_hddm.HDDMGamma(data, group_only_nodes = ['sz','st','sv'], informative=False, **self.init_kwargs)


#HDDMRegressor Estimation
class EstimationHDDMRegressor(EstimationHDDMBase):

    def __init__(self, data, **kwargs):
        super(EstimationHDDMRegressor, self).__init__(data, **kwargs)

    def init_model(self, data):
            self.model = HDDMRegressor(data, group_only_nodes = ['sz','st','sv', 'v_slope'], **self.init_kwargs)

#single HDDMRegressors Estimation
class SingleRegressor(Estimation):

    def __init__(self, data, **kwargs):
        super(SingleRegressor, self).__init__(data, **kwargs)

        #create an HDDM model for each subject
        grouped_data = data.groupby('subj_idx')
        self.models = []
        for subj_idx, subj_data in grouped_data:
            model = HDDMRegressor(subj_data.to_records(), is_group_model=False, **kwargs)
            self.models.append(model)

    def estimate(self, **ddm_kwargs):
        samples = ddm_kwargs.pop('samples', 10000)
        ddm_kwargs.pop('map', False)
        [model.sample(samples, **ddm_kwargs) for model in self.models]

    def rename_index(self, node_name, subj_idx):
        if '(' in node_name:
            knode_name, cond = node_name.split('(')
            name = knode_name + '_subj' + '(' + cond + '.' + str(subj_idx)
        else:
            name = node_name + '_subj.' + str(subj_idx)
        return name

    def get_stats(self):

        for subj_idx, model in enumerate(self.models):
            subj_stats = model.gen_stats()
            subj_stats.rename(index=lambda x:self.rename_index(x, subj_idx), inplace=True)
            if subj_idx == 0:
                stats = subj_stats
            else:
                stats = stats.append(subj_stats)

        return stats

def compute_single_v1(v0, shift, name):
    """
    compute v1 stats from traces of v0 and shift
    """
    v1 = v0 +  shift

    quantiles = pm.utils.quantiles(v1, qlist=[2.5, 25, 50, 75, 97.5])
    s = {}
    s['mean']   = v1.mean()
    s['std']    = v1.std()
    s['2.5q']   = quantiles[2.5]
    s['25q']    = quantiles[25]
    s['50q']    = quantiles[50]
    s['75q']    = quantiles[75]
    s['97.5q']  = quantiles[97.5]

    return pd.DataFrame(pd.Series(s), columns=[name]).T

#HDDM with 2 conditions estimation
class EstimationHDDM2(EstimationHDDMBase):

    def __init__(self, data, **kwargs):
        super(self.__class__, self).__init__(data, **kwargs)

    def init_model(self, data):
            self.model = HDDMRegressor(data, group_only_nodes = ['sz','st','sv'], **self.init_kwargs)


    def compute_all_v1(self):
        """
        compute the stats of v_subj(c1)
        """
        v1_stats = pd.DataFrame()
        v0 = self.model.nodes_db.ix['v(c0)']['node'].trace()
        shift = self.model.nodes_db.ix['v_shift']['node'].trace()
        v1_stats = v1_stats.append(compute_single_v1(v0, shift, name='v(c1)'))

        for i_subj in range(self.model.num_subjs):
            v0 = self.model.nodes_db.ix['v(c0)_subj.%d' % i_subj]['node'].trace()
            shift = self.model.nodes_db.ix['v_shift_subj.%d' % i_subj]['node'].trace()
            v1_stats = v1_stats.append(compute_single_v1(v0, shift, name='v(c1)_subj.%d' % i_subj))

        return v1_stats



    def rename_v_nodes(self, stats):

        stats = stats.rename(index={'v(c0)_var':'v_var'})

        def rename_func(name):
            if name.startswith('v(c0)_subj'):
                prefix, subj_idx = name.split('.')
                return 'v_subj(c0).' + subj_idx
            elif name.startswith('v(c1)_subj'):
                prefix, subj_idx = name.split('.')
                return 'v_subj(c1).' + subj_idx
            else:
                return name

        return stats.rename(index=rename_func)


    def get_stats(self):

        stats = self.model.gen_stats()
        stats = stats.append(self.compute_all_v1())
        stats = self.rename_v_nodes(stats)

        return stats

#single HDDM2 Estimation
class EstimationHDDM2Single(SingleRegressor):

    def __init__(self, data, **kwargs):
        super(EstimationHDDM2Single, self).__init__(data, **kwargs)

    def compute_v1(self, model):

        v1 = v0 +  shift


        quantiles = pm.utils.quantiles(v1, qlist=[2.5, 25, 50, 75, 97.5])
        s = {}
        s['mean']   = v1.mean()
        s['std']    = v1.std()
        s['2.5q']   = quantiles[2.5]
        s['25q']    = quantiles[25]
        s['50q']    = quantiles[50]
        s['75q']    = quantiles[75]
        s['97.5q']  = quantiles[97.5]

        return pd.DataFrame(pd.Series(s), columns=['v1']).T

    def get_stats(self):

        for subj_idx, model in enumerate(self.models):
            subj_stats = model.gen_stats()
            v0 = model.nodes_db.ix['v(c0)']['node'].trace()
            shift = model.nodes_db.ix['v_shift']['node'].trace()
            subj_stats = subj_stats.append(compute_single_v1(v0, shift, 'v(c1)'))
            subj_stats.rename(index=lambda x:self.rename_index(x, subj_idx), inplace=True)
            if subj_idx == 0:
                stats = subj_stats
            else:
                stats = stats.append(subj_stats)

        return stats

#HDDM with outliers Estimation
class EstimationHDDMOutliers(EstimationHDDMsharedVar):

    def __init__(self, data, **kwargs):
        kwargs = copy.deepcopy(kwargs)
        kwargs['include'] += ['p_outlier']
        super(EstimationHDDMOutliers, self).__init__(data, **kwargs)


#Single MAP Estimation
class EstimationSingleMAP(Estimation):

    def __init__(self, data, **kwargs):
        super(EstimationSingleMAP, self).__init__(data, **kwargs)

        #create an HDDM model for each subject
        grouped_data = data.groupby('subj_idx')
        self.models = []
        for subj_idx, subj_data in grouped_data:
            model = my_hddm.HDDMGamma(subj_data.to_records(), is_group_model=False, **kwargs)
            self.models.append(model)

    def estimate(self, pool_size=1, **map_kwargs):
        single_map = lambda model: model.map(method='fmin_powell', **map_kwargs)

        if pool_size > 1:
            pool = Pool(processes=pool_size)
            pool.map(single_map, self.models)
        else:
            [single_map(model) for model in self.models]

    def get_stats(self):
        stats = {}
        for subj_idx, model in enumerate(self.models):
            values_tuple = [None] * len(model.get_stochastics())
            for (i_value, (node_name, node_row)) in enumerate(model.iter_stochastics()):
                if '(' in node_name:
                    knode_name, cond = node_name.split('(')
                    name = knode_name + '_subj' + '(' + cond + '.' + str(subj_idx)
                else:
                    name = node_name + '_subj.' + str(subj_idx)

                values_tuple[i_value] = (name, float(node_row['node'].value))
            stats.update(dict(values_tuple))
        return pd.Series(stats)

#Single MAP Estimate with p_outliers
class EstimationSingleMAPoutliers(EstimationSingleMAP):

    def __init__(self, data, **kwargs):
        kwargs = copy.deepcopy(kwargs)
        kwargs['include'] += ['p_outlier']
        super(EstimationSingleMAPoutliers, self).__init__(data, **kwargs)

#Single G^2 Estimation
class EstimationSingleOptimization(Estimation):

    def __init__(self, data, **kwargs):
        super(EstimationSingleOptimization, self).__init__(data, **kwargs)

        #create an HDDM model for each subject
        grouped_data = data.groupby('subj_idx')
        self.models = []
        for subj_idx, subj_data in grouped_data:
            model = my_hddm.HDDMGamma(subj_data.to_records(), is_group_model=False, **kwargs)
            self.models.append(model)

    def estimate(self, **quantiles_kwargs):
        self.results = [model.optimize(**quantiles_kwargs) for model in self.models]

    def get_stats(self):
        stats = {}
        for subj_idx, model in enumerate(self.models):
            values_tuple = [None] * len(model.get_stochastics())
            for (i_value, (node_name, node_row)) in enumerate(model.iter_stochastics()):
                if len(self.models) == 1:
                    name = node_name
                else:
                    if '(' in node_name:
                        knode_name, cond = node_name.split('(')
                        name = knode_name + '_subj' + '(' + cond + '.' + str(subj_idx)
                    else:
                        name = node_name + '_subj.' + str(subj_idx)

                values_tuple[i_value] = (name, float(node_row['node'].value))
            stats.update(dict(values_tuple))
        return pd.Series(stats)

#single  Regression Estimation
class SingleRegOptimization(EstimationSingleOptimization):

    def __init__(self, data, **kwargs):
        Estimation.__init__(self, data, **kwargs)

        #create an HDDM model for each subject
        grouped_data = data.groupby('subj_idx')
        self.models = []
        for subj_idx, subj_data in grouped_data:
            model = HDDMRegressor(subj_data.to_records(), is_group_model=False, **kwargs)
            self.models.append(model)


#HDDM Estimation
class EstimationGroupOptimization(Estimation):

    def __init__(self, data, **kwargs):
        super(self.__class__, self).__init__(data, **kwargs)
        self.model = my_hddm.HDDMGamma(data, **kwargs)

    def estimate(self, **kwargs):
        self.results = self.model.optimize(**kwargs)

    def get_stats(self):
        return pd.Series(self.results)


###################################
#
#

def put_all_params_in_a_single_dict(joined_params, group_params, subj_noise, depends_on):
    p_dict = joined_params.copy()

    #if there is only one subject then there is nothing to do
    if len(group_params.values()[0]) == 1:
        return p_dict

    #put subj params in p_dict
    for cond, cond_params in group_params.iteritems():
        for idx, subj_dict in enumerate(cond_params):
            for (name, value) in subj_dict.iteritems():
                if name in depends_on:
                    p_dict['%s_subj(%s).%i'%(name, cond, idx)] = value
                else:
                    p_dict['%s_subj.%i'%(name, idx)] = value

    #put group noise in the p_dict
    for (name, value) in subj_noise.iteritems():
        p_dict[name + '_var'] = value

    return p_dict

def make_hash(o):
    """
    Makes a hash from a dictionary, list, tuple or set to any level, that contains
    only other hashable types (including any lists, tuples, sets, and
    dictionaries).
    """
    try:
        return hashlib.md5(cPickle.dumps(o)).hexdigest()
    except TypeError:
        oo = copy.deepcopy(o)
        oo['init']['regressor']['func'] = 123
        return hashlib.md5(cPickle.dumps(oo)).hexdigest()

def single_recovery_fixed_n_trials(estimation, kw_dict, raise_errors=True, action='run',
                                   single_runs_folder='.', run_type=None):
    """run analysis for a single Estimation.
    Input:
        seed <int> - a seed to generate params and data
        estimation <class> - the class of the Estimation
        kw_dict - a dictionary that holds 4 keywords arguments dictionaries, each
            for a different fucntions:
            params - for hddm.generate.gen_rand_params
            data - for hddm.generate.gen_rand_data
            init - for the constructor of the estimation
            estimate - for Estimation().estimate
    """


    #generate params and data for regression experiments
    n_conds = kw_dict['n_conds']
    if run_type == 'regress':
        np.random.seed(kw_dict['seed_params'])
        _ = kw_dict['params'].pop('n_conds', None)
        params = genreg.gen_reg_params(**kw_dict['params'])
        joined_params = params
    #generate params and data for other experiments
    else:
        np.random.seed(kw_dict['seed_params'])
        cond_v =  (np.random.rand()*0.4 + 0.1) * 2**np.arange(n_conds)
        params, joined_params = hddm.generate.gen_rand_params(cond_dict={'v':cond_v}, **kw_dict['params'])


    np.random.seed(kw_dict['seed_data'])

    #create a job hash
    kw_dict['estimator_class'] = estimation.__name__
    h = make_hash(kw_dict)

    # check if job was already run, if so, load it!
    fname = os.path.join(single_runs_folder, '%s.dat' % str(h))
    if os.path.isfile(fname) and (action != 'rerun'):
        if action == 'collect':
            stats = pd.load(fname)
            print "Loading job %s" % h
            run_estimation=False
            if len(stats) == 0:
                return stats
        elif action == 'run':
            stats = pd.load(fname)
            print "Skiping job %s" % h
            return stats
        elif action == 'delete':
            os.remove(fname)
            return pd.DataFrame()
        else:
            raise ValueError('Unknown action')

    else:
        #create a file that holds the results and to make sure that no other worker would start
        #working on this job
        pd.DataFrame().save(fname)

        #create a temporary file with a unique name
        temp_fname = fname + '.' + str(os.getpid())
        pd.DataFrame().save(temp_fname)

        #get list of files
        #if the length of the list is larger than one, then more than one worker is trying to perform the same job
        #in this case we leave only the job with the "largest file name"

        files = glob.glob(fname + '.*')
        #if we need to kill the job
        if (len(files) > 1) and (max(files) != temp_fname):
            os.remove(temp_fname)
            stats = pd.load(fname)
            print "Loading job %s" % h
            run_estimation=False
            if len(stats) == 0:
                return stats

        #else we will continue as usuall
        else:
            print "Working on job %s (%s)" % (h, estimation)
            pprint.pprint(kw_dict)
            run_estimation=True

    #generate params and data
    if run_type == 'regress':
        params['reg_outcomes'] = 'v'
        data, group_params = genreg.gen_regression_data(params, **kw_dict['data'])
        group_params = {'c1': group_params}
        subj_noise = kw_dict['data']['subj_noise']
    else:
        data, group_params = hddm.generate.gen_rand_data(params, **kw_dict['data'])
        if kw_dict['data']['subjs'] == 1 and n_conds == 1:
            group_params = {'c0': [group_params]}
        elif n_conds == 1:
            group_params = {'c0': group_params}
        elif kw_dict['data']['subjs'] == 1:
            for key, value in group_params.iteritems():
                group_params[key] = [value]
        subj_noise = kw_dict['data']['subj_noise']

        # prepare data for HDDMShift
        if estimation.__name__ in ESTIMATTIONS_WITH_REGRESSORS:
            cond = np.zeros(len(data['condition']))
            cond[data.condition == 'c1'] = 1
            data['condition'] = cond

    if n_conds > 1:
        depends_on = {'v': 'condition'}
    else:
        depends_on = {}

    group_params = put_all_params_in_a_single_dict(joined_params, group_params, subj_noise, depends_on=depends_on)

    #estimate
    if run_estimation:
        try:
            print "Estimation began on %s" % time.ctime()
            data = DataFrame(data)
            est = estimation(data, **kw_dict['init'])
            est.estimate(**kw_dict['estimate'])
            stats = est.get_stats()
            stats.save(fname)
            os.remove(temp_fname)
            print "Estimation ended on %s" % time.ctime()


            if hasattr(est, 'geweke_problem') and est.geweke_problem:
                print "Warning!!! Geweke problem was found"
                with open('geweke_problems','a') as g_file:
                    g_file.write('******* %s\n ' % time.ctime())
                    g_file.write('%s\n' % pd.Series(kw_dict))
                    g_file.write('fname: %s\n' % fname)

        #raise or log errors
        except Exception as err:
            tb = traceback.format_exc()
            print tb
            if raise_errors:
                raise err
            else:
                with open('err.log','a') as f:
                    f.write('******* %s\n ' % time.ctime())
                    f.write('%s\n' % pd.Series(kw_dict))
                    f.write('%s: %s\n' % (type(err), err))
                return pd.DataFrame()

    group_params = pd.Series(group_params)
    if run_type in ['priors', 'trials', 'regress']:
        group_params = group_params.select(lambda x:'reg_outcomes' not in x)

    output = combine_params_and_stats(group_params, stats)
    return output

def combine_params_and_stats(params, stats):
    if isinstance(stats, pd.DataFrame):
        params = pd.DataFrame(params, columns=['truth'])
        stats = stats.rename(columns={'mean': 'estimate'})
        comb = pd.concat([params, stats], axis=1)
    else:
        comb = pd.concat([params, stats], axis=1, keys=['truth', 'estimate'])
    comb['Err'] = np.abs(np.asarray((comb['truth'] - comb['estimate']), dtype=np.float32))

    return comb

def multi_recovery_fixed_n_trials(estimation, equal_seeds, seed_params, single_runs_folder,
                                  seed_data, n_params, n_datasets, kw_dict, path=None, view=None,
                                  action='run', run_type=None):

    #create seeds for params and data
    p_seeds = seed_params + np.arange(n_params)
    d_seeds = seed_data + np.arange(n_datasets)

    p_results = {}
    for p_seed in p_seeds:
        d_results = {}
        if equal_seeds:
            d_seeds = [p_seed]
        for d_seed in d_seeds:
            kw_seed = copy.deepcopy(kw_dict)
            kw_seed['seed_params'] = p_seed
            kw_seed['seed_data'] = d_seed
            if view is None:
                d_results[d_seed] = single_recovery_fixed_n_trials(estimation, kw_seed, raise_errors=True,
                                                                   action=action, single_runs_folder=single_runs_folder,
                                                                   run_type=run_type)
            else:
                # append to job queue
                d_results[d_seed] = view.apply_async(single_recovery_fixed_n_trials, estimation,
                                                     kw_seed, False, action, single_runs_folder=single_runs_folder,
                                                     run_type=run_type)

        p_results[p_seed] = d_results

    return p_results


def example_singleMAP():

    #include params
    params = {'include': ('v','t','a')}

    #kwards for gen_rand_data
    subj_noise = {'v':0.1, 'a':0.1, 't':0.05}
    data = {'subjs': 5, 'subj_noise': subj_noise}

    #kwargs for initialize estimation
    init = {}

    #kwargs for estimation
    estimate = {'runs': 3}

    #creat kw_dict
    kw_dict = {'params': params, 'data': data, 'init': init, 'estimate': estimate}

    #run analysis
    all_params, all_stats = multi_recovery_fixed_n_trials(EstimationSingleMAP, seed_data=1, seed_params=1,
                                                           n_runs=3, mpi=False, kw_dict=kw_dict, path='delete_me')

    return all_params, all_stats

def example_singleMLE():

    #include params
    include = ('v','t','a')
    params = {'include': include}

    #kwards for gen_rand_data
    subj_noise = {'v':0.1, 'a':0.1, 't':0.05}
    data = {'subjs': 5, 'subj_noise': subj_noise}

    #kwargs for initialize Estimation
    init = {}

    #kwargs for estimation
    estimate = {'include': include}

    #creat kw_dict
    kw_dict = {'params': params, 'data': data, 'init': init, 'estimate': estimate}

    #run analysis
    results = multi_recovery_fixed_n_trials(EstimationSingleMLE, seed=1, n_runs=4,
                                            kw_dict=kw_dict, path='delete_me')

    return results

def example_singleOptimization():

    #include params
    include = ('v','t','a')
    params = {'include': include}

    #kwards for gen_rand_data
    subj_noise = {'v':0.1, 'a':0.1, 't':0.05}
    data = {'subjs': 4, 'subj_noise': subj_noise, 'size': 200}

    #kwargs for initialize Estimation
    init = {}

    #kwargs for estimation
    estimate = {'method': 'gsquare', 'quantiles': (0.1, 0.3, 0.5, 0.7, 0.9)}

    #creat kw_dict
    kw_dict = {'params': params, 'data': data, 'init': init, 'estimate': estimate}

    #run analysis
    results = multi_recovery_fixed_n_trials(EstimationSingleOptimization, seed_data=1, seed_params=1, n_params=2,
                                            n_datasets=1, kw_dict=kw_dict, path='delete_me')

    return results

def example_GroupOptimization():

    #include params
    include = ('v','t','a')
    params = {'include': include}

    #kwards for gen_rand_data
    subj_noise = {'v':0.1, 'a':0.1, 't':0.05}
    data = {'subjs': 4, 'subj_noise': subj_noise, 'size': 200}

    #kwargs for initialize Estimation
    init = {}

    #kwargs for estimation
    estimate = {'method': 'gsquare', 'quantiles': (0.1, 0.3, 0.5, 0.7, 0.9)}

    #creat kw_dict
    kw_dict = {'params': params, 'data': data, 'init': init, 'estimate': estimate}

    #run analysis
    results = multi_recovery_fixed_n_trials(EstimationGroupOptimization, seed_data=1, seed_params=1, n_params=2,
                                            n_datasets=1, kw_dict=kw_dict, path='delete_me')

    return results



def fix_wrong_subjects_name(data):
    to_remove = []
    for t_idx in data.index:
        if t_idx[-1].startswith('v(c') and '_subj' in t_idx[-1]:
            wrong_name = t_idx[-1]
            cond = wrong_name[3]
            subj_idx = wrong_name.split('.')[1]

            #create the correct index
            t_idx2 = list(t_idx)
            t_idx2[-1] = 'v_subj(c%s).%s' % (cond, subj_idx)

            estimate_value = data.get_value(t_idx, col='estimate')
            data.set_value(tuple(t_idx2), col='estimate', value=estimate_value)
            to_remove.append(t_idx)

    #remove wrong indecies
    data = data.drop(to_remove)

    #get MSE, Err and stuff
    data['Err'] = np.abs(np.asarray((data['truth'] - data['estimate']), dtype=np.float32))

    return data

def use_group_truth_value_for_subjects_in_HDDMsharedVar(data):
    """
    assign the group truth value for subjects nodes that do not have one in HDDMsharedVar
    """

    for t_idx in data.index:
        if t_idx[3] == 'HDDMsharedVar' and t_idx[-1].startswith('s') and '_subj' in t_idx[-1]:
            group_idx = list(t_idx)
            group_idx[-1] = group_idx[-1][:2]
            group_idx = tuple(group_idx)

            estimate_value = data.get_value(group_idx, col='estimate')
            data.set_value(t_idx, col='estimate', value=estimate_value)

    data['Err'] = np.abs(np.asarray((data['truth'] - data['estimate']), dtype=np.float32))

    return data

def get_knode_group_node_name(full_name):

    #if group node
    if 'subj' not in full_name:
        return full_name

    else:
        name, rest = full_name.split('_subj')
        if '(' in rest:
            name += rest.split('.')[0]
        return name


def add_group_stat_to_SingleOptimation(data, stat, estimators=('HDDM2Single', 'Quantiles_subj', 'ML')):

    data = data.copy()
    for method in estimators:
        sdata = data.select(lambda x:(x[3] == method) and ('subj' in x[-1]))
        groups = sdata.groupby(lambda x:tuple(list(x[:6]) + [get_knode_group_node_name(x[-1])]))
        for (t_idx, t_data) in groups:
            group_estimate = stat(t_data['estimate'])
            data.set_value(t_idx, col='estimate', value=group_estimate)

    data['Err'] = np.abs(np.asarray((data['truth'] - data['estimate']), dtype=np.float32))

    return data

def add_group_stat_to_SingleRegressor(data):

    data = data.copy()
    sdata = data.select(lambda x:(x[3] == 'SingleRegressor') and ('subj' in x[-1]))
    groups = sdata.groupby(lambda x:tuple(list(x[:6])))
    means = np.zeros(len(groups))
    stds = np.zeros(len(groups))
    for (t_idx, t_data) in groups:
        slopes = t_data.select(lambda x:x[-1].startswith('v_slope_subj'))[['estimate', 'std']]
        pooled_var = 1. / sum(1. / (slopes['std']**2))
        pooled_mean = sum(slopes['estimate'] / (slopes['std']**2)) * pooled_var
        mass_under = scipy.stats.norm.ppf(0.025, pooled_mean, np.sqrt(pooled_var))

        slope_idx = tuple(list(t_idx) + ['v_slope'])
        data.set_value(slope_idx, col='estimate', value=pooled_mean)
        data.set_value(slope_idx, col='std', value=np.sqrt(pooled_var))
        data.set_value(slope_idx, col='2.5q', value=mass_under)

        true_value = data.get_value(slope_idx, col='truth')
        data.set_value(slope_idx, col='Err', value=abs(true_value - pooled_mean))


    return data

def geweke_test_problem(model):

    for name, node_desc in model.iter_stochastics():
        node = node_desc['node']
        output = pm.geweke(node)
        values = np.array(output)[:,1]
        if np.any(np.abs(values) > 2):
            print
            print "Geweke problem was found in: %s" % name
            return True
    return False

def add_var_to_SingleOptimation(data, subj_noise, estimators=('Quantiles_subj', 'ML', 'HDDM2Single')):
    data = data.copy()

    params_var = {}
    params = pd.DataFrame(columns=['a','v','t'], index=['std_name', 'subj_name', 'truth'])
    params['a'] = ['a_var', 'a_subj', subj_noise['a']]
    params['t'] = ['t_var', 't_subj', subj_noise['t']]
    params['v'] = ['v_var', 'v_subj(c0)', subj_noise['v']]

    for method in estimators:
        for param, tt in params.iteritems():
            sdata = data.select(lambda x:x[3] == method and  x[-1].startswith(tt['subj_name'])).estimate
            err_std = lambda s,true_value=tt['truth']: np.std(s - true_value, ddof=1)
            groups = sdata.groupby(lambda x:x[:6]).agg(err_std)
            groups.rename(lambda x:tuple(list(x) + [tt['std_name']]), inplace=True)
            data['Err'].ix[groups.index] = groups

    return data