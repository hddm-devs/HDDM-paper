import hddm
import numpy as np
import pandas as pd
import copy
import os.path

from scipy.optimize import fmin_powell
from multiprocessing import Pool
from pandas import DataFrame



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



#Single MLE Estimation
class EstimationSingleMLE(Estimation):

    def __init__(self, data, **kwargs):
        super(self.__class__, self).__init__(data, **kwargs)

        #create an HDDM model for each subject
        data = hddm.utils.flip_errors(data)
        self.grouped_data = data.groupby('subj_idx')
        self.models = []
        self.stats = {}


    def estimate(self, include):
        wiener_kw = {'err': 1e-4, 'nT':2, 'nZ':2,
                         'use_adaptive':1, 'simps_err':1e-3}
        wiener_kw.update(hddm.generate.gen_rand_params(include))

        #define objective function
        def mle_objective_func(values, include, wiener_kw):
            wiener_kw.update(zip(include, values))
            return -hddm.wfpt.wiener_like(**wiener_kw)

        #estimate for each subject
        for subj_idx, subj_data in self.grouped_data:
            #estimate
            wiener_kw['x'] = subj_data['rt'].values
            values0 = np.array([wiener_kw[param] for param in include])
            xopt = fmin_powell(mle_objective_func, values0, args=(include, wiener_kw))
            #analyze
            for i_param, param in enumerate(include):
                self.stats[param + `subj_idx`] = xopt[i_param]

    def get_stats(self):
        return self.stats

#Single MAP Estimation
class EstimationSingleMAP(Estimation):

    def __init__(self, data, **kwargs):
        super(EstimationSingleMAP, self).__init__(data, **kwargs)

        #create an HDDM model for each subject
        grouped_data = data.groupby('subj_idx')
        self.models = []
        for subj_idx, subj_data in grouped_data:
            model = hddm.HDDMTruncated(subj_data.to_records(), is_group_model=False, **kwargs)
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
        super(self.__class__, self).__init__(data, **kwargs)

        #create an HDDM model for each subject
        grouped_data = data.groupby('subj_idx')
        self.models = []
        for subj_idx, subj_data in grouped_data:
            model = hddm.HDDMTruncated(subj_data.to_records(), is_group_model=False, **kwargs)
            self.models.append(model)

    def estimate(self, **quantiles_kwargs):
        self.results = [model.optimize(**quantiles_kwargs) for model in self.models]

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


#HDDM Estimation
class EstimationGroupOptimization(Estimation):

    def __init__(self, data, **kwargs):
        super(self.__class__, self).__init__(data, **kwargs)
        self.model = hddm.HDDMTruncated(data, **kwargs)

    def estimate(self, **kwargs):
        self.results = self.model.optimize(**kwargs)

    def get_stats(self):
        return pd.Series(self.results)


###################################
#
#

def put_all_params_in_a_single_dict(joined_params, group_params, subj_noise, depends_on):
    p_dict = joined_params.copy()

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

    if isinstance(o, set) or isinstance(o, tuple) or isinstance(o, list):
        return tuple([make_hash(e) for e in o])

    elif not isinstance(o, dict):
        return hash(o)

    new_o = copy.deepcopy(o)
    for k, v in new_o.items():
        new_o[k] = make_hash(v)

    return hash(tuple(frozenset(new_o.items())))

def single_recovery_fixed_n_trials(estimation, kw_dict):
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

    #generate params and data
    np.random.seed(kw_dict['seed_params'])
    n_conds = kw_dict['params'].pop('n_conds', 4)
    cond_v =  (np.random.rand()*0.4 + 0.1) * 2**np.arange(n_conds)
    params, joined_params = hddm.generate.gen_rand_params(cond_dict={'v':cond_v}, **kw_dict['params'])
    np.random.seed(kw_dict['seed_data'])

    #create a job hash
    kw_dict['estimator_class'] = estimation.__name__
    h = make_hash(kw_dict)

    # check if job was already run, if so, load it!
    fname = os.path.join('temp', '%s.dat' % str(h))
    if os.path.isfile(fname):
        stats = pd.load(fname)
        print "Loading job %s" % h
        generate_data=False
        if len(stats) == 0:
            return stats
    else:
        pd.DataFrame().save(fname)
        generate_data=True

    #generate params and data
    data, group_params = hddm.generate.gen_rand_data(params, generate_data=generate_data, **kw_dict['data'])
    group_params = put_all_params_in_a_single_dict(joined_params, group_params, kw_dict['data']['subj_noise'], kw_dict['init']['depends_on'])

    #estimate
    if generate_data:
        data = DataFrame(data)
        est = estimation(data, **kw_dict['init'])
        est.estimate(**kw_dict['estimate'])
        stats = est.get_stats()
        stats.save(fname)

    return combine_params_and_stats(pd.Series(group_params), stats)

def combine_params_and_stats(params, stats):
    if isinstance(stats, pd.DataFrame):
        params = pd.DataFrame(params, columns=['truth'])
        stats = stats.rename(columns={'mean': 'estimate'})
        comb = pd.concat([params, stats], axis=1)
    else:
        comb = pd.concat([params, stats], axis=1, keys=['truth', 'estimate'])
    comb['MSE'] = np.asarray((comb['truth'] - comb['estimate'])**2, dtype=np.float32)
    comb['Err'] = np.abs(np.asarray((comb['truth'] - comb['estimate']), dtype=np.float32))
    comb['relErr'] = np.abs(np.asarray((comb['Err'] / comb['truth']), dtype=np.float32))

    return comb

def multi_recovery_fixed_n_trials(estimation, seed_params,
                                  seed_data, n_params, n_datasets, kw_dict, path=None, view=None):

    single = single_recovery_fixed_n_trials
    analysis_func = lambda kw_seed: single(estimation, kw_seed)

    #create seeds for params and data
    p_seeds = seed_params + np.arange(n_params)
    d_seeds = seed_data + np.arange(n_datasets)

    p_results = {}
    for p_seed in p_seeds:
        d_results = {}
        for d_seed in d_seeds:
            kw_seed = copy.deepcopy(kw_dict)
            kw_seed['seed_params'] = p_seed
            kw_seed['seed_data'] = d_seed
            if view is None:
                d_results[d_seed] = analysis_func(kw_seed)
            else:
                # append to job queue
                d_results[d_seed] = view.apply_async(single_recovery_fixed_n_trials, estimation, kw_seed)

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
    data['MSE'] = np.asarray((data['truth'] - data['estimate'])**2, dtype=np.float32)
    data['Err'] = np.abs(np.asarray((data['truth'] - data['estimate']), dtype=np.float32))
    data['relErr'] = np.abs(np.asarray((data['Err'] / data['truth']), dtype=np.float32))

    return data

