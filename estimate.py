import hddm
import numpy as np
import pandas as pd
import cPickle

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
class EstimationHDDM(Estimation):

    def __init__(self, data, **kwargs):
        super(EstimationHDDM, self).__init__(data, **kwargs)
        self.model = hddm.HDDM(data, **kwargs)

    def estimate(self, **kwargs):
        samples = kwargs.pop('samples', 10000)
        self.model.approximate_map()
        self.model.sample(samples, **kwargs)

    def get_stats(self):
        stats = self.model.print_stats()

        return stats['mean']


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
        super(self.__class__, self).__init__(data, **kwargs)

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
        for idx, model in enumerate(self.models):
            model.mcmc()
            values_tuple = [(x.__name__ + '_subj.' + str(idx), x.value) for x in model.mc.stochastics]
            stats.update(dict(values_tuple))
        return pd.Series(stats)



###################################
#
#

def put_all_params_in_a_single_dict(params, group_params, subj_noise):
    p_dict = params.copy()

    #put subj params in p_dict
    for idx, subj_dict in enumerate(group_params):
        for (name, value) in subj_dict.iteritems():
            p_dict['%s_subj.%i'%(name, idx)] = value

    #put group noise in the p_dict
    for (name, value) in subj_noise.iteritems():
        p_dict[name + '_std'] = value

    return p_dict

def single_recovery_fixed_n_trials(seed_params, seed_data, estimation, kw_dict):
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
    np.random.seed(seed_params)
    params = hddm.generate.gen_rand_params(**kw_dict['params'])
    np.random.seed(seed_data)
    data, group_params = hddm.generate.gen_rand_data(params, **kw_dict['data'])
    group_params = put_all_params_in_a_single_dict(params, group_params, kw_dict['data']['subj_noise'])
    data = DataFrame(data)

    #estimate
    est = estimation(data, **kw_dict['init'])
    est.estimate(**kw_dict['estimate'])
    return combine_params_and_stats(pd.Series(group_params), est.get_stats())

def combine_params_and_stats(params, stats):
    comb = pd.concat([params, stats], axis=1, keys=['truth', 'estimate'])
    comb['MSE'] = np.asarray((comb['truth'] - comb['estimate'])**2, dtype=np.float32)

    return comb

def multi_recovery_fixed_n_trials(estimation, seed_params,
                                  seed_data, n_runs, kw_dict, path=None, view=None):

    single = single_recovery_fixed_n_trials
    analysis_func = lambda seeds: single(view, seeds[0], seeds[1], estimation, kw_dict)

    #create seeds for params and data
    p_seeds = seed_params + np.arange(n_runs)
    d_seeds = seed_data + np.arange(n_runs)

    p_results = {}
    for p_seed in p_seeds:
        d_results = {}
        for d_seed in d_seeds:
            if view is None:
                d_results[d_results] = analysis_func((p_seed, d_seed))
            else:
                # append to job queue
                d_results[d_seed] = view.apply_async(single_recovery_fixed_n_trials, p_seed, d_seed, estimation, kw_dict)

        p_results[p_seed] = d_results

    # if path is not None:
    #     with open(path,'w') as file:
    #         cPickle.dump([all_params, all_stats], file, cPickle.HIGHEST_PROTOCOL)

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

def select_param(stats, param_names, also_contains='subj'):
    if isinstance(param_names, str):
        param_names = [param_names]

    extracted = {}
    index = stats.index
    for name in param_names:
        select = [ix for ix in index if ix[-1].startswith(name) and also_contains in ix[-1]]
        extracted[name] = stats.ix[select]

    return pd.concat(extracted, names=['params'])


