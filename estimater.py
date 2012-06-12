import hddm
import time
import kabuki
import numpy as np
from scipy.optimize import fmin_powell
from hddm.generate import gen_rand_params, gen_rand_data
from multiprocessing import Pool
from pandas import DataFrame


# For each method that you would like to check you need do create a ass that inherits from
# the Estimater class and implement the estimate and get_stats attributes.


class Estimater(object):
    def __init__(self, data, include=(), pool_size=1, **kwargs):

        #save args
        self.data = data
        self.include = include
        self.stats = {}


#HDDM Estimater
class EstimaterHDDM(Estimater):

    def __init__(self, data, **kwargs):
        super(self.__class__, self).__init__(data, **kwargs)
        self.model = hddm.HDDM(data, **kwargs)

    def estimate(self, **kwards):
        self.model.sample(**kwards)

    def get_stats(self):
        return self.model.stats()


#Single MLE Estimater
class EstimaterSingleMLE(Estimater):

    def __init__(self, data, **kwargs):
        super(self.__class__, self).__init__(data, **kwargs)

        #create an HDDM model for each subject
        data = hddm.utils.flip_errors(data)
        self.grouped_data = data.groupby('subj_idx')
        self.models = []
        self.stats ={}


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

#Single MAP Estimater
class EstimaterSingleMAP(Estimater):

    def __init__(self, data, **kwargs):
        super(self.__class__, self).__init__(data, **kwargs)

        #create an HDDM model for each subject
        grouped_data = data.groupby('subj_idx')
        self.models = []
        for subj_idx, subj_data in grouped_data:
            model = hddm.HDDM(subj_data.to_records(), is_group_model=False, **kwargs)
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
            values_tuple = [(x.__name__ + str(idx), x.value) for x in model.mc.stochastics]
            stats.update(dict(values_tuple))
        return stats



###################################
#
#

def single_analysis(seed, estimater, kw_dict):
    """run analysis for a single Estimater.
    Input:
        seed <int> - a seed to generate params and data
        estimater <class> - the class of the Estimater
        kw_dict - a dictionary that holds 4 keywords arguments dictionaries, each
            for a different fucntions:
            params - for hddm.generate.gen_rand_params
            data - for hddm.generate.gen_rand_data
            init - for the constructor of the estimater
            estimate - for Estimater().estimate
    """
    np.random.seed(seed)
    params = hddm.generate.gen_rand_params(**kw_dict['params'])
    data, group_params = hddm.generate.gen_rand_data(params, **kw_dict['data'])
    data = DataFrame(data)

    est = estimater(data, **kw_dict['init'])
    est.estimate(**kw_dict['estimate'])
    return group_params, est.get_stats()


def multi_analysis(estimater, seed, n_runs, mpi, kw_dict):

    analysis_func = lambda seed: single_analysis(seed, estimater, kw_dict)
    seeds = range(seed, seed + n_runs)

    if mpi:
        import mpi4py_map
        results = mpi4py_map.map(analysis_func, seeds)
    else:
        results = [analysis_func(x) for x in seeds]

    return results


def example_singleMAP():

    #include params
    params = {'include': ('v','t','a')}

    #kwards for gen_rand_data
    subj_noise = {'v':0.1, 'a':0.1, 't':0.05}
    data = {'subjs': 5, 'subj_noise': subj_noise}

    #kwargs for initialize estimater
    init = {}

    #kwargs for estimation
    estimate = {'runs': 3}

    #creat kw_dict
    kw_dict = {'params': params, 'data': data, 'init': init, 'estimate': estimate}

    #run analysis
    results = multi_analysis(EstimaterSingleMAP, seed=1, n_runs=10, mpi=False, kw_dict=kw_dict)

    return results

def example_singleMLE():

    #include params
    include = ('v','t','a')
    params = {'include': include}

    #kwards for gen_rand_data
    subj_noise = {'v':0.1, 'a':0.1, 't':0.05}
    data = {'subjs': 5, 'subj_noise': subj_noise}

    #kwargs for initialize estimater
    init = {}

    #kwargs for estimation
    estimate = {'include': include}

    #creat kw_dict
    kw_dict = {'params': params, 'data': data, 'init': init, 'estimate': estimate}

    #run analysis
    results = multi_analysis(EstimaterSingleMLE, seed=1, n_runs=4, mpi=False, kw_dict=kw_dict)

    return results


if __name__=='__main__':
    example()
