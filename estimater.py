import hddm
import time
import kabuki
import numpy as np
from scipy.optimize import fmin_powell
from hddm.generate import gen_rand_params, gen_rand_data
from multiprocessing import Pool


class Estimater(object):

    def __init__(self, data, include=(), pool_size=1, **kwargs):

        #save args
        self.data = data
        self.include = include
        self.stats = {}
        self.init_model(data, **kwargs)
    
    def split_data_to_subjs(self, data):
        subj_data = data.groupby('subj_idx').groups
        for s in subj_data:
            del s.subj_idx
        return subj_data



#HDDM Estimater
class EstimaterHDDM(Estimater):

    def __init__(self, data, **kwargs):
        super(self.__class__, self).__init__(data, **kwargs)
        self.model = hddm.HDDM(data, **kwargs)

    def estimate(self, iter, burn=0, thin=1):
        self.model.sample(iter, burn, thin)

    def get_stats(self):
        return self.model.stats()


#Group MLE Estimater
class EstimaterGroupMLE(Estimater):

    def __init__(self, data, **kwargs):
        super(self.__class__, self).__init__(data, **kwargs)
        subj_data = self.split_data_to_subj(data)
#        self.models = []
#        for i_subj in range(len(subj_data)):
#            model = hddm.HDDM(data, *args, **kwargs)
#            self.models.append(model)

    def estimate(self):
        pass

    def get_stats(self):
        pass

#Single MAP Estimater
class EstimaterSingleMAP(Estimater):

    def __init__(self, data, **kwargs):
        super(self.__class__, self).__init__(data, **kwargs)
        subj_data = self.split_data_to_subj(data)
        self.models = []
        for i_subj in range(len(subj_data)):
            model = hddm.HDDM(data, **kwargs)
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
