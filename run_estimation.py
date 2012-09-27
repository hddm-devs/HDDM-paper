import estimate as est
import pandas as pd
from copy import deepcopy
import time
import matplotlib.pyplot as plt

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

def compare_models(n_subjs=8, n_samples=(10, 40, 100)):
    #include params
    params = {'include': ('v','t','a')}
    recover = est.multi_recovery_fixed_n_samples

    models_samples = []
    for cur_samples in n_samples:
        #kwards for gen_rand_data
        subj_noise = {'v':0.1, 'a':0.1, 't':0.05}
        data = {'subjs': n_subjs, 'subj_noise': subj_noise, 'size': cur_samples}

        #kwargs for initialize estimation
        init = {}

        #kwargs for estimation
        estimate = {'runs': 3}

        #creat kw_dict
        kw_dict = {'params': params, 'data': data, 'init': init, 'estimate': estimate}

        models_params = {est.EstimationSingleMAP: {'runs': 3},
                         est.EstimationHDDM: {'samples': 5000, 'burn': 3000}}

        models_ests = []
        for model, est_dict in models_params.iteritems():
            kw_dict_model = deepcopy(kw_dict)
            kw_dict_model['estimate'] = est_dict
            #run analysis
            models_ests.append(recover(model, seed_data=1, seed_params=1, n_runs=3, mpi=False,
                                       kw_dict=kw_dict_model))


        models_samples.append(pd.concat(models_ests, keys=['SingleMAP', 'HDDM'], names=['estimation']))

    results = pd.concat(models_samples, keys=n_samples, names=['n_samples'])
    results.index.names[-1] = 'ind_param'

    results = est.select_param(results, ['v', 'a', 't'])
    return results

def plot_n_samples(res):
    grouped = res.MSE.dropna().groupby(level=('n_samples', 'estimation', 'params')).mean()

    fig = plt.figure()
    for i, (param_name, param_data) in enumerate(grouped.groupby(level=('params',))):
        ax = fig.add_subplot(3, 1, i+1)
        ax.set_title(param_name)
        for est_name, est_data in param_data.groupby(level=['estimation']):
            ax.plot(est_data.index.get_level_values('n_samples'), est_data, label=est_name)

        ax.set_xlabel('n_samples')
        ax.set_ylabel('MSE')

    plt.legend()





if __name__ == "__main__":
    singleMAP_fixed_n_samples()