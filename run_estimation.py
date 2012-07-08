import estimate as est
import time

def singleMAP_fixed_n_samples(n_subjs=8, n_samples=200):

    #include params
    params = {'include': ('v','t','a')}

    #kwards for gen_rand_data
    subj_noise = {'v':0.1, 'a':0.1, 't':0.05}
    data = {'subjs': n_subjs, 'subj_noise': subj_noise, 'samples': n_samples}

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
    recover(est.EstimationSingleMAP, seed_data=1, seed_params=1,n_runs=3, mpi=False, 
            kw_dict=kw_dict, path=filename)


if __name__ == "__main__":
    singleMAP_fixed_n_samples()