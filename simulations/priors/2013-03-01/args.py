def args():
    """
    priors experiment
    """
    run_type='priors'
    estimators=['ML']
    n_subjs=1
    n_trials=[20,30,40,50,75,100,150,250]
    n_params=120
    n_datasets=1
    equal_seeds=True
    include=['st', 'sv', 'sz', 'z']
    depends_on = {'v':'condition'}
    factor3_vals=[2,3]

    return locals()
