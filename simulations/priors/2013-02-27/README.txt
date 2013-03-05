run priors experiment
commit: 6e535c38ad7fa729a45c9dfbc47a4a0a7aef509c
estimators=['HDDMGamma', 'ML', 'Quantiles_subj']
                exp = run_experiments(n_subjs=1, n_trials=[20,30,40,50,75,100,150,250], n_params=120, n_datasets=1, equal_seeds=True,include=include, view=view, depends_on = {'v':'condition'}, estimators=estimators, factor3_vals=[2,3], run_type=run_type, action=action)

*sz in ML is smaller than zero
