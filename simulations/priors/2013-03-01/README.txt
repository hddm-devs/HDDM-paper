run priors experiment
commit: d210e0f0ec8be4b5805e12475ba15cc6708d2e59
replace the ML from 2013-02-29 experiment that did not come out well (sz was smaller than 0).
I just recompile the source code

estimators=['ML']
exp = run_experiments(n_subjs=1, n_trials=[20,30,40,50,75,100,150,250], n_params=120, n_datasets=1, equal_seeds=True,
include=include, view=view, depends_on = {'v':'condition'}, estimators=estimators,
factor3_vals=[2,3], run_type=run_type, action=action)
