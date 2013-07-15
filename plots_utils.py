from collections import OrderedDict
from copy import deepcopy, copy
import time
import argparse
import sys
import kabuki
import scikits.bootstrap as bootstrap
import matplotlib.pyplot as plt
try:
    from mpl_toolkits.axes_grid1 import Grid
except IOError:
   pass
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import ttest_rel, ttest_1samp, scoreatpercentile

#utils.one_vs_others(utils.select(data, include, depends_on= {'v': ['c0', 'c1', 'c2']}, subj=False, require=lambda x:x[2]==3 and x[1]==20, estimators=estimators), 'HDDMGamma')

def binomial_ste(a):
    p = a.mean()
    return np.sqrt( p * (1- p) / len(a))

def ste(a):
    return np.std(a) / np.sqrt(len(a))

def upper_trimmed_mean(a, percentile=95):
    limit = scoreatpercentile(a,percentile, interpolation_method='higher')
    return np.mean(a[a < limit])

def trimmed_mean(a, per=5):
    n = int(np.ceil(len(a)*per/100))
    a = a.copy()
    a.sort()
    return np.mean(a[:-n])

def trimmed_2side_mean(a, per=5):
    n = int(np.ceil(len(a)* ((per / 2.) / 100)))
    a = a.copy()
    a.sort()
    return np.mean(a[n:-n])

def trimmed_2side_ste(a, per=5):
    n = int(np.ceil(len(a)* ((per / 2.) / 100)))
    a = a.copy()
    a.sort()
    return np.std(a[n:-n]) / np.sqrt(len(a) - 2*n)

def trimmed_2side_ci(a, per=5):
    n = int(np.ceil(len(a)* ((per / 2.) / 100)))
    a = a.copy()
    a.sort()
    a = a[n:-n]
    return bootstrap.ci(a, np.mean, n_samples=10000)


def select(stats, param_names, depends_on, subj=True, require=None, estimators=None):

    if isinstance(param_names, str):
        param_names = [param_names]

    if estimators is None:
        if subj:
            estimators = ['SingleMAP', 'HDDMsharedVar', 'HDDMTruncated', 'Quantiles_subj',
            'SingleMAPoutliers', 'HDDMOutliers', 'HDDMGamma', 'HDDMRegressor', 'SingleRegressor']
        else:
            estimators = ['HDDMTruncated', 'Quantiles_group', 'HDDMsharedVar', 'HDDMOutliers',
            'Quantiles_subj', 'HDDMGamma', 'HDDMRegressor', 'SingleRegressor']

    extracted = {}

    if require is None:
        require=lambda x:True

    index = stats.index
    for name in param_names:
        for cond in depends_on.get(name, [None]):
            if (cond is not None):
                fullname = "%s(%s)" % (name, cond)
            else:
                fullname = name

            selected = []
            for ix in index:
                if subj:
                    if ix[-4] in estimators and ix[-1].startswith(name) and 'subj' in ix[-1]:
                        if ((cond is None) or (('(%s)' % cond) in ix[-1])) and require(ix):
                            selected.append(ix)
                else:
                    if ix[-4] in estimators and ((ix[-1] == name) or (ix[-1].startswith(name + '('))):
                        if ((cond is None) or (('(%s)' % cond) in ix[-1])) and require(ix):
                            selected.append(ix)

            extracted[fullname] = stats.ix[selected]

    return pd.concat(extracted, names=['knode'])


def plot_exp(data, stat, plot_type, figname, savefig, col='abserr'):

    level_name, xlabel = get_levelname_and_xlabel(plot_type)
    grouped = data[col].dropna().groupby(level=(level_name, 'estimation', 'knode')).agg(stat)
    n_params = len(grouped.groupby(level=('knode',)).groups.keys())

    fig = plt.figure(figsize=(8, n_params*3))
    grid = Grid(fig, 111, nrows_ncols=(n_params, 1), add_all=True, share_all=False,
                label_mode='L', share_x=True, share_y=False, axes_pad=.25)

    for i, (param_name, param_data) in enumerate(grouped.groupby(level=('knode',))):
        ax = grid[i]
        ax.set_ylabel(param_name)
        for est_name, est_data in param_data.groupby(level=['estimation']):
            ax.errorbar(est_data.index.get_level_values(level_name),
                        est_data, label=est_name, lw=2.,
                        marker='o')

    ax.set_xlabel(xlabel)
    plt.legend(loc=0)
    title = '%s_exp_%s'%(plot_type, figname)
    plt.suptitle(title)

    if savefig:
        plt.savefig(title + '.png')
        plt.savefig(title + '.svg')



def plot_errors(data, stat, plot_type, savefig, col='abserr', main='HDDM2', other='ML'):

    if stat.func_name == 'mean':
        err_stat = lambda a:np.std(a)/sqrt(len(a))
        err_stat.func_name = 'ste'
    elif stat.func_name == 'trimmed_2side_mean':
        err_stat = trimmed_2side_ste
        err_stat = trimmed_2side_ci


    level_name, xlabel = get_levelname_and_xlabel(plot_type)
    err = (data.xs(other,level='estimation') - data.xs(main,level='estimation'))[col].dropna()
    grouped = err.groupby(level=(level_name, 'knode')).agg([stat])
    abs_ci = err.groupby(level=(level_name, 'knode')).apply(err_stat)
    abs_ci = np.vstack(abs_ci.values)
    grouped['low_ci'] = grouped[stat.func_name] - abs_ci[:,0]
    grouped['high_ci'] = abs_ci[:,1] - grouped[stat.func_name]
    n_params = len(grouped.groupby(level=('knode',)).groups.keys())

    fig = plt.figure(figsize=(8, n_params*3))
    grid = Grid(fig, 111, nrows_ncols=(n_params, 1), add_all=True, share_all=False,
                label_mode='L', share_x=True, share_y=False, axes_pad=.25)

    for i, (param_name, param_data) in enumerate(grouped.groupby(level=('knode',))):
        ax = grid[i]
        ax.set_ylabel(param_name)
        ax.errorbar(param_data.index.get_level_values(level_name),
                    param_data[stat.func_name], yerr=param_data[['low_ci', 'high_ci']].values.T,
                    label=param_name, lw=2.,
                    marker='o')
        if param_name != 't':
            ax.set_ylim(0, ax.get_ylim()[1])
        else:
            ax.set_ylim(-0.0002, ax.get_ylim()[1])

        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        [x.label.set_fontsize(20) for x in ax.yaxis.get_major_ticks()]
        [x.label.set_fontsize(20) for x in ax.xaxis.get_major_ticks()]


    ax.set_xlabel(xlabel)
    title = '%s_exp_%s'%(plot_type, 'errors')
    plt.suptitle(title)

    if savefig:
        plt.savefig(title + '.png')
        plt.savefig(title + '.svg')


def plot_recovery_exp(data, tag='', abs_min=-5, abs_max=5, gridsize=100, save=True):

    data = data[['truth', 'estimate','Err']].dropna()
    ni = len(data.dropna().groupby(level=['estimation']))
    nj = len(data.dropna().groupby(level=('knode',)))

    data = data[(data['estimate'] > abs_min) & (data['estimate'] < abs_max)]
    data_params = data.groupby(level=('knode',))[['truth', 'estimate']]
    mini = data_params.min().min(axis=1)
    maxi = data_params.max().max(axis=1)
    print mini
    print maxi

    fig = plt.figure(figsize=(9, 3*nj))
    grid = Grid(fig, 111, nrows_ncols=(nj, ni), add_all=True, share_all=False, label_mode='L', share_x=False, share_y=False)
    for i, (est_name, est_data) in enumerate(data.dropna().groupby(level=['estimation'])):
        nj = len(est_data.groupby(level=('knode',)))
        for j, (param_name, param_data) in enumerate(est_data.groupby(level=('knode',))):
            ax = grid.axes_column[i][j] #plt.subplot2grid((nj, ni), (j, i))
            # if i == 0:
            #     ax.set_title(est_name)
            #ax.plot(param_data.truth, param_data.estimate, 'x', label=est_name)
            minimaxi = (mini[param_name], maxi[param_name])
            ax.set_xlim(minimaxi)
            ax.set_ylim(minimaxi)
            ax.set_xlabel(est_name)
            ax.set_ylabel(PARAM_NAMES[param_name])
            kwargs = {'gridsize': gridsize, 'bins': 'log', 'extent': (mini[param_name], maxi[param_name], mini[param_name], maxi[param_name])}
            ax.hexbin(param_data.truth, param_data.estimate, label='post pred lb', **kwargs)
#            kabuki.debug_here()

#            plt.legend()

    if save:
        plt.savefig('recovery_exp_%s.png'%(tag), dpi=600)
        plt.savefig('recovery_exp_%s.svg'%(tag))

def one_vs_others(data, main_estimator, tag='', gridsize=100, save=False, fig=None, color='b'):

    data = data[['truth', 'estimate','Err']].dropna()
    data_params = data.groupby(level=('knode',))[['truth', 'estimate']]
    mini = data_params.min().min(axis=1)
    maxi = data_params.max().max(axis=1)
    print mini
    print maxi

    grouped_data = data.groupby(level=['estimation'])
    main_data = grouped_data.get_group(main_estimator)
    ni = len(grouped_data) - 1
    nj = len(data.groupby(level=('knode',)))
    if fig is None:
        fig = plt.figure()#figsize=(9, 3*nj))
    counter = 0
    for j, (param_name, param_data) in enumerate(data.groupby(level=('knode',))):
        for i, (est_name, est_data) in enumerate(param_data.groupby(level=('estimation',))):
            if est_name == main_estimator:
                continue
            counter = counter + 1
            ax = fig.add_subplot(nj, ni, counter)
            # minimaxi = (mini[param_name], maxi[param_name])
#            ax.set_xlim(minimaxi)
#            ax.set_ylim(minimaxi)
            ax.set_xlabel(est_name)
            # ax.set_ylabel(PARAM_NAMES[param_name])
            ax.set_ylabel(param_name)
            # kwargs = {'gridsize': gridsize, 'bins': 'log'}
#                      'extent': (mini[param_name], maxi[param_name], mini[param_name], maxi[param_name])}
#            ax.hexbin(np.abs(est_data.relErr), main_data.ix[param_name].relErr, **kwargs)
#            ax.scatter(np.abs(est_data.relErr), np.abs(main_data.ix[param_name].relErr))
            ax.scatter(est_data.Err, main_data.ix[param_name].Err,c=color)
            lb = min(ax.get_xlim()[0], ax.get_ylim()[0])
            ub = min(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([lb, ub], [lb, ub])
            # ax.axis('equal')
            ax.plot()
            # ax.axis('scaled')


    if save:
        plt.savefig('recovery_exp_%s.png'%(tag), dpi=600)
        plt.savefig('recovery_exp_%s.svg'%(tag))

    return fig

def likelihood_of_detection(data, plot_type, figname=None, savefig=False):

    level_name, xlabel = get_levelname_and_xlabel(plot_type)
    if plot_type == 'regress':
        h_method='HDDMRegressor'
        h_param = 'v_slope'
        ttest_methods = ['MLRegressor', 'SingleRegressor']
        ttest_param = 'v_slope_subj'
        subj_ttest = subj_ttest_1samp
        ncols = 3
    else:
        h_method = 'HDDM2'
        h_param = 'v_shift'
        ttest_methods = ['Quantiles_subj', 'ML', 'HDDM2Single']
        ttest_param = 'v_subj'
        subj_ttest = subj_ttest_rel
        ncols = 1

    fig = plt.figure()
    grid = Grid(fig, 111, nrows_ncols=(ncols, 1), add_all=True, share_all=False,
                label_mode='L', share_x=True, share_y=False, axes_pad=.25)
    for i_effect, (effect, ef_data) in enumerate(data.groupby(level='p_outliers')):
        ax = grid[i_effect]

        #HDDM2 likelihood
        hddm2_shift = ef_data.xs([h_method, h_param], level=['estimation','param'])
        detect = hddm2_shift['2.5q'] > 0
        grouped = detect.groupby(level=level_name).agg((np.mean, binomial_ste))
        ax.errorbar(grouped.index.values,
                    grouped['mean'], yerr=grouped['binomial_ste'], label='HDDM', lw=2.,
                    marker='o', markersize=10)

        for method in ttest_methods:
            shift = ef_data.xs(method, level='estimation').select(lambda x:ttest_param in x[-1])
            res_ttest = shift.estimate.groupby(level=[level_name, 'param_seed']).agg(subj_ttest)
            grouped = res_ttest.groupby(level=level_name).agg((np.mean, binomial_ste))
            ax.errorbar(grouped.index.values,
                        grouped['mean'], yerr=grouped['binomial_ste'], label=method, lw=2.,
                        marker='o', markersize=10)

        ax.set_ylim(-0.1,1.1)
        [x.label.set_fontsize(20) for x in ax.yaxis.get_major_ticks()]
        [x.label.set_fontsize(20) for x in ax.xaxis.get_major_ticks()]

    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel('prob of detection', fontsize=20)

    plt.legend(loc=0)
    title = 'likelihood of detection'
    plt.suptitle(title)

    if savefig:
        if figname is None:
            figname = title + " - " + plot_type
        plt.savefig(figname + '.png')
        plt.savefig(figname + '.svg')


def subj_ttest_rel(data, threshold=0.025):
    """
    compute ttest on results of single subjects models with 2 conditions
    Output:
        is_rejected <boolean> : whether the null hypothesis was rejected
    """
    c0 = data.select(lambda x:x[-1].startswith('v_subj(c0)'))
    c1 = data.select(lambda x:x[-1].startswith('v_subj(c1)'))
    c0 = c0.sort_index()
    c1 = c1.sort_index()
    t_res, p_value = ttest_rel(c0.values, c1.values)
    return p_value < threshold

def subj_ttest_1samp(data, threshold=0.025):
    """
    compute ttest on results of single subjects models used in regression experiment
    Output:
        is_rejected <boolean> : whether the null hypothesis was rejected
    """
    t_res, p_value = ttest_1samp(data.values, 0)
    return p_value < threshold

def get_levelname_and_xlabel(plot_type):
    if plot_type == 'subjs':
        level_name = 'n_subjs'
        xlabel = 'subjs'
    elif plot_type == 'trials':
        level_name = 'n_trials'
        xlabel = 'trials'
    elif plot_type == 'regress':
        level_name = 'n_trials'
        xlabel = 'trials'
    elif plot_type == 'priors':
        level_name = 'n_trials'
        xlabel = 'trials'
    else:
        raise ValueError('unknown plot_type')

    return level_name, xlabel

def small_correlation_test(data, rho, iter=100):
    """
    funtion used by correlation test
    """
    randn = np.random.randn
    r = [pearsonr(rho*data.truth + np.sqrt(1-rho**2)*randn(len(data))*0.2, data.estimate)[1] < 0.05 for x in range(iter)]
    return np.mean(r)

def correlation_test(data, plot_type, param='a', effect=0.5, iter=100, savefig=False):

    level_name, xlabel = get_levelname_and_xlabel(plot_type)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    corr = lambda x,rho=effect, iter=iter: small_correlation_test(x, rho, iter)

    for method in ['Quantiles_subj', 'ML', 'HDDM2Single', 'HDDM2']:
        a_data = data.xs(method, level='estimation').select(lambda x:param + '_subj' in x[-1])
        res_test = a_data.groupby(level=[level_name, 'param_seed']).apply(corr)
        grouped = res_test.groupby(level=level_name).agg(np.mean)
        ax.errorbar(grouped.index.values,
                    grouped, label=method, lw=2.,
                    marker='o')


    ax.set_xlabel(xlabel)
    ax.set_ylabel('prob of detection')
    plt.legend(loc=0)
    title = 'correlation detection'
    plt.title(title)

    if savefig:
        plt.savefig(title + '.png')
        plt.savefig(title + '.svg')

