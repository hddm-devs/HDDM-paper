from collections import OrderedDict
from copy import deepcopy, copy
import time
import argparse
import sys
import kabuki

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid

import numpy as np
import pandas as pd

#utils.one_vs_others(utils.select(data, include, depends_on= {'v': ['c0', 'c1', 'c2']}, subj=False, require=lambda x:x[2]==3 and x[1]==20, estimators=estimators), 'HDDMGamma')

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


def plot_exp(data, stat, plot_type, figname, savefig):
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
    grouped = data.Err.dropna().groupby(level=(level_name, 'estimation', 'knode')).agg(stat)
    n_params = len(grouped.groupby(level=('knode',)).groups.keys())

    fig = plt.figure(figsize=(8, n_params*3))
    grid = Grid(fig, 111, nrows_ncols=(n_params, 1), add_all=True, share_all=False,
                label_mode='L', share_x=True, share_y=False, axes_pad=.25)

    for i, (param_name, param_data) in enumerate(grouped.groupby(level=('knode',))):
        ax = grid[i]
        ax.set_ylabel(param_name)
#        ax.set_ylabel(PARAM_NAMES[param_name])
#        ax.set_xlim(2, 30)
#        ax.set_yscale('log')
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


def likelihood_of_detection(data, subj, savefig):
    data = select(data, ['v_slope'], depends_on={}, subj=subj)
    detect = data['2.5q'] > 0
    grouped = detect.dropna().groupby(level=('n_trials', 'p_outliers', 'estimation')).agg(np.mean)

    fig = plt.figure()
    grid = Grid(fig, 111, nrows_ncols=(3, 1), add_all=True, share_all=False,
                label_mode='L', share_x=True, share_y=False, axes_pad=.25)
    for i_effect, (effect, ef_data) in enumerate(grouped.groupby(level='p_outliers')):
        ax = grid[i_effect]
        for i_est, (est_name, est_data) in enumerate(ef_data.groupby(level='estimation')):
            ax.errorbar(est_data.index.get_level_values('n_trials'),
                        est_data, label=est_name, lw=2.,
                        marker='o')

    ax.set_xlabel('trials')
    ax.set_ylabel('prob of detection')
    plt.legend(loc=0)
    if subj:
        title = 'regress_single'
    else:
        title = 'regress_group'
    plt.suptitle(title)

    if savefig:
        plt.savefig(title + '.png')
        plt.savefig(title + '.svg')