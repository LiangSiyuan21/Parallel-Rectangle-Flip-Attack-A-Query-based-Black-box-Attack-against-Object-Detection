"""
Implements some handy plotting functions
"""
import json
import os
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection

from utils.helper_fcts import data_path_join, get_data_dir, create_dir


def figsize(scale):
    fig_width_pt = 469.755  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def pgf_setup():
    mpl.use('pgf')
    pgf_with_latex = {  # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
        "font.family": "serif",
        "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 28,  # LaTeX default is 10pt font.
        "lines.markersize": 20,
        "lines.linewidth": 5,
        "font.size": 20,
        "legend.fontsize": 25,  # Make the legend/label fonts a little smaller
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "text.usetex": True,  # use LaTeX to write all text
        "text.latex.preamble": [],
        "figure.figsize": figsize(1.5),  # default fig size of 0.9 textwidth
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts becasue your computer can handle it :)
            r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
            r"\usepackage{amsmath}", r'\boldmath',
            r"\usepackage[scaled=0.85]{beramono}"
        ]
    }
    mpl.rcParams.update(pgf_with_latex)


def bf(text):
    """
    boldify text
    :param text:
    :return:
    """
    return r"\textbf{%s}" % text


def plot_keep_k_sign_exp(files):
    pgf_setup()
    create_dir(os.path.join(get_data_dir(), 'keep_k_res'))
    for i, file in enumerate(files):
        # load data
        dset = os.path.split(file)[1].split('_')[0]
        p = os.path.split(file)[1].split('_')[1]
        with open(file, 'r') as f:
            res = json.load(f)
        # process data
        step_size = 2
        xticks = [(_ix, bf(r"{0:.0f}%".format(_x * 100))) for _ix, _x in enumerate(res['retain_p'])][::step_size]
        res = res[dset]
        ys_rand = [1 - _y for _y in res['random']['adv_acc']]
        ys_top = [1 - _y for _y in res['top']['adv_acc']]
        plt.clf()
        ax = plt.subplot()
        ax.plot(ys_rand, label=bf('random-k'), linestyle='--', marker='.')
        ax.plot(ys_top, label=bf('top-k'), linestyle='--', marker='*')
        if i == 0: ax.legend()  # show legend for the first
        plt.xticks(*list(zip(*xticks)))
        ax.set_ylabel(bf('misclassification rate'))
        ax.set_xlabel(bf('k percent of {} coordinates'.format(dset.upper())))
        plt.tight_layout()
        plt.savefig(data_path_join('keep_k_res', 'keep_k_sign_{}_{}.pdf'.format(dset, p)))


def plt_img(img_numpy):
    """

    :param img_numpy:
    :return:
    """
    plt.imshow(img_numpy)
    plt.show()


def plot_as_3d_ts(arr,
                  xlabel,
                  ylabel,
                  zlabel,
                  xticks=None):
    """
    Plot a numpy mxn array as m n-long time series
    with a 3d view.
    :param mat:
    :return:
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(azim=-90, elev=45)
    verts = [list(zip(range(len(row) + 2), [0] + list(row / row.sum()) + [0])) for row in arr]
    zs = np.arange(arr.shape[0]) / arr.shape[0]
    poly = PolyCollection(verts, facecolors=['darkorange'], edgecolor='k')
    poly.set_alpha(0.5)
    ax.add_collection3d(poly, zs=zs, zdir='y')

    ax.set_xlim3d(0, arr.shape[1] + 2)
    # ax.set_ylim3d(0, 1)
    ax.set_zlabel(zlabel)
    # ax.set_zlim3d(0, 1)
    # ax.set_zlim(0, 1)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def plt_from_h5tbl(h5_filenames):
    """
    creates list of plots from a list of h5_file
    It is assumed that the file contains a table named `tbl`, which corresponds to
     a dataframe with the following columns
    `dataset`
    `p`
    `attack`
    `iteration`
    `batch_id`
    `total_successes`
    `total_failures`
    `total_cos_sim`
    `total_loss`
    `total_loss_queries`
    `total_crit_queries`
    :param h5_filename:
    :return:
    """
    pgf_setup()
    h5_filename = h5_filenames[0]
    _dir = os.path.join(os.path.dirname(os.path.abspath(h5_filename)),
                        '{}_plots'.format(os.path.basename(h5_filename).split('.')[0]))
    print(" storing plots at {}".format(_dir))
    create_dir(_dir)
    df = pd.DataFrame()
    for h5_filename in h5_filenames:
        _df = pd.read_hdf(h5_filename, 'tbl')
        df = df.append(_df)
    # df = pd.read_csv(h5_filename)

    sign_agg_fail_rate = 0
    other_agg_fail_rate = 0
    sign_agg_num_loss_queries = 0
    other_agg_num_loss_queries = 0
    total_sets = 0.  # to compute the aggregated perofrmance
    for (dset, p), _dp_df in df.groupby(['dataset', 'p']):
        total_sets += 1.
        tbl_df = pd.DataFrame()
        loss_fig, loss_ax = plt.subplots()
        ham_fig, ham_ax = plt.subplots()
        cos_fig, cos_ax = plt.subplots()
        scs_fig, scs_ax = plt.subplots()
        qry_fig, qry_ax = plt.subplots()
        # to compute aggregated failure rate and loss queries
        other_fail_rate = 1
        other_num_loss_queries = np.float("inf")
        sign_fail_rate = 1
        sign_num_loss_queries = np.float("inf")
        for attack, _at_df in _dp_df.groupby('attack'):
            # replace the name
            attack_name = attack.replace('Attack', '').replace('Sign', 'SignHunter').replace('Bandit',
                                                                                             'Bandits$_{TD}$').replace(
                'ZOSignHunter', 'ZOSign').replace('Simple', 'SIMBA')
            attack_name = bf(r"""\texttt{%s}""" % attack_name)
            # temp df to store for each batch the latest record (latest in terms of iteration)
            _df = _at_df.groupby('batch_id').apply(lambda _: _[_.iteration == _.iteration.max()])
            agg_at_df = _at_df.groupby('iteration').sum().reset_index()

            # update aggregated records over iterations by adding contributions from batches whose last iteration
            # was smaller than the current iteration.
            def update_fields(row):
                update_row = _df[_df.iteration < row.iteration].sum()
                for key in row.keys():
                    if key in ['iteration',
                               'batch_id',
                               'num_loss_queries_per_iteration',
                               'num_crit_queries_per_iteration']:
                        continue
                    row[key] += update_row[key]
                return row

            agg_at_df = agg_at_df.apply(update_fields, axis=1)

            its = agg_at_df.iteration.tolist()
            # success rate per iteration
            scs_rate = (agg_at_df.total_successes / (agg_at_df.total_successes + agg_at_df.total_failures)).tolist()
            # average num of queries used per successful attack per iteration
            avg_scs_loss_queries = [0 if np.isnan(_) else _ for _ in
                                    (agg_at_df.total_loss_queries / agg_at_df.total_successes).tolist()]
            # to get the number of queries per example per iteration.
            loss_queries = np.cumsum(agg_at_df.num_loss_queries_per_iteration / len(_df))
            # find batch id with max iteration
            max_batch_id = _df[_df.iteration == _df.iteration.max()].batch_id.tolist()[0]
            print(max_batch_id)
            loss_queries = _at_df[_at_df.batch_id == max_batch_id].num_loss_queries_per_iteration.cumsum().tolist()
            # average cosine / ham / loss values per example (be it successful or failed)
            avg_cos_sim = (agg_at_df.total_cos_sim / (agg_at_df.total_successes + agg_at_df.total_failures)).tolist()
            avg_ham_sim = (agg_at_df.total_ham_sim / (agg_at_df.total_successes + agg_at_df.total_failures)).tolist()
            avg_loss = (agg_at_df.total_loss / (agg_at_df.total_successes + agg_at_df.total_failures)).tolist()

            scs_ax.plot(loss_queries, scs_rate, label=attack_name)
            ham_ax.plot(loss_queries, avg_ham_sim, label=attack_name)
            cos_ax.plot(loss_queries, avg_cos_sim, label=attack_name)
            loss_ax.plot(loss_queries, avg_loss, label=attack_name)
            if scs_rate[0] > 1e-5:  # to complete the graph from 0 success rate (for which it's zero loss queries)
                qry_ax.plot([0] + scs_rate, [0] + avg_scs_loss_queries, label=attack_name)
            else:
                qry_ax.plot(scs_rate, avg_scs_loss_queries, label=attack_name)

            # Compute std loss queries: some bookkeeping is needed to extract
            # the number of queries used for each datapoint and compute the std accordingly

            def process_at_df(_):
                """
                takes the at_df dataframe and replaces the num_loss_queries by the cumulative sum
                since for some methods the queries used vary from one iteration to the other
                """
                _['cum_loss_queries'] = _.num_loss_queries_per_iteration.cumsum()
                _['cum_crit_queries'] = _.num_crit_queries_per_iteration.cumsum()
                return _

            _std_df = _at_df.groupby(['batch_id']).apply(
                process_at_df).reset_index(drop=True)

            std_df = _std_df.groupby(['batch_id', 'total_successes']).apply(
                lambda _: _[_.iteration == _.iteration.min()]).reset_index(drop=True)

            total_loss_query = std_df.groupby('batch_id').apply(
                lambda _: _.cum_loss_queries * (
                    _.total_successes.diff().fillna(_.total_successes))).sum()
            total_loss_query_squared = std_df.groupby('batch_id').apply(
                lambda _: _.cum_loss_queries ** 2 * (
                    _.total_successes.diff().fillna(_.total_successes))).sum()
            total_success = _at_df.groupby('batch_id').apply(
                lambda _: _[_.iteration == _.iteration.max()]['total_successes']).sum()

            total_crit_query = std_df.groupby('batch_id').apply(
                lambda _: _.cum_crit_queries * (
                    _.total_successes.diff().fillna(_.total_successes))).sum()

            avg_loss_queries = total_loss_query / total_success
            avg_crit_queries = total_crit_query / total_success
            std_loss_queries = np.sqrt(total_loss_query_squared / (total_success - 1) -
                                       total_success * avg_loss_queries ** 2 / (total_success - 1))

            print("attack: {}, l-{}, failure rate: {}, avg. loss.: {}, std. loss.: {}".format(
                attack_name,
                p,
                1 - scs_rate[-1],
                avg_loss_queries,
                std_loss_queries
            ))

            tbl_df = tbl_df.append(pd.DataFrame.from_records([{
                'attack': attack_name,
                'p': p,
                'failure_rate': 1 - scs_rate[-1],
                'avg. loss': avg_scs_loss_queries[-1] + avg_crit_queries,
                #'std. loss': std_loss_queries
            }]), ignore_index=True)

            if attack == 'SignAttack':
                sign_fail_rate = 1 - scs_rate[-1]
                sign_num_loss_queries = avg_scs_loss_queries[-1]
            elif attack == 'RandAttack': pass
            else:
                other_fail_rate = min(other_fail_rate, 1 - scs_rate[-1])
                other_num_loss_queries = min(other_num_loss_queries, avg_scs_loss_queries[-1])

        sign_agg_fail_rate += sign_fail_rate
        other_agg_fail_rate += other_fail_rate
        # 10000 the maximum budget allocation
        sign_agg_num_loss_queries += sign_num_loss_queries * (1 - sign_fail_rate) + sign_fail_rate * 10000
        other_agg_num_loss_queries += other_num_loss_queries * (1 - other_fail_rate) + other_fail_rate * 10000

        print("Data set: {}".format(dset))
        print(tbl_df.set_index('attack'))


        # if you 'd like to show all the legends here
        ham_ax.legend()
        cos_ax.legend()
        loss_ax.legend()
        #if dset == 'mnist' and p == 'inf':
        qry_ax.legend(loc='upper left')
        #elif p == 'inf':
        scs_ax.legend(loc=4)

        scs_ax.set_xlabel(bf('\# queries'))
        ham_ax.set_xlabel(bf('\# queries'))
        cos_ax.set_xlabel(bf('\# queries'))
        loss_ax.set_xlabel(bf('\# queries'))
        qry_ax.set_xlabel(bf('success rate'))

        scs_ax.set_ylabel(bf('success rate'))
        ham_ax.set_ylabel(bf('average Hamming similarity'))
        cos_ax.set_ylabel(bf('average cosine similarity'))
        loss_ax.set_ylabel(bf('average loss'))
        qry_ax.set_ylabel(bf('average \# queries'))

        scs_fig.tight_layout()
        ham_fig.tight_layout()
        cos_fig.tight_layout()
        loss_fig.tight_layout()
        qry_fig.tight_layout()

        scs_fig.savefig(os.path.join(_dir, '{}_{}_scs_plt.pdf'.format(dset, p)))
        ham_fig.savefig(os.path.join(_dir, '{}_{}_ham_plt.pdf'.format(dset, p)))
        cos_fig.savefig(os.path.join(_dir, '{}_{}_cos_plt.pdf'.format(dset, p)))
        loss_fig.savefig(os.path.join(_dir, '{}_{}_loss_plt.pdf'.format(dset, p)))
        qry_fig.savefig(os.path.join(_dir, '{}_{}_qrt_plt.pdf'.format(dset, p)))

    sign_agg_fail_rate /= total_sets
    other_agg_fail_rate /= total_sets
    sign_agg_num_loss_queries /= total_sets
    other_agg_num_loss_queries /= total_sets

    # This will raise a warning (div by zero) if no SignHunter AND no other algorithm is included in the passed .tbl files
    # which you can ignore safely.
    print("SignHunter uses {} times less queries and fails {} times less often than SOTA combined. "
          "These numbers are valid ONLY when the data of signhunter  AND one or more other algorithms are included.".format(
        other_agg_num_loss_queries / sign_agg_num_loss_queries,
        other_agg_fail_rate / sign_agg_fail_rate
    ))


def plot_adv_cone_res(pickle_fname, is_legend=True):
    pgf_setup()
    _dir = os.path.join(os.path.dirname(os.path.abspath(pickle_fname)),
                        '{}_plots'.format(os.path.basename(pickle_fname).split('.')[0]))
    print(" storing plots at {}".format(_dir))
    create_dir(_dir)

    with open(pickle_fname, 'rb') as f:
        res_ = pickle.load(f)

    setups = [_ for _ in res_.keys() if _ != 'epsilon' \
              and _ != 'adv-cone-orders' \
              and _ != 'sign-hunter-step' \
              and _ != 'num_queries']

    plot_fnames = []
    for ie, _eps in enumerate(res_['epsilon']):
        plt.clf()
        for setup, color in zip(setups, ['red', 'blue']):
            res = res_[setup]
            _m = '_{\\text{adv-ens4}}' if 'ens' in setup else ''
            plt.plot(res_['adv-cone-orders'], res['grad-sign'][ie, :], label=bf("v3$%s$-\\texttt{GAAS}" % _m),
                     linewidth=2, linestyle='--', color=color)
            plt.plot(res_['adv-cone-orders'], res['sign-hunter'][ie, :], label=bf("v3$%s$-\\texttt{SAAS}" % _m),
                     linewidth=3, linestyle='-', color=color)
            plt.xlabel(bf('k'))
            plt.xticks([1, 50, 100])
            plt.ylim(0.0, 0.9)
            plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8])
            if is_legend:
                plt.legend()
                plt.ylabel(bf('Probability'))
        plot_fnames.append(os.path.join(_dir, 'eps-{}.pdf'.format(int(_eps * 255))))
        plt.savefig(plot_fnames[-1])

    return plot_fnames


if __name__ == '__main__':
    # To plot k sign plots
    # plot_keep_k_sign_exp([
    #     '../../data/keep_k_res/mnist_linf_res.json',
    #     '../../data/keep_k_res/cifar10_linf_res.json',
    #     '../../data/keep_k_res/imagenet_linf_res.json',
    #     '../../data/keep_k_res/mnist_l2_res.json',
    #     '../../data/keep_k_res/cifar10_l2_res.json',
    #     '../../data/keep_k_res/imagenet_l2_res.json'
    # ])
    #
    # plot tuning results
    # plt_from_h5tbl(['../../data/blackbox_attack_exp/tune_tbl.h5'])
    #
    # # plot challenges trace
    # plt_from_h5tbl(['../../data/blackbox_attack_exp/adv_cifar10_sign_tbl.h5'])
    # plt_from_h5tbl(['../../data/blackbox_attack_exp/adv_mnist_sign_tbl.h5'])
    # plt_from_h5tbl(['../../data/blackbox_attack_exp/ens_imagenet_tbl.h5',
    #                 '../../data/blackbox_attack_exp/ens_imagenet_tbl_1.h5',
    #                 '../../data/blackbox_attack_exp/ens_imagenet_tbl_2.h5',
    #                 '../../data/blackbox_attack_exp/ens_imagenet_tbl_3.h5',
    #                 ])
    #
    # # plot all
    plt_from_h5tbl([
         #'../../data/blackbox_attack_exp/mnist_naive_tbl.h5',
         #'../../data/blackbox_attack_exp/mnist_sign_tbl.h5',
        '../../data/blackbox_attack_exp/cifar_simple_attack_tbl.h5',
         '../../data/blackbox_attack_exp/cifar10_linf_sign_tbl.h5',
        #'../../data/blackbox_attack_exp/cifar_simulated_l2_nes_bandit__linf_2.291_tbl.h5',
         '../../data/blackbox_attack_exp/cifar10_l2_sign_tbl.h5',
        #'../../data/blackbox_attack_exp/cifar10_l2_sota_tbl.h5',
         #'../../data/blackbox_attack_exp/mnist_cifar_rand_tbl.h5',

    #     '../../data/blackbox_attack_exp/shc_mnist_cifar_linf_tbl.h5',
    #     '../../data/blackbox_attack_exp/shc_mnist_cifar_l2_tbl.h5'
       # '../../data/blackbox_attack_exp/imagenet_naive_tbl.h5',
       # '../../data/blackbox_attack_exp/imagenet_l2_sign_tbl.h5',
       # '../../data/blackbox_attack_exp/imagenet_rand_tbl.h5',
        #'../../data/blackbox_attack_exp/shc_imagenet_linf_tbl.h5',
        #'../../data/blackbox_attack_exp/shc_imagenet_l2_tbl.h5',

     ])



    plt_from_h5tbl([
        #'../../data/blackbox_attack_exp/mnist_sota_tbl.h5',
        #'../../data/blackbox_attack_exp/mnist_sign_tbl.h5',
        #'../../data/blackbox_attack_exp/mnist_cifar_rand_tbl.h5',
        #'../../data/blackbox_attack_exp/imagenet_rand_tbl.h5',
        # '../../data/blackbox_attack_exp/cifar10_l2_sign_tbl.h5',
        # '../../data/blackbox_attack_exp/cifar10_l2_sota_tbl.h5',
        # '../../data/blackbox_attack_exp/imagenet_l2_sota_tbl.h5',
        # '../../data/blackbox_attack_exp/imagenet_l2_sign_tbl.h5',
        ])

    # # plot adv cone plots
    # adv_cone_files = [  'adv-cone_step-10_query-1000.p',
    #                     'adv-cone_step-10_query-500.p',
    #                     'adv-cone_step-16_query-1000.p',
    #                     'adv-cone_step-16_query-500.p',
    #                     'adv-cone_step-4_query-1000.p',
    #                     'adv-cone_step-4_query-500.p']
    # for _i, _f in enumerate(adv_cone_files):
    #     plot_adv_cone_res(data_path_join('adv_cone_exp', _f), is_legend= _i == 4)
