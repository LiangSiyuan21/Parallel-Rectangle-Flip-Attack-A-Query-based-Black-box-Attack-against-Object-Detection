"""
Implement some functions for latexifying results
"""
import json

import pandas as pd

pd.options.display.max_colwidth = 255
import numpy as np
import os
from utils.helper_fcts import get_config_dir, create_dir, get_data_dir


def df_2_tex(df, filepath):
    """
    Writes a df to tex file
    :param df:
    :param filepath:
    :return:
    """
    tex_prefix = r"""\documentclass{standalone}
    \usepackage{booktabs}
    \usepackage{bm}
    \begin{document}
    
    \resizebox{\textwidth}{!}{
    """

    tex_suffix = r"""}
    
    \end{document}"""

    with open(filepath, 'w') as f:
        f.write(tex_prefix)
        tex_string = df.to_latex(escape=False, multicolumn_format='l', index=True)
        print(tex_string)
        f.write(tex_string)
        f.write(tex_suffix)


def beautify(elem):
    """
    function to check if strings are numbers and format them
    :param elem:
    :return:
    """
    # print(elem)
    if type(elem) == int: return "$%d$" % elem
    if type(elem) == float: return "$%.2f$" % elem
    if elem.replace('.', '', 1).isdigit():
        # print(elem, "before")
        elem = "${:.2f}$".format(float(elem))
        # print(elem, "after")
    # print("elem final", elem)
    return elem


def file_rank(filename):
    """
    assign a rank to the file can be used for sorting
    :param filename:
    :return:
    """
    p_order = {'linf': 0, 'l2': 1}  # p norm
    d_order = {'mnist': 0, 'cifar10': 1, 'imagenet': 2}  # d dataset

    p_val = 0
    d_val = 0
    for key in p_order.keys():
        p_val = p_order[key] if filename.rfind(key) != -1 else p_val
    for key in d_order.keys():
        d_val = d_order[key] if filename.rfind(key) != -1 else d_val

    return d_val * len(p_order) + p_val


def res_json_2_tbl_latex(json_file):
    with open(json_file, 'r') as f:
        res = json.load(f)

    level_one_col_map = {
        'failure_rate': 0,
        'average_num_loss_queries': 1,
        'average_num_crit_queries': 2
    }
    level_two_col_map = {
        'inf': 0,
        '2': 1
    }

    level_one_offset = len(level_two_col_map)
    num_cols = len(level_one_col_map) * len(level_two_col_map)

    # we will order the result as a matrix then cast it as pd

    for dset, dset_res in res.items():
        dset_res = sorted(dset_res, key=lambda _: _['config']['attack_name'])
        att_ns = []

        tbl = []
        for i, attack_res in enumerate(dset_res):
            att_n = attack_res['config']['attack_name']
            level_two_key = attack_res['config']['p']
            try:
                j = att_ns.index(att_n)
            except ValueError:
                j = len(att_ns)
                att_ns.append(att_n)
                tbl.append([np.nan] * num_cols)

            for key, idx in level_one_col_map.items():
                tbl[j][idx * level_one_offset + level_two_col_map[level_two_key]] = attack_res[key]

        lps = [r"""$\bm\ell_\infty$""", r"""$\bm\ell_2$"""]
        tups = ([(r"""\bf{Failure Rate}""", lp) for lp in lps] +
                [(r"""\bf{Avg. Loss Queries}""", lp) for lp in lps] +
                [(r"""\bf{Avg. Stop Queries}""", lp) for lp in lps]
                )
        att_ns = [_.replace('Attack', '').replace('Sign', 'SignHunter').replace('Bandit', 'Bandits$_{TD}$').replace(
            'ZOSignHunter', 'ZOSign') for _ in att_ns]
        att_ns = [r"""\texttt{%s}""" % _ for _ in att_ns]
        df = pd.DataFrame(np.array(tbl), columns=pd.MultiIndex.from_tuples(tups), index=att_ns)
        df.index.name = r"""\bf{Attack}"""
        # df = df[df.columns[-1:] + df.columns[:-1]]
        df = df.applymap(beautify)
        # drop the average stop queries
        df = df.drop(level=0, columns=r"""\bf{Avg. Stop Queries}""")

        df_2_tex(df, json_file.replace('.json', '_{}.tex'.format(dset)))


def config_json_2_tbl_latex(json_files):
    """
    take a list of json file path names for the *same attack* but on different datasets / constraints
    and export a table of its parameters
    :param json_files:
    :return:
    """
    _dir = os.path.join(get_data_dir(), 'tex_tbls')
    create_dir(_dir)
    attacks = set(map(lambda _: os.path.basename(_).split('_')[1], json_files))
    # dsets = set(map(lambda _: os.path.basename(_).split('_')[0], json_files))

    param_dict = {
        'lr': r"""$\eta$ (image $\ell_p$ learning rate)""",
        'fd_eta': r"""$\delta$ (finite difference probe)""",
        'prior_lr': r"""$\kappa$ (online convex optimization learning rate)""",
        'epsilon': r"""$\epsilon$ (allowed perturbation)""",
        'prior_size': r"""Tile size (data-dependent prior)""",
        'q': r"""$q$ (number of finite difference estimations per step)""",
        'prior_exploration': r"""$\zeta$ (bandit exploration)""",
        'num_eval_examples': r"""Test set size""",
        'max_loss_queries': r"""Max allowed queries""",
        'attack_name': r"""Attack name"""
    }
    assert len(attacks) == 1, "json files should all be for one attack method"
    attack_name = attacks.pop()
    df = pd.DataFrame()
    for json_file in json_files:
        with open(json_file, 'r') as f:
            config = json.load(f)

        dset_name = config['dset_name']
        p = config['attack_config']['p'].replace('inf', '\infty')
        vals = []
        hparams = []
        for key, val in config['attack_config'].items():
            if key == 'p' or key == 'data_size': continue
            hparams.append(param_dict[key])
            vals.append(val)
        hparams.append(param_dict['num_eval_examples'])
        vals.append(config['num_eval_examples'])
        hparams.append(param_dict['attack_name'])
        vals.append(config['attack_name'])
        _df = pd.DataFrame({r"""\bf{Hyperparameter}""": hparams,
                            r"""\texttt{{{}}} $\ell_{{{}}}$""".format(dset_name, p): vals}).set_index(
            r"""\bf{Hyperparameter}""")

        df = pd.concat([df, _df], axis=1)
    # df.columns = pd.MultiIndex.from_product([[r"""\bf{Value}"""], df.columns])
    df.columns = pd.MultiIndex.from_tuples([tuple((r"""\bf{Value} """ + col).split()) for col in df.columns])

    df.applymap(beautify)
    df_2_tex(df, os.path.join(_dir, '{}_param_tbl.tex'.format(attack_name)))


if __name__ == '__main__':
    # to export summary table of algorithms (this should be used at the end of attacks/blackbox/run_attack/)
    is_export_res = True
    # res_json_2_tbl_latex('../../data/blackbox_attack_exp/mnist_res.json')
    if is_export_res:
        res_json_2_tbl_latex('../../data/blackbox_attack_exp/mnist_res.json')
        res_json_2_tbl_latex('../../data/blackbox_attack_exp/cifar_l2_res.json')
        res_json_2_tbl_latex('../../data/blackbox_attack_exp/cifar10_linf_res.json')

    # to export parameter table of algorithms
    is_export_params = False
    if is_export_params:
        import glob


        def get_jsonfiles(pattern):
            return sorted(glob.glob(os.path.join(get_config_dir(), pattern)), key=lambda _: file_rank(_))


        json_files = get_jsonfiles('*bandit*')
        config_json_2_tbl_latex(json_files)

        json_files = get_jsonfiles('*nes*')
        config_json_2_tbl_latex(json_files)

        json_files = get_jsonfiles('*zosignsgd*')
        config_json_2_tbl_latex(json_files)

        json_files = get_jsonfiles('*sign_*')
        config_json_2_tbl_latex(json_files)
