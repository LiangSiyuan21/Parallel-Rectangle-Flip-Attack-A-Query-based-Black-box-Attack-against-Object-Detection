"""
Implements the base class for black-box attacks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch as ch
import torch
from torch import Tensor as t
import pdb
from mmdet.core import bbox2result

from blackbox_attack.utils.compute_fcts import l2_proj_maker, linf_proj_maker
import attack_demo.util as demo_utils
import sys

class BlackBoxAttack(object):
    def __init__(self, max_loss_queries=np.inf,
                 max_crit_queries=np.inf,
                 epsilon=0.5, p='inf', lb=0., ub=1., name = "nes", attack_model=None, attack_mode=None, attack_logistics=None, loss=None, targeted=None, ori_img=None, model_name=None, zeta=None, lambda1=None, patch_attack=None, keypoints_models=None):
        """
        Args:
            max_loss_queries ([int]): [ maximum number of calls allowed to loss oracle per data pt]
            epsilon ([int]): [radius of lp-ball of perturbation]
            p ([str]): [specifies lp-norm  of perturbation]
            fd_eta ([int]): [forward difference step]
            lb ([float]]): [data lower bound]
            ub ([float]): [data lower bound]
            name ([str]): [name of the attack method]
            attack_model ([model]): [object detection model]
            attack_mode ([bool]): [if True, we will attack the detection model]
            loss ([list]): [optimize object function]
            targeted ([bool]): [if targeted attack, the value is True]
            ori_img ([tensor]): [clean img]
            model_name ([str]): [the name of the attacked model]
            zeta ([float]): [the threshold of the IoU]
            lambda1 ([float]): [the banlance of the iou loss]
            patch_attack([str]): [the way to attacking images for object detection]
        """
        assert p in ['inf', '2'], "L-{} is not supported".format(p)
        assert not (np.isinf(max_loss_queries) and np.isinf(max_crit_queries)), "one of the budgets has to be finite!"

        # self.epsilon = epsilon
        self.epsilon = epsilon * (ub - lb)
        self.p = p
        self.max_loss_queries = max_loss_queries
        self.max_crit_queries = max_crit_queries
        self.total_loss_queries = 0
        self.total_crit_queries = 0
        self.total_successes = 0
        self.total_failures = 0
        self.lb = lb
        self.ub = ub
        self.name = name
        self.attack_model = attack_model
        self.attack_mode = attack_mode
        self.attack_logistics = attack_logistics
        self.loss = loss.split(',')
        self.targeted = targeted
        self.ori_img = ori_img
        self.model_name = model_name
        self.zeta = zeta
        self.lambda1 = lambda1
        self.patch_attack = patch_attack
        self.keypoints_models = keypoints_models
        # self.square_expansion = square_expansion
        # self.attack_parallel = attack_parallel
        # self.square_init = square_init
        # the _proj method takes pts and project them into the constraint set:
        # which are
        #  1. epsilon lp-ball around xs
        #  2. valid data pt range [lb, ub]
        # it is meant to be used within `self.run` and `self._suggest`
        self._proj = None
        # a handy flag for _suggest method to denote whether the provided xs is a
        # new batch (i.e. the first iteration within `self.run`)
        self.is_new_batch = False

    def summary(self):
        """
        returns a summary of the attack results (to be tabulated)
        :return:
        """
        self.total_loss_queries = int(self.total_loss_queries)
        self.total_crit_queries = int(self.total_crit_queries)
        self.total_successes = int(self.total_successes)
        self.total_failures = int(self.total_failures)
        return {
            "total_loss_queries": self.total_loss_queries,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "average_num_queries": "NaN" if self.total_successes == 0 else (self.total_loss_queries + self.total_crit_queries)/ self.total_successes,
            "average_num_loss_queries": "NaN" if self.total_successes == 0 else self.total_loss_queries / self.total_successes,
            "failure_rate": self.total_failures / (self.total_successes + self.total_failures),
            "total_crit_queries": self.total_crit_queries,
            "total_queries": self.total_crit_queries + self.total_loss_queries,
            "average_num_crit_queries": "NaN" if self.total_successes == 0 else self.total_crit_queries / self.total_successes,
            "config": self.config()
        }

    def config(self):
        """
        return the attack's parameter configurations as a dict
        :return:
        """
        raise NotImplementedError

    def _suggest(self, xs_t, loss_fct, img_meats, clean_info):
        """
        :param xs_t: batch_size x dim x .. (torch tensor)
        :param loss_fct: function to query (the attacker would like to maximize) (batch_size data pts -> R^{batch_size}
        :param img_metas: image information for detection

        :return: suggested xs as a (torch tensor)and the used number of queries per data point
            i.e. a tuple of (batch_size x dim x .. tensor, batch_size array of number queries used)
        Note
        1. this method is assumed to be used only within run, one implication is that
        `self._proj` is redefined every time `self.run` is called which might be used by `self._suggest`
        """
        raise NotImplementedError

    def proj_replace(self, xs_t, sugg_xs_t, dones_mask_t):
        sugg_xs_t = self._proj(sugg_xs_t)
        # replace xs only if not done
        xs_t = sugg_xs_t * (1. - dones_mask_t) + xs_t * dones_mask_t
        return xs_t

    def run(self, data, loss_fct, early_stop_crit_fct, vis_attack_step=None, log=None, results_records=None, quires_records=None, gt_info=None):
        """
        attack with `xs` as data points using the oracle `l` and
        the early stopping criterion `early_stop_crit_fct`
        :param xs: data points to be perturbed adversarially (numpy array)
        :param loss_fct: loss function (m data pts -> R^m)
        :param early_stop_crit_fct: early stop function (m data pts -> {0,1}^m)
                ith entry is 1 if the ith data point is misclassified

        :return: a dict of logs whose length is the number of iterations
        """
        if vis_attack_step is not None:
            vis_result = []
        if self.name == 'IoUzosignsgd':
            results_records_iter_list = demo_utils.results_records_iter_list_for_zosignsgd
        elif self.name == 'IoUsquare':
            results_records_iter_list = demo_utils.results_records_iter_list_for_square
        elif self.name == 'IoUss':
            results_records_iter_list = demo_utils.results_records_iter_list_for_square
        elif self.name == 'IoUssGT':
            results_records_iter_list = demo_utils.results_records_iter_list_for_square
        elif self.name == 'IoUsign':
            results_records_iter_list = demo_utils.results_records_iter_list_for_signhunter
        elif self.name == 'IoUnes':
            results_records_iter_list = demo_utils.results_records_iter_list_for_nes
        # get objects list
        with torch.no_grad():
            result = self.attack_model(return_loss=False, rescale=True, attack_mode=self.attack_mode, **data)
        if 'IoU' in self.name and 'GT' not in self.name:
            bboxes_clean, bbox_scores_clean, labels_clean = demo_utils.get_bboxes_scores_and_labels(result, ncls=80)
        elif 'GT' in self.name and gt_info is not None:
            bboxes_clean, bbox_scores_clean, labels_clean, labels_dic = gt_info
            temp = np.expand_dims(np.max(bbox_scores_clean, axis=1), 1)
            gt_bboxes_and_scores = np.concatenate((bboxes_clean, temp), axis=1) 
            gt_bboxes_and_scores_tor = torch.from_numpy(gt_bboxes_and_scores).unsqueeze(0)
            labels_clean_tor = torch.from_numpy(labels_clean).unsqueeze(0)      
            gt_bbox_results = [
                bbox2result(gt_bboxes_and_scores_tor[i], labels_clean_tor[i],
                            80)
                for i in range(len(gt_bboxes_and_scores_tor))
            ]

            if vis_attack_step is not None:
                vis_result.append(gt_bbox_results)
                if results_records is not None:
                    results_records[0].append(gt_bbox_results[0])                
        else:
            _, labels_clean = demo_utils.get_scores_and_labels(result, ncls=80)

        objects_clean = np.unique(labels_clean)

        # get img information
        img_metas = data['img_metas']

        # visual attack step setting
        if vis_attack_step is not None and gt_info is None:
            with torch.no_grad():
                clean_result = self.attack_model(return_loss=False, rescale=True, **data)
            vis_result.append(clean_result)
            if results_records is not None:
                results_records[results_records_iter_list.index(0)].append(clean_result[0])
                
        # convert to tensor
        xs = data['img'][0].numpy().transpose(0, 2, 3, 1)
        xs_t = t(xs)
        xs_to = ch.zeros_like(xs_t)
        # feature_oo, _ = loss_fct(xs, com = True)
        # feature_o = ch.zeros_like(feature_oo)
        
        batch_size = xs.shape[0]
        num_axes = len(xs.shape[1:])
        num_loss_queries = np.zeros(batch_size)
        if quires_records is not None:
            quires_records[results_records_iter_list.index(0)].append(np.array(0))
        num_crit_queries = np.zeros(batch_size)
        
        if 'IoU' in self.name and 'GT' not in self.name:
            clean_info = [bboxes_clean, bbox_scores_clean, labels_clean, objects_clean]
        elif 'GT' in self.name and gt_info is not None:
            clean_info = [bboxes_clean, bbox_scores_clean, labels_clean, objects_clean, labels_dic]
        else:
            clean_info = [objects_clean]
        
        dones_mask = early_stop_crit_fct(self, xs, img_metas, clean_info)
        correct_classified_mask = np.logical_not(dones_mask)
        records_list_index = 0

    
        # list of logs to be returned
        logs_dict = {
            # 'total_loss': [],
            'total_cos_sim': [],
            'total_ham_sim': [],
            'total_successes': [],
            'total_failures': [],
            'iteration': [],
            'total_loss_queries': [],
            'total_crit_queries': [],
            'total_queries': [],
            'num_loss_queries_per_iteration': [],
            'num_crit_queries_per_iteration': []
        }

        # ignore this batch of xs if all are misclassified
        if sum(correct_classified_mask) == 0:
            if results_records is not None:
                return num_loss_queries, vis_result, results_records, quires_records
            if vis_attack_step is not None and vis_attack_step:
                return num_loss_queries, vis_result
            
            return num_loss_queries

        # init losses and cosine similarity for performance tracking
        # losses = np.zeros(batch_size)
        cos_sims = np.zeros(batch_size)
        ham_sims = np.zeros(batch_size)
        # its = 0
        # make a projector into xs lp-ball and within valid pixel range
        if self.p == '2':
            _proj = l2_proj_maker(xs_t, self.epsilon)
            self._proj = lambda _: ch.clamp(_proj(_), self.lb, self.ub)
        elif self.p == 'inf':
            _proj = linf_proj_maker(xs_t, self.epsilon)
            self._proj = lambda _: ch.clamp(_proj(_), self.lb, self.ub)
        else:
            raise Exception('Undefined l-p!')
        
        # iterate till model evasion or budget exhaustion
        # to inform self._suggest this is  a new batch
        self.is_new_batch = True
        while True:
            if self.name == "IoUnes" or self.name == "IoUzosignsgd" or self.name == "bandit":
                # if np.any(num_loss_queries + num_crit_queries >= self.max_loss_queries):
                if np.any(num_loss_queries >= self.max_loss_queries):
                    if log is not None:
                        log.logger.info("#loss queries exceeded budget, exiting")
                    else:
                        print("#loss queries exceeded budget, exiting")
                    break
            else:
                if np.any(num_loss_queries >= self.max_loss_queries):
                    if log is not None:
                        log.logger.info("#loss queries exceeded budget, exiting")
                    else:
                        print("#loss queries exceeded budget, exiting")
                    break

            if np.any(num_crit_queries >= self.max_crit_queries):
                if log is not None:
                    log.logger.info("#crit_queries exceeded budget, exiting")
                else:
                    print("#crit_queries exceeded budget, exiting")
                break
            if np.all(dones_mask):
                if log is not None:
                    log.logger.info("all data pts are misclassified, exiting")
                else:
                    print("all data pts are misclassified, exiting")
                
                if results_records is not None and quires_records is not None:
                    stop_index = demo_utils.get_first(results_records_iter_list, self.max_loss_queries)
                    with torch.no_grad():
                        result = self.attack_model(return_loss=False, rescale=True, **data)                
                    # for index in range(len(results_records)):
                        # if index > records_list_index and len(results_records[index]) != 0:
                    for index in range(stop_index):
                        if index > records_list_index:
                            results_records[index].append(result[0])
                            quires_records[index].append(num_loss_queries)
                break
            # propose new perturbations
            # if self.patch_attack is not None and self.name == 'IoUsquare':
            #     self.                    
            sugg_xs_t, num_loss_queries_per_step = self._suggest(xs_t, loss_fct, img_metas, clean_info)
            # project around xs and within pixel range and
            # replace xs only if not done

            ##updated x here
            dones = np.all(dones_mask)
            xs_t = self.proj_replace(xs_t, sugg_xs_t, t(dones.reshape(-1, *[1] * num_axes).astype(np.float32)))

            # feature, _ = loss_fct(xs_t.cpu().numpy(), com = True)
            # inner = ch.mm(feature_oo,feature.transpose(0,1))
            # norm1 = ch.norm(feature_oo, dim = 1,keepdim = True)      
            # norm2 = ch.norm(feature, dim = 1,keepdim = True)
            # cos = inner/(ch.mm(norm1,norm2.transpose(0,1)))
            # inner_query = ch.mm(feature_o,feature.transpose(0,1))
            # norm3 = ch.norm(feature_o, dim = 1,keepdim = True)
            # #norm4 = ch.norm(feature, dim = 1,keepdim = True)
            # cos_query = inner_query/(ch.mm(norm3,norm2.transpose(0,1)))

            
            # print(np.linalg.norm(xs_t.view(xs_t.shape[0], -1).cpu(), axis=1))
            # update number of queries (note this is done before updating dones_mask)
            # num_loss_queries += num_loss_queries_per_step * (1. - dones_mask)
            num_loss_queries += num_loss_queries_per_step
            # update total loss and total cos_sim (these two values are used for performance monitoring)
            # cos_sims = cur_cos_sims * (1. - dones_mask) + cos_sims * dones_mask
            # ham_sims = cur_ham_sims * (1. - dones_mask) + ham_sims * dones_mask
            # losses = loss_fct(xs_t.cpu().numpy()) * (1. - dones_mask) + losses * dones_mask
            # update dones mask
            # dones_mask = np.logical_or(dones_mask, early_stop_crit_fct(xs_t.cpu().numpy()))
            dones_mask = early_stop_crit_fct(self, xs_t.cpu().numpy(), img_metas, clean_info)
            # its += 1
            self.is_new_batch = False

            # success_mask = dones_mask * correct_classified_mask
            
            # import pdb; pdb.set_trace()
            #pdb.set_trace()
            # feature_o = feature
            xs_to = xs_t
            if len(img_metas) != 1:
                data['img'][0] = torch.FloatTensor(xs_t.cpu().numpy().transpose(0,3,1,2))
                data['img'][1] = torch.FloatTensor(xs_t.cpu().numpy().transpose(0,3,1,2))
            else:
                data['img'][0] = torch.FloatTensor(xs_t.cpu().numpy().transpose(0,3,1,2))
            
            # visualize opt step
            if vis_attack_step is not None and vis_attack_step:
                if self.name == 'sign':
                    vis_start = 4
                else:
                    vis_start = 1

                vis_max = int(self.max_loss_queries)
                vis_step = int((vis_max-vis_start) / 4)
                if vis_step != 0:
                    for index in range(vis_start, vis_max, vis_step):
                        if num_loss_queries[0] == index:
                            with torch.no_grad():
                                result = self.attack_model(return_loss=False, rescale=True, **data)
                            vis_result.append(result)

            if results_records is not None and quires_records is not None:
                records_num = int(num_loss_queries)
                if records_num in results_records_iter_list:
                    records_list_index = results_records_iter_list.index(records_num)
                    with torch.no_grad():
                        result = self.attack_model(return_loss=False, rescale=True, **data)                
                    results_records[records_list_index].append(result[0])
                    quires_records[records_list_index].append(np.array(records_num))
                # else: 
                #     if np.all(dones_mask):
                #         with torch.no_grad():
                #             result = self.attack_model(return_loss=False, rescale=True, **data)                
                #         for index in range(records_list_index, len(results_records)):
                #             results_records[index].append(result[0])
                    
            
            # update logs
            # logs_dict['total_loss'].append(sum(losses))
            # logs_dict['total_cos_sim'].append(sum(cos_sims))
            # logs_dict['total_ham_sim'].append(sum(ham_sims))
            # logs_dict['total_successes'].append(sum(dones_mask * correct_classified_mask))
            # logs_dict['total_failures'].append(sum(np.logical_not(dones_mask) * correct_classified_mask))
            # logs_dict['iteration'].append(its)
            # # assuming all data pts consume the same number of queries per step
            # logs_dict['num_loss_queries_per_iteration'].append(num_loss_queries_per_step[0])
            # logs_dict['num_crit_queries_per_iteration'].append(1)
            # logs_dict['total_loss_queries'].append(sum(num_loss_queries * dones_mask * correct_classified_mask))
            # logs_dict['total_crit_queries'].append(sum(num_crit_queries * dones_mask * correct_classified_mask))
            # logs_dict['total_queries'].append(sum(num_crit_queries * dones_mask * correct_classified_mask) + sum(num_loss_queries * dones_mask * correct_classified_mask))


            # if its % 1 == 0:
            #     success_mask = dones_mask * correct_classified_mask
            #     total_successes = float(success_mask.sum())
            #     # import pdb; pdb.set_trace()
            #     print ("Iteration : ", its, 'ave_loss_queries : ', (num_loss_queries * success_mask).sum() / total_successes, \
            #         "ave_crit_queries : ", (num_crit_queries * success_mask).sum()  / total_successes, \
            #         "ave_queries : ", (num_loss_queries * success_mask).sum() / total_successes + (num_crit_queries * success_mask).sum()  / total_successes, \
            #         "successes : ", success_mask.sum() / float(success_mask.shape[0]), \
            #         "failures : ", (np.logical_not(dones_mask) * correct_classified_mask).sum()  / float(success_mask.shape[0]))
            #     sys.stdout.flush()

        # success_mask = dones_mask * correct_classified_mask
        # self.total_loss_queries += (num_loss_queries * success_mask).sum()
        # self.total_crit_queries += (num_crit_queries * success_mask).sum()
        # self.total_successes += success_mask.sum()
        # self.total_failures += (np.logical_not(dones_mask) * correct_classified_mask).sum()

        # set self._proj to None to ensure it is intended use
        self._proj = None
        
        if results_records is not None:
            return num_loss_queries, vis_result, results_records, quires_records
        if vis_attack_step is not None and vis_attack_step:
            return num_loss_queries, vis_result

        return num_loss_queries

