#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   iou_ss_attack.py
@Time    :   2021/02/26 11:05:10
@Author  :   Siyuan Liang
'''

# here put the import lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import Tensor as t

from blackbox_attack.black_box_attack import BlackBoxAttack
from blackbox_attack.utils.compute_fcts import lp_step
from attack_demo.util import bbox_to_attack_points, mask_to_attack_points, unique_rows, reppoints_to_attack_points

class IoUSSAttack(BlackBoxAttack):
    def __init__(self, max_loss_queries, epsilon, p, p_init, lb, ub, name, attack_model, attack_mode, attack_logistics, loss, targeted, ori_img, model_name, zeta, lambda1, patch_attack, keypoints_models, square_expansion, attack_parallel, square_init):
        
        super().__init__(
            max_loss_queries=max_loss_queries,
            epsilon=epsilon,
            p=p,
            lb=lb,
            ub=ub,
            name=name,
            attack_model=attack_model,
            attack_mode=attack_mode,
            loss=loss,
            targeted=targeted,
            ori_img=ori_img,
            model_name=model_name,
            zeta=zeta,
            lambda1=lambda1,
            patch_attack=patch_attack,
            keypoints_models=keypoints_models
            # square_expansion=square_expansion,
            # square_init=square_init,
            # attack_parallel=attack_parallel
        )

        self.best_loss = None
        self.i = 0
        self.p_init = p_init
        self.attack_parallel = attack_parallel
        self.square_expansion = square_expansion
        self.square_init = square_init
        self.flip_flag = None
        self.flip_center_hs = None
        self.flip_center_ws = None
        self.flip_deltas = None 
        self.flip_zone = None
        self.p_change = None
        if self.patch_attack is not None:
            self.attack_points = None
        
    # def attack_points_parallel(x, x_best, center_h, center_w, s):

    #     x_window = x[i_img, :, center_h:center_h+s, center_w:center_w+s]
    #     x_best_window = x_best[i_img, :, center_h:center_h+s, center_w:center_w+s]

    #     # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
    #     if not self.flip_flag:
    #         while np.sum(np.abs(np.clip(x_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], self.lb, self.ub) - x_best_window) < 10**-7) == c*s*s:
    #             deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = np.random.choice([-self.epsilon, self.epsilon], size=[c, 1, 1])
    #             self.flip_deltas = deltas
    #             self.flip_flag = True
    #             self.flip_zone = 'w1'
    #     else:
    #         # flip twice on w
    #         if self.flip_zone == 'w1':
    #             flip_deltas_sign = np.ones(size=[1, c, s, s], dtype='float32')
    #             flip_deltas_sign[:, :, :, :flip_w_size] = flip_deltas_sign[:, :, :, :flip_w_size] * -1.0 
    #             deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = self.flip_deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] * flip_deltas_sign
    #             self.flip_zone = 'w2'
    #         elif self.flip_zone == 'w2':
    #             flip_deltas_sign = np.ones(self.flip_deltas.shape, dtype='float32')
    #             flip_deltas_sign[:, :, :flip_w_size:] = flip_deltas_sign[:, :, :flip_w_size] * -1.0
    #             deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = self.flip_deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] * flip_deltas_sign
    #             self.flip_zone = 'h1'
    #             # self.flip_flag = False
    #         elif self.flip_zone == 'h1':
    #             flip_deltas_sign = np.ones(self.)

    def attack_points_selection(self, patch_attacks, it, h, w, clean_info, img_metas, s, ori_img):
        # print('it is {}'.format(it))
        attack_points = np.empty(shape=[0, 2])
        if it in [0, 11, 51, 201, 501, 1001, 2001, 4001, 6001, 8001, 10001]:
            for patch_attack in patch_attacks.split(','):
                if patch_attack == 'bbox':
                    attack_points_bbox = bbox_to_attack_points(h, w, clean_info, img_metas, s)
                    attack_points = np.append(attack_points, attack_points_bbox, axis=0)
                elif patch_attack == 'maskrcnn':
                    data = {}
                    data['img'] = [ori_img]
                    data['img_metas'] = img_metas
                    with torch.no_grad():
                        results = self.keypoints_models[patch_attack](return_loss=False, rescale=True, **data)
                    seg_masks = results[0][1]
                    attack_points_mask = mask_to_attack_points(clean_info, seg_masks, img_metas, s=s)
                    attack_points = np.append(attack_points, attack_points_mask, axis=0)
                elif patch_attack == 'reppoints':
                    data = {}
                    data['img'] = [ori_img]
                    data['img_metas'] = img_metas
                    with torch.no_grad():
                        results = self.keypoints_models[patch_attack](return_loss=False, rescale=True, **data, get_points=True)
                    attack_points_rep = reppoints_to_attack_points(results, h=h, w=w, s=s)
                    attack_points = np.append(attack_points, attack_points_rep, axis=0)
                elif patch_attack == 'proposal':
                    data = {}
                    data['img'] = [ori_img]
                    data['img_metas'] = img_metas
                    with torch.no_grad():
                        results = self.keypoints_models[patch_attack](return_loss=False, rescale=True, **data, get_proposal=True)
                    proposals = results[0].cpu().detach().numpy()[:, :4]
                    attack_points_bbox = bbox_to_attack_points(h, w, proposals, img_metas, s, get_proposal=True)
                    attack_points = np.append(attack_points, attack_points_bbox, axis=0)  
            attack_points_unique = unique_rows(attack_points)
            self.attack_points = np.array(attack_points_unique, dtype='int64')
        return self.attack_points
    

    def attack_parallel_selection(self, parallel_num_init, it, n_iters):
        # it = int(it / n_iters * 10000)
        # parallel_num_init = int(parallel_num_init)
        # if 20 < it <= 100:
        #     parallel_num = parallel_num_init * 2
        # elif 100 < it <= 400:
        #     parallel_num = parallel_num_init * 4
        # elif 400 < it <= 1000:
        #     parallel_num = parallel_num_init * 8
        # elif 1000 < it <= 2000:
        #     parallel_num = parallel_num_init * 16
        # elif 2000 < it <= 4000:
        #     parallel_num = parallel_num_init * 32
        # elif 4000 < it <= 8000:
        #     parallel_num = parallel_num_init * 64
        # else:
        #     parallel_num = parallel_num_init
        
        # reverse
        if it <= 20:
            parallel_num = parallel_num_init * 4
        if 20 < it <= 100:
            parallel_num = parallel_num_init * 4
        elif 100 < it <= 400:
            parallel_num = parallel_num_init * 2
        elif 400 < it <= 1000:
            parallel_num = parallel_num_init * 2
        elif 1000 < it <= 2000:
            parallel_num = parallel_num_init * 1
        elif 2000 < it <= 4000:
            parallel_num = parallel_num_init * 1
        elif 4000 < it <= 8000:
            parallel_num = parallel_num_init * 1
        else:
            parallel_num = parallel_num_init

        if it in [0, 21, 101, 401, 1001, 2001, 4001]:
            flag = True
        else:
            flag = False

        return parallel_num, flag

    def p_selection(self, p_init, it, n_iters):
        """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
        it = int(it / n_iters * 10000)

        # if 10 < it <= 50:
        #     p = p_init / 2
        # elif 50 < it <= 200:
        #     p = p_init / 4
        # elif 200 < it <= 500:
        #     p = p_init / 8
        # elif 500 < it <= 1000:
        #     p = p_init / 16
        # elif 1000 < it <= 2000:
        #     p = p_init / 32
        # elif 2000 < it <= 4000:
        #     p = p_init / 64
        # elif 4000 < it <= 6000:
        #     p = p_init / 128
        # elif 6000 < it <= 8000:
        #     p = p_init / 256
        # elif 8000 < it <= 10000:
        #     p = p_init / 512
        # else:
        #     p = p_init

        if 20 < it <= 100:
            p = p_init / 2
        elif 100 < it <= 400:
            p = p_init / 4
        elif 400 < it <= 1000:
            p = p_init / 8
        elif 1000 < it <= 2000:
            p = p_init / 16
        elif 2000 < it <= 4000:
            p = p_init / 32
        elif 4000 < it <= 8000:
            p = p_init / 64
        else:
            p = p_init / 128

        return p


    def _suggest(self, xs_t, loss_fct, img_metas, clean_info):
        xs = xs_t.cpu().numpy().transpose(0,3,1,2)
        c, h, w = xs.shape[1:]
        n_features = c*h*w
        n_queries = np.zeros(xs.shape[0])

        p = self.p_selection(self.p_init, self.i, 10000)
        if self.p == 'inf':
            if self.is_new_batch:
                self.p_change = p
                self.x = xs.copy()
                if self.square_init is None or self.square_init == 'False':
                    init_delta = np.random.choice([-self.epsilon, self.epsilon], size=[xs.shape[0], c, 1, w])
                    xs = xs + init_delta
                else:
                    s = int(round(np.sqrt(p * n_features / c)))
                    s = min(max(s, 1), h-1)
                    if self.patch_attack is not None:
                        attack_points = self.attack_points_selection(self.patch_attack, self.i, h, w, clean_info, img_metas, s=s, ori_img=self.ori_img)
                    else:
                        attack_points = []
                     
                    for _ in range(0, int(self.attack_parallel)):
                        center_h_init = None,
                        center_w_init = None
                        if len(attack_points) > 10000:
                            center_h_init, center_w_init = attack_points[np.random.randint(0, len(attack_points))]
                        else:
                            center_h_init = np.random.randint(0, h - s)
                            center_w_init = np.random.randint(0, w - s)                                
                        init_delta = np.random.choice([-self.epsilon, self.epsilon], size=[xs.shape[0], c, 1, s])  
                        xs[xs.shape[0]-1, :, center_h_init:center_h_init+s, center_w_init:center_w_init+s] = xs[xs.shape[0]-1, :, center_h_init:center_h_init+s, center_w_init:center_w_init+s] + init_delta
                xs = np.clip(xs, self.lb, self.ub)
                self.best_loss = loss_fct(self, xs.transpose(0,2,3,1), img_metas, clean_info)
                n_queries += np.ones(xs.shape[0])
                self.i = 0
                self.flip_flag = False
            
            deltas = xs - self.x
            p = self.p_selection(self.p_init, self.i, 10000)
            if self.p_change != p:
                self.flip_flag = False
                self.p_change = p
            for i_img in range(xs.shape[0]):
                s = int(round(np.sqrt(p * n_features / c)))
                s = min(max(s, 1), h-1)

                if self.patch_attack is not None:
                    attack_points = self.attack_points_selection(self.patch_attack, self.i, h, w, clean_info, img_metas, s=s, ori_img=self.ori_img)
                else:
                    attack_points = []

                center_hs = []
                center_ws = []
                
                # attack_parallel_num, parallel_init_flag = self.attack_parallel_selection(self.attack_parallel, self.i, 10000)
                # if parallel_init_flag or not self.flip_flag:
                #     for _ in range(0, int(attack_parallel_num)):    
                #     # for _ in range(0, int(self.attack_parallel)):    
                #         if len(attack_points) > 10000:                        
                #             center_h, center_w = attack_points[np.random.randint(0, len(attack_points))]
                #         else:
                #             center_h = np.random.randint(0, h - s)
                #             center_w = np.random.randint(0, w - s)
        
                #     center_hs.append(center_h)
                #     center_ws.append(center_w)
                
                # if parallel_init_flag and int(attack_parallel_num) != 1:
                #     self.flip_flag = False

                # attack one point
                attack_parallel_num = self.attack_parallel
                for _ in range(0, int(attack_parallel_num)):    
                    if len(attack_points) > 10000:                        
                        center_h, center_w = attack_points[np.random.randint(0, len(attack_points))]
                    else:
                        center_h = np.random.randint(0, h - s)
                        center_w = np.random.randint(0, w - s)

                if not self.flip_flag:
                    self.flip_center_hs = np.array(center_hs)
                    self.flip_center_ws = np.array(center_ws)
                else:
                    center_hs = self.flip_center_hs
                    center_ws = self.flip_center_ws
                
                # attack_points_parallel
                for count, (center_h, center_w) in enumerate(zip(center_hs, center_ws)):
                    if center_h > h-s or center_w > w -s:
                        continue
                    x_window = self.x[i_img, :, center_h:center_h+s, center_w:center_w+s]
                    x_best_window = xs[i_img, :, center_h:center_h+s, center_w:center_w+s]

                    # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
                    if not self.flip_flag:
                        while np.sum(np.abs(np.clip(x_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], self.lb, self.ub) - x_best_window) < 10**-7) == c*s*s:
                            deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = np.random.choice([-self.epsilon, self.epsilon], size=[c, 1, 1])
                            self.flip_deltas = deltas
                            if count+1 == attack_parallel_num:
                                self.flip_flag = True
                                # self.flip_zone = 'w1'
                                self.flip_zone = 'h1'
                    else:
                        flip_size = int(s/2)
                        # flip twice on w
                        # if self.flip_zone == 'w1':
                        #     flip_deltas_sign = np.ones((1, c, s, s), dtype='float32')
                        #     flip_deltas_sign[:, :, :, :flip_size] = flip_deltas_sign[:, :, :, :flip_size] * -1.0 
                        #     deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = self.flip_deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] * flip_deltas_sign
                        #     self.flip_zone = 'w2'
                        # elif self.flip_zone == 'w2':
                        #     flip_deltas_sign = np.ones((1, c, s, s), dtype='float32')
                        #     flip_deltas_sign[:, :, :, flip_size:] = flip_deltas_sign[:, :, :, flip_size:] * -1.0
                        #     deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = self.flip_deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] * flip_deltas_sign
                        #     self.flip_flag = False
                        #     self.flip_zone = 'h1'
                        if self.flip_zone == 'h1':
                            flip_deltas_sign = np.ones((1, c, s, s), dtype='float32')
                            flip_deltas_sign[:, :, flip_size:, :] = flip_deltas_sign[:, :, flip_size:, :] * -1.0
                            deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = self.flip_deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] * flip_deltas_sign
                            if count+1 == attack_parallel_num:
                                self.flip_zone = 'h2'
                        elif self.flip_zone == 'h2':
                            flip_deltas_sign = np.ones((1, c, s, s), dtype='float32')
                            flip_deltas_sign[:, :, :flip_size, :] = flip_deltas_sign[:, :, :flip_size, :] * -1.0
                            deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = self.flip_deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] * flip_deltas_sign
                            if count+1 == attack_parallel_num:
                                self.flip_flag = False
                            
                # x_window = self.x[i_img, :, center_h:center_h+s, center_w:center_w+s]
                # x_best_window = xs[i_img, :, center_h:center_h+s, center_w:center_w+s]

                # # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
                # if not self.flip_flag:
                #     while np.sum(np.abs(np.clip(x_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], self.lb, self.ub) - x_best_window) < 10**-7) == c*s*s:
                #         deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = np.random.choice([-self.epsilon, self.epsilon], size=[c, 1, 1])
                #         self.flip_deltas = deltas[i_img, :, center_h:center_h+s, center_w:center_w+s]
                #         self.flip_flag = True
                #         self.flip_zone = 'w1'
                # else:
                #     # flip twice on w
                #     flip_w_size = int(self.flip_deltas.shape[2] / 2)
                #     if self.flip_zone == 'w1':
                #         flip_deltas_sign = np.ones(self.flip_deltas.shape, dtype='float32')
                #         flip_deltas_sign[:, :, :flip_w_size] = flip_deltas_sign[:, :, :flip_w_size] * -1.0 
                #         deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = self.flip_deltas * flip_deltas_sign
                #         self.flip_zone = 'w2'
                #     elif self.flip_zone == 'w2':
                #         flip_deltas_sign = np.ones(self.flip_deltas.shape, dtype='float32')
                #         flip_deltas_sign[:, :, flip_w_size:] = flip_deltas_sign[:, :, flip_w_size:] * -1.0
                #         deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = self.flip_deltas * flip_deltas_sign
                #         self.flip_flag = False

            x_new = np.clip(self.x + deltas, self.lb, self.ub).transpose(0,2,3,1)

            new_loss = loss_fct(self, x_new, img_metas, clean_info)
            n_queries += np.ones(xs.shape[0])
            idx_improved = new_loss > self.best_loss
            self.best_loss = idx_improved * new_loss + ~idx_improved * self.best_loss
            xs = xs.transpose(0,2,3,1)
            idx_improved = np.reshape(idx_improved, [-1, *[1]*len(x_new.shape[:-1])])
            x_new = idx_improved * x_new + ~idx_improved * xs
            self.i += 1

            return t(x_new), n_queries

