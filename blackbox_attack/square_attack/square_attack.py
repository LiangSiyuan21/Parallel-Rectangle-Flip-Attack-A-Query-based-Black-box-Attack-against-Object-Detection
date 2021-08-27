import argparse
import time
import numpy as np
import data
import os
import utils
import torch
import mmcv
from datetime import datetime
from mmdet.core import tensor2imgs
import attack_demo.util as demo_utils
import blackbox_attack.square_attack.utils as sq_utils
np.set_printoptions(precision=5, suppress=True)

# import pydevd
# pydevd.settrace("192.168.40.146",port=8877)


def p_selection(p_init, it, n_iters):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init

    return p


def pseudo_gaussian_pert_rectangles(x, y):
    delta = np.zeros([x, y])
    x_c, y_c = x // 2 + 1, y // 2 + 1

    counter2 = [x_c - 1, y_c - 1]
    for counter in range(0, max(x_c, y_c)):
        delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
              max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2

        counter2[0] -= 1
        counter2[1] -= 1

    delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return delta


def meta_pseudo_gaussian_pert(s):
    delta = np.zeros([s, s])
    n_subsquares = 2
    if n_subsquares == 2:
        delta[:s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s)
        delta[s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
        if np.random.rand(1) > 0.5: delta = np.transpose(delta)

    elif n_subsquares == 4:
        delta[:s // 2, :s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, :s // 2] = pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
        delta[:s // 2, s // 2:] = pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return delta


def square_attack_l2(model, x, y, corr_classified, eps, n_iters, p_init, metrics_path, targeted, loss_type):
    """ The L2 square attack """
    np.random.seed(0)

    min_val, max_val = 0, 1
    c, h, w = x.shape[1:]
    n_features = c * h * w
    n_ex_total = x.shape[0]
    x, y = x[corr_classified], y[corr_classified]

    ### initialization
    delta_init = np.zeros(x.shape)
    s = h // 5
    log.print('Initial square side={} for bumps'.format(s))
    sp_init = (h - s * 5) // 2
    center_h = sp_init + 0
    for counter in range(h // s):
        center_w = sp_init + 0
        for counter2 in range(w // s):
            delta_init[:, :, center_h:center_h + s, center_w:center_w + s] += meta_pseudo_gaussian_pert(s).reshape(
                [1, 1, s, s]) * np.random.choice([-1, 1], size=[x.shape[0], c, 1, 1])
            center_w += s
        center_h += s

    x_best = np.clip(x + delta_init / np.sqrt(np.sum(delta_init ** 2, axis=(1, 2, 3), keepdims=True)) * eps, 0, 1)

    logits = model.predict(x_best)
    loss_min = model.loss(y, logits, targeted, loss_type=loss_type)
    margin_min = model.loss(y, logits, targeted, loss_type='margin_loss')
    n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query

    time_start = time.time()
    s_init = int(np.sqrt(p_init * n_features / c))
    metrics = np.zeros([n_iters, 7])
    for i_iter in range(n_iters):
        idx_to_fool = (margin_min > 0.0)

        x_curr, x_best_curr = x[idx_to_fool], x_best[idx_to_fool]
        y_curr, margin_min_curr = y[idx_to_fool], margin_min[idx_to_fool]
        loss_min_curr = loss_min[idx_to_fool]
        delta_curr = x_best_curr - x_curr

        p = p_selection(p_init, i_iter, n_iters)
        s = max(int(round(np.sqrt(p * n_features / c))), 3)

        if s % 2 == 0:
            s += 1

        s2 = s + 0
        ### window_1
        center_h = np.random.randint(0, h - s)
        center_w = np.random.randint(0, w - s)
        new_deltas_mask = np.zeros(x_curr.shape)
        new_deltas_mask[:, :, center_h:center_h + s, center_w:center_w + s] = 1.0

        ### window_2
        center_h_2 = np.random.randint(0, h - s2)
        center_w_2 = np.random.randint(0, w - s2)
        new_deltas_mask_2 = np.zeros(x_curr.shape)
        new_deltas_mask_2[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 1.0
        norms_window_2 = np.sqrt(
            np.sum(delta_curr[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] ** 2, axis=(-2, -1),
                   keepdims=True))

        ### compute total norm available
        curr_norms_window = np.sqrt(
            np.sum(((x_best_curr - x_curr) * new_deltas_mask) ** 2, axis=(2, 3), keepdims=True))
        curr_norms_image = np.sqrt(np.sum((x_best_curr - x_curr) ** 2, axis=(1, 2, 3), keepdims=True))
        mask_2 = np.maximum(new_deltas_mask, new_deltas_mask_2)
        norms_windows = np.sqrt(np.sum((delta_curr * mask_2) ** 2, axis=(2, 3), keepdims=True))

        ### create the updates
        new_deltas = np.ones([x_curr.shape[0], c, s, s])
        new_deltas = new_deltas * meta_pseudo_gaussian_pert(s).reshape([1, 1, s, s])
        new_deltas *= np.random.choice([-1, 1], size=[x_curr.shape[0], c, 1, 1])
        old_deltas = delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] / (1e-10 + curr_norms_window)
        new_deltas += old_deltas
        new_deltas = new_deltas / np.sqrt(np.sum(new_deltas ** 2, axis=(2, 3), keepdims=True)) * (
            np.maximum(eps ** 2 - curr_norms_image ** 2, 0) / c + norms_windows ** 2) ** 0.5
        delta_curr[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 0.0  # set window_2 to 0
        delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] = new_deltas + 0  # update window_1

        hps_str = 's={}->{}'.format(s_init, s)
        x_new = x_curr + delta_curr / np.sqrt(np.sum(delta_curr ** 2, axis=(1, 2, 3), keepdims=True)) * eps
        x_new = np.clip(x_new, min_val, max_val)
        curr_norms_image = np.sqrt(np.sum((x_new - x_curr) ** 2, axis=(1, 2, 3), keepdims=True))

        logits = model.predict(x_new)
        loss = model.loss(y_curr, logits, targeted, loss_type=loss_type)
        margin = model.loss(y_curr, logits, targeted, loss_type='margin_loss')

        idx_improved = loss < loss_min_curr
        loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
        margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr

        idx_improved = np.reshape(idx_improved, [-1, *[1] * len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1

        acc = (margin_min > 0.0).sum() / n_ex_total
        acc_corr = (margin_min > 0.0).mean()
        mean_nq, mean_nq_ae, median_nq, median_nq_ae = np.mean(n_queries), np.mean(
            n_queries[margin_min <= 0]), np.median(n_queries), np.median(n_queries[margin_min <= 0])

        time_total = time.time() - time_start
        log.print(
            '{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.1f} med#q_ae={:.1f} {}, n_ex={}, {:.0f}s, loss={:.3f}, max_pert={:.1f}, impr={:.0f}'.
                format(i_iter + 1, acc, acc_corr, mean_nq_ae, median_nq_ae, hps_str, x.shape[0], time_total,
                       np.mean(margin_min), np.amax(curr_norms_image), np.sum(idx_improved)))
        metrics[i_iter] = [acc, acc_corr, mean_nq, mean_nq_ae, median_nq, margin_min.mean(), time_total]
        # if (i_iter <= 500 and i_iter % 500) or (i_iter > 100 and i_iter % 500) or i_iter + 1 == n_iters or acc == 0:
            # np.save(metrics_path, metrics)
        if acc == 0:
            curr_norms_image = np.sqrt(np.sum((x_best - x) ** 2, axis=(1, 2, 3), keepdims=True))
            print('Maximal norm of the perturbations: {:.5f}'.format(np.amax(curr_norms_image)))
            break

    curr_norms_image = np.sqrt(np.sum((x_best - x) ** 2, axis=(1, 2, 3), keepdims=True))
    print('Maximal norm of the perturbations: {:.5f}'.format(np.amax(curr_norms_image)))

    return n_queries, x_best


def square_attack_linf(model, img_info, scores_labels_clean, eps, n_iters, p_init, targeted, loss_type, attack_logistics=None, ncls=None, vis_attack_step=None):
    """ The Linf square attack """

    # visual attack step setting
    if vis_attack_step is not None:
        vis_result = []
        with torch.no_grad():
            vis_result.append(model(return_loss=False, rescale=True, **img_info))    

    np.random.seed(0)  # important to leave it here as well
    x = img_info['img'][0].numpy()
    # min_val, max_val = 0, 1 if x.max() <= 1 else 255
    min_val, max_val = x.min(), x.max()
    # print('min: {}, max: {}'.format(min_val, max_val))
    c, h, w = x.shape[1:]
    n_features = c*h*w
    n_ex_total = x.shape[0]
    if attack_logistics is not None and attack_logistics:
        scores_logit_clean, labels_clean = scores_labels_clean
    else:
        scores_smooth_clean, labels_clean = scores_labels_clean
    objects_clean = np.unique(labels_clean)

    if n_iters == 0:
        n_queries = np.zeros(x.shape[0])
        return n_queries 

    # [c, 1, w], i.e. vertical stripes work best for untargeted attacks
    eps_real = (max_val - min_val) * eps
    init_delta = np.random.choice([-eps_real, eps_real], size=[x.shape[0], c, 1, w]).astype('float32')
    x_best = np.clip(x + init_delta, min_val, max_val)
    img_info['img'][0] = torch.from_numpy(x_best.astype('float32'))
    # return torch.from_numpy(x_best.astype('float32'))

    # visual attack step setting
    if vis_attack_step is not None:
        with torch.no_grad():
            vis_result.append(model(return_loss=False, rescale=True, **img_info)) 


    # logits = model.predict(img_best)
    fool_flag = False
    if attack_logistics is not None and attack_logistics:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, attack_mode=True, attack_logistics=attack_logistics, **img_info)
        _, labels_result = demo_utils.get_scores_and_labels(result, ncls=ncls)
        scores_logit_result = result[2][0].cpu().detach().numpy()
        scores_result = scores_logit_result
    else:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, attack_mode=True, **img_info)
        scores_smooth_result, labels_result = demo_utils.get_scores_and_labels(result, ncls=ncls)
        scores_result = scores_smooth_result
    
    n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query
    
    if len(labels_result) != 0:      
        fool_flag = True
    
    scores_adv, labels_adv = demo_utils.filter_scores_labels(scores_result, labels_result, objects_clean)
    labels_target  = sq_utils.random_classes_except_current(labels_adv, ncls) if targeted else labels_adv
    if len(labels_target) == 0:
        return n_queries
    labels_target_onehot = sq_utils.dense_to_onehot(labels_target, n_cls=ncls)
    
    if fool_flag:
        loss_min = demo_utils.sq_loss(labels_target_onehot, scores_adv, targeted, loss_type=loss_type).sum(0)
        margin_min = demo_utils.sq_loss(labels_target_onehot, scores_adv, targeted, loss_type='margin_loss').sum(0)

        # time_start = time.time()
        # metrics = np.zeros([n_iters, 7])
        for i_iter in range(n_iters - 1):
            x_curr, x_best_curr = x, x_best
            loss_min_curr, margin_min_curr = loss_min, margin_min
            deltas = x_best_curr - x_curr

            p = p_selection(p_init, i_iter, n_iters)
            for i_img in range(x_best_curr.shape[0]):
                s = int(round(np.sqrt(p * n_features / c)))
                s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
                center_h = np.random.randint(0, h - s)
                center_w = np.random.randint(0, w - s)

                x_curr_window = x_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
                x_best_curr_window = x_best_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
                # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
                # temp = np.clip(x_curr_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], min_val, max_val) - x_best_curr_window
                # temp[:, 0:3, 0:3] = np.random.choice([-eps, eps], size=[c, 1, 1])
                while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], min_val, max_val) - x_best_curr_window) < 10**-7) == c*s*s:
                    deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = np.random.choice([-eps, eps], size=[c, 1, 1])

            x_new = np.clip(x_curr + deltas, min_val, max_val)
            img_info['img'][0] = torch.from_numpy(x_new.astype('float32'))
            
            if attack_logistics is not None and attack_logistics:
                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, attack_mode=True, attack_logistics=attack_logistics, **img_info)
                _, labels_result = demo_utils.get_scores_and_labels(result, ncls=ncls)
                scores_logit_clean = result[2][0].cpu().detach().numpy()
                scores_result = scores_logit_clean
            else:
                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, attack_mode=True, **img_info)
                scores_smooth_result, labels_result = demo_utils.get_scores_and_labels(result, ncls=ncls)
                scores_result = scores_smooth_result
            
            n_queries += 1
            
            scores_adv, labels_adv = demo_utils.filter_scores_labels(scores_result, labels_result, objects_clean)
            labels_target = sq_utils.random_classes_except_current(labels_adv, ncls) if targeted else labels_adv
            if len(labels_target) == 0:
                return n_queries
            
            labels_target_onehot = sq_utils.dense_to_onehot(labels_target, n_cls=ncls)
            
            loss = demo_utils.sq_loss(labels_target_onehot, scores_adv, targeted, loss_type=loss_type).sum(0)
            margin = demo_utils.sq_loss(labels_target_onehot, scores_adv, targeted, loss_type='margin_loss').sum(0)

            improved = loss < loss_min_curr
            if improved:
                loss_min = loss
                margin_min = margin
                x_best = x_new
            
            # img_info['img'][0] = torch.from_numpy(x_new.astype('float32'))

            if vis_attack_step is not None and vis_attack_step:
                vis_start = 2

                vis_max = int(n_iters) - 1 
                vis_step = int((vis_max-vis_start) / 2)
                if vis_step != 0:
                    for index in range(vis_start, vis_max, vis_step):
                        if n_queries[0] == index:
                            with torch.no_grad():
                                result = model(return_loss=False, rescale=True, **img_info)
                            vis_result.append(result)
            # acc = (margin_min > 0.0).sum() / n_ex_total
            # acc_corr = (margin_min > 0.0).mean()
            # mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min <= 0]), np.median(n_queries[margin_min <= 0])
            # avg_margin_min = np.mean(margin_min)
            
            # time_total = time.time() - time_start
            # log.print('{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.2f} med#q={:.1f}, avg_margin={:.2f} (n_ex={}, eps={:.3f}, {:.2f}s)'.
            #     format(i_iter+1, acc, acc_corr, mean_nq_ae, median_nq_ae, avg_margin_min, x.shape[0], eps, time_total))

            # metrics[i_iter] = [acc, acc_corr, mean_nq, mean_nq_ae, median_nq_ae, margin_min.mean(), time_total]
            # if (i_iter <= 500 and i_iter % 20 == 0) or (i_iter > 100 and i_iter % 50 == 0) or i_iter + 1 == n_iters or acc == 0:
                # np.save(metrics_path, metrics)
            if loss_min == 0:
                break
    img_info['img'][0] = torch.from_numpy(x_best.astype('float32'))

    if vis_attack_step is not None and vis_attack_step:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **img_info)        
        return n_queries, vis_result
    return n_queries


