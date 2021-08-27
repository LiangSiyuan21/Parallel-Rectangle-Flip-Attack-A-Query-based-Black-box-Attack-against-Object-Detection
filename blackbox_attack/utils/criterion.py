# -*- encoding: utf-8 -*-
'''
@File    :   criterion.py
@Time    :   2020/12/01 14:08:52
@Author  :   liangsiyuan 
'''

# here put the import lib
import torch
import attack_demo.util as demo_utils
import blackbox_attack.square_attack.utils as sq_utils
from mmdet.core import bbox_overlaps
import numpy as np

def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    # S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    # S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    # sum_area = S_rec1 + S_rec2
    # intersect = (right_line - left_line) * (bottom_line - top_line)
    # return (intersect / (sum_area - intersect))*1.0
    
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return torch.DoubleTensor([0.0001]).cuda()
    else:
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
        # computing the sum_area
        sum_area = S_rec1 + S_rec2
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0



def cw_loss(logit, label, target=False, labels_dic=None):
    if target:
        # targeted cw loss: logit_t - max_{i\neq t}logit_i
        _, argsort = logit.sort(dim=1, descending=True)
        target_is_max = argsort[:, 0].eq(label)
        second_max_index = target_is_max.long() * argsort[:, 1] + (~ target_is_max).long() * argsort[:, 0]
        target_logit = logit[torch.arange(logit.shape[0]), label]
        second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
        return target_logit - second_max_logit 
    else:
        # untargeted cw loss: max_{i\neq y}logit_i - logit_y
        
        _, argsort = logit.sort(dim=1, descending=True)
        if labels_dic is not None:
            untargetd_labels = [labels_dic[int(item)] for item in label]
            untargetd_labels = torch.tensor(untargetd_labels, dtype=torch.long).cuda()
        gt_is_max = argsort[:, 0].eq(label)
        # second_max_index = gt_is_max.long() * argsort[:, 1] + (1 - gt_is_max).long() * argsort[:, 0]
        if labels_dic is None:
            second_max_index = gt_is_max.long() * argsort[:, 1] + (~gt_is_max).long() * argsort[:, 0]
        else:
            second_max_index = gt_is_max.long() * untargetd_labels + (~gt_is_max).long() * argsort[:, 0]
        gt_logit = logit[torch.arange(logit.shape[0]), label]
        second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
        return second_max_logit - gt_logit

def xent_loss(logit, label, target=False):
    if not target:
        return torch.nn.CrossEntropyLoss(reduction='none')(logit, label)                
    else:
        return -torch.nn.CrossEntropyLoss(reduction='none')(logit, label)

def iou_loss(pred, gt, target=False):
    if target:
        return compute_iou(pred, gt)
    else:
        return 1.0 / compute_iou(pred, gt)

def loss_fct(attacker, xs, img_metas, clean_info):
    epsilon = attacker.epsilon 
    x_eval = torch.FloatTensor(xs.transpose(0,3,1,2))
    x_eval = torch.clamp(x_eval - attacker.ori_img, -epsilon, epsilon) + attacker.ori_img
    x_eval = torch.clamp(x_eval, attacker.lb, attacker.ub)

    objects_clean = clean_info[0]
    # zip images and image metas
    data = {}
    if attacker.model_name == "CornerNet":
        data['img'] = [x_eval, x_eval]
    else:
        data['img'] = [x_eval]
    data['img_metas'] = img_metas

    with torch.no_grad():
        result = attacker.attack_model(return_loss=False, rescale=True, attack_mode=attacker.attack_mode, **data)
    
    scores_smooth_result, labels_result = demo_utils.get_scores_and_labels(result, ncls=80)
    scores_result = scores_smooth_result
    
    scores_adv, labels_adv = demo_utils.filter_scores_labels(scores_result, labels_result, objects_clean)
    
    # if adv score is zero, it means no box to attack
    if len(scores_adv) == 0:
        return np.array(0.0)
    
    labels_target = sq_utils.random_classes_except_current(labels_adv, 80) if attacker.targeted else labels_adv

    for loss_type in attacker.loss:
        if loss_type == 'cw_loss':
            criterion = cw_loss
            loss_cls = criterion(torch.DoubleTensor(scores_adv).cuda(), torch.LongTensor(labels_target).cuda(), attacker.targeted).sum(0).unsqueeze(0)
        elif loss_type == 'xent_loss':
            criterion = xent_loss        
            loss_cls = criterion(torch.DoubleTensor(scores_adv).cuda(), torch.LongTensor(labels_target).cuda(), attacker.targeted).sum(0).unsqueeze(0)
        
        # if loss_type == 'iou_loss':
        #     print('iou_loss')
        #     print(type(loss_cls))

    return loss_cls.detach().cpu().data.numpy()

def loss_fct_with_iou(attacker, xs, img_metas, clean_info):
    epsilon = attacker.epsilon 
    x_eval = torch.FloatTensor(xs.transpose(0,3,1,2))
    x_eval = torch.clamp(x_eval - attacker.ori_img, -epsilon, epsilon) + attacker.ori_img
    x_eval = torch.clamp(x_eval, attacker.lb, attacker.ub)

    # unzip clean info
    if len(clean_info) == 4:
        bboxes_clean, bbox_scores_clean, labels_clean, objects_clean = clean_info
        labels_dic = None
    if len(clean_info) == 5:
        bboxes_clean, bbox_scores_clean, labels_clean, objects_clean, labels_dic = clean_info
    # zip images and image metas
    data = {}
    if attacker.model_name == "CornerNet":
        data['img'] = [x_eval, x_eval]
    else:
        data['img'] = [x_eval]
    data['img_metas'] = img_metas

    with torch.no_grad():
        result = attacker.attack_model(return_loss=False, rescale=True, attack_mode=attacker.attack_mode, **data)
    bbox_results, score_results, label_results = demo_utils.get_bboxes_scores_and_labels(result, ncls=80)
    
    bboxes_adv, scores_adv, labels_adv = demo_utils.filter_bboxes_scores_labels(bbox_results, score_results, label_results, objects_clean)
    
    # if adv score is zero, it means no box to attack
    if len(labels_adv) == 0:
        return np.array(0.0)
    
    labels_target = sq_utils.random_classes_except_current(labels_adv, 80) if attacker.targeted else labels_adv
    
    
    for loss_type in attacker.loss:
        if loss_type == 'cw_loss':
            cls_criterion = cw_loss
        elif loss_type == 'xent_loss':
            cls_criterion = xent_loss
        if loss_type == 'iou_loss':
            iou_criterion = iou_loss

    loss_cls = torch.DoubleTensor([0.0]).cuda()
    loss_iou = torch.DoubleTensor([0.0]).cuda()

    # for object_clean in objects_clean:
    #     pred_indexes = np.where(labels_adv==object_clean)[0]
    #     gt_indexes = np.where(labels_clean==object_clean)[0]        
    #     for pred_index in pred_indexes: 
    #         for gt_index in gt_indexes:
    #             if score_results[int(pred_index)][object_clean] < attacker.zeta:     
    #                 loss_cls += cls_criterion(torch.DoubleTensor(scores_adv[int(pred_index)]).unsqueeze(0).cuda(), torch.LongTensor([labels_target[int(pred_index)]]).cuda(), attacker.targeted).sum(0).unsqueeze(0) 
    #             else:
    #                 loss_iou += iou_criterion(torch.DoubleTensor(bboxes_adv[int(pred_index)]).cuda(), torch.DoubleTensor(bboxes_clean[int(gt_index)]).cuda(), attacker.targeted).sum(0).unsqueeze(0)

    # scores_mask = [ scores.max() < attacker.zeta for scores in scores_adv]
    scores_mask = [ scores.max() < 1.0 for scores in scores_adv]
    pred_scores = scores_adv[scores_mask]
    labels_target = labels_target[scores_mask]
    if labels_dic is not None and loss_type == 'cw_loss':
        loss_cls += cls_criterion(torch.DoubleTensor(pred_scores).cuda(), torch.LongTensor(labels_target).cuda(), False, labels_dic=labels_dic).sum(0).unsqueeze(0)      
    else:
        loss_cls += cls_criterion(torch.DoubleTensor(pred_scores).cuda(), torch.LongTensor(labels_target).cuda(), False).sum(0).unsqueeze(0)
       
    for object_clean in objects_clean:
        pred_indexes = np.where(labels_adv==object_clean)[0]
        gt_indexes = np.where(labels_clean==object_clean)[0]        

        pred_bboxes = bbox_results[pred_indexes]
        pred_scores_adv = score_results[pred_indexes]
        # pick scores > 0.05 bbox
        pred_bboxes = pred_bboxes[score_results[pred_indexes, object_clean] > attacker.zeta]
        # pred_bboxes = pred_bboxes[score_results[pred_indexes, object_clean] > 0.05]
        
        pred_bboxes_weight = pred_scores_adv[score_results[pred_indexes, object_clean] > attacker.zeta]
        if pred_bboxes_weight.shape[0] != 0:
            pred_bboxes_weight = pred_bboxes_weight.max()

        pred_bboxes_tor = torch.from_numpy(pred_bboxes).unsqueeze(0).float()
        gt_bboxes_tor = torch.from_numpy(bboxes_clean[gt_indexes]).unsqueeze(0).float()
        # pick iou scores > 0.5 bbox
        ious = bbox_overlaps(pred_bboxes_tor, gt_bboxes_tor, mode='iou', is_aligned=False).clamp(min=1e-6)
        if ious.size(-2) == 0:
            loss_iou += torch.DoubleTensor([0.0]).cuda()
        else:
            ious = ious.view(-1)
            # print(ious)
            # print(-ious.log().sum())
            loss_iou += -ious.log().sum().cuda()
            # print(loss_iou)

    # for loss_type in attacker.loss:
    #     if loss_type == 'cw_loss':
    #         criterion = cw_loss
    #         loss_cls = criterion(torch.DoubleTensor(scores_adv).cuda(), torch.LongTensor(labels_target).cuda(), attacker.targeted).sum(0).unsqueeze(0)
    #     elif loss_type == 'xent_loss':
    #         criterion = xent_loss        
    #         loss_cls = criterion(torch.DoubleTensor(scores_adv).cuda(), torch.LongTensor(labels_target).cuda(), attacker.targeted).sum(0).unsqueeze(0)        
        
    #     if loss_type == 'iou_loss':
    #         print('test')
    # print('cls loss')
    # print(loss_cls)
    # print('iou loss')
    # print(loss_iou*attacker.lambda1)
    # print('total loss')
    # print(loss_cls+loss_iou*attacker.lambda1)
    # print('cls loss:%f, iou loss:%f, total loss:%f'.format(loss_cls, loss_iou*attacker.lambda1, loss_cls+loss_iou*attacker.lambda1))
    return (loss_cls+loss_iou*attacker.lambda1).detach().cpu().data.numpy()

def early_stop_crit_fct(attacker, xs, img_metas, clean_info):
    epsilon = attacker.epsilon   
    x_eval = torch.FloatTensor(xs.transpose(0,3,1,2))    
    x_eval = torch.clamp(x_eval - attacker.ori_img, -epsilon, epsilon) + attacker.ori_img
    x_eval = torch.clamp(x_eval, attacker.lb, attacker.ub)

    objects_clean = clean_info[0]    

    # zip images and image metas
    data = {}
    if attacker.model_name == "CornerNet":
        data['img'] = [x_eval, x_eval]
    else:
        data['img'] = [x_eval]
    data['img_metas'] = img_metas

    # get scores and labels
    with torch.no_grad():
        result = attacker.attack_model(return_loss=False, rescale=True, attack_mode=attacker.attack_mode, **data)
    score_smooth_results, label_results = demo_utils.get_scores_and_labels(result, ncls=80)
    scores_result = score_smooth_results

    if len(label_results) == 0:
        return [False]
    
    # get attack scores and labels which belongs to clean objects label
    correct = list()
    for label in label_results:
        if label in objects_clean:
            correct.append(True)
        else:
            correct.append(False)
    correct = np.array(correct)
    
    if attacker.targeted:
        return correct
    else:
        return np.logical_not(correct)    

def early_stop_crit_fct_with_iou(attacker, xs, img_metas, clean_info):
    epsilon = attacker.epsilon
    x_eval = torch.FloatTensor(xs.transpose(0,3,1,2))
    x_eval = torch.clamp(x_eval-attacker.ori_img, -epsilon, epsilon) + attacker.ori_img
    x_eval = torch.clamp(x_eval, attacker.lb, attacker.ub)

    if len(clean_info) == 4:
        bboxes_clean, bbox_scores_clean, labels_clean, objects_clean = clean_info
    if len(clean_info) == 5:
        bboxes_clean, bbox_scores_clean, labels_clean, objects_clean, _ = clean_info
    # zip images and image img_metas
    data = {}
    if attacker.model_name == "CornerNet":
        data['img'] = [x_eval, x_eval]
    else:
        data['img'] = [x_eval]
    data['img_metas'] = img_metas

    # get bboxes, scores, labels
    with torch.no_grad():
        result = attacker.attack_model(return_loss=False, rescale=True, attack_mode=attacker.attack_mode, **data)
    bbox_results, score_results, label_results = demo_utils.get_bboxes_scores_and_labels(result, ncls=80)

    if len(label_results) == 0:
        return [False]
    
    # iou<0.3 or scores<0.05
    # correct = list()
    # for object_clean in objects_clean:
    #     pred_indexes = np.where(label_results==object_clean)[0]
    #     gt_indexes = np.where(labels_clean==object_clean)[0]        
    #     for pred_index in pred_indexes:
    #         flag = False
    #         for gt_index in gt_indexes:
    #             # if (compute_iou(bbox_results[int(pred_index)], bboxes_clean[int(gt_index)]) > 0.5) and (score_results[int(pred_index)][object_clean] > 0.05):
    #             if (score_results[int(pred_index)][object_clean] > 0.05) and (compute_iou(bbox_results[int(pred_index)], bboxes_clean[int(gt_index)]) > 0.5):
    #                 flag = True
    #         correct.append(flag)
    # correct = np.array(correct)    

    # improve iou<0.3 or scores<0.05
    correct = list()
    for object_clean in objects_clean:
        pred_indexes = np.where(label_results==object_clean)[0]
        gt_indexes = np.where(labels_clean==object_clean)[0]        
        # for pred_index in pred_indexes:
        #     flag = False
        #     for gt_index in gt_indexes:
        #         # if (compute_iou(bbox_results[int(pred_index)], bboxes_clean[int(gt_index)]) > 0.5) and (score_results[int(pred_index)][object_clean] > 0.05):
        #         if (score_results[int(pred_index)][object_clean] > 0.05) and (compute_iou(bbox_results[int(pred_index)], bboxes_clean[int(gt_index)]) > 0.5):
        #             flag = True
        #     correct.append(flag)
        pred_bboxes = bbox_results[pred_indexes]
        # pick scores > 0.05 bbox
        pred_bboxes = pred_bboxes[score_results[pred_indexes, object_clean] > 0.05]
        pred_bboxes_tor = torch.from_numpy(pred_bboxes).unsqueeze(0).float()
        gt_bboxes_tor = torch.from_numpy(bboxes_clean[gt_indexes]).unsqueeze(0).float()
        # pick iou scores > 0.5 bbox
        ious = bbox_overlaps(pred_bboxes_tor, gt_bboxes_tor, mode='iou', is_aligned=False).clamp(min=1e-6)[0].numpy().flatten() > 0.5
        correct.extend(ious)
    correct = np.array(correct)     

    if attacker.targeted:
        return correct
    else:
        return np.logical_not(correct)