import cv2
import numpy as np
import torch
import random
import logging
from logging import handlers
from mmdet.models import build_detector
from mmcv.runner import (load_checkpoint, wrap_fp16_model)
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from PIL import Image
import mmcv

results_records_iter_list_for_square = [0, 12, 52, 202, 502, 1002, 2002, 4002, 6002, 8002, 10002]
results_records_iter_list_for_signhunter = [0, 10, 50, 200, 500, 1000, 2000, 4000, 6000, 8000, 10000]
results_records_iter_list_for_zosignsgd = [0, 101, 202, 404, 505, 1010, 2020, 4040, 6060, 8080, 10100]
results_records_iter_list_for_nes = [0, 100, 200, 400, 800, 1000, 2000, 4000, 6000, 8000, 10000]

def get_first(a, max_iter):
    res = None
    for i, item in enumerate(a):
        if item > max_iter:
            res = i
            break
    return res

def unique_rows(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1) 
    return a[ui]

def label_smooth(scores, labels, n_cls):
    result = np.ones([len(labels), n_cls])
    for i in range(len(labels)):
        result[i] = (1.0 - scores[i]) / (n_cls - 1) * result[i]
        result[i, labels[i]] = scores[i]
    return result

def get_scores_and_labels(result, ncls):
    for i in result[0]:
        bbox_scores = i[:, -1]
    bbox_labels = result[1][0].int()

    scores_clean = bbox_scores.cpu().detach().numpy()
    labels_clean = bbox_labels.cpu().detach().numpy()
    scores_smooth_clean = label_smooth(scores_clean, labels_clean, ncls)
    return scores_smooth_clean, labels_clean

def get_bboxes_scores_and_labels(result, ncls, to_Tensor=False):
    for i in result[0]:
        bbox_scores = i[:, -1]
        bbox_coors = i[:, 0:-1]
    bbox_labels = result[1][0].int()

    if not to_Tensor:
        scores_clean = bbox_scores.cpu().detach().numpy()
        labels_clean = bbox_labels.cpu().detach().numpy()
        bbox_coors = bbox_coors.cpu().detach().numpy()
        scores_smooth_clean = label_smooth(scores_clean, labels_clean, ncls)
    else:
        scores_clean = bbox_scores
        labels_clean = bbox_labels.long()
        bbox_coors = bbox_coors
        return bbox_coors, scores_clean, labels_clean
    return bbox_coors, scores_smooth_clean, labels_clean

def get_gt_bboxes_scores_and_labels(Anns, cat2label, img_name, scale_factor, ncls):
    bboxes = []
    scores = []
    labels = []

    img_id = int(img_name.split('.')[0])
    img2bboxes = [Anns[img_id][i]['bbox'] for i in range(len(Anns[img_id]))]
    img2labels = [cat2label[Anns[img_id][i]['category_id']]  for i in range(len(Anns[img_id]))]
    # img2bboxes = np.array(img2bboxes*scale_factor)
    img2bboxes = np.array(img2bboxes)
    xs_left, ys_left, ws, hs = img2bboxes[:, 0], img2bboxes[:, 1], img2bboxes[:, 2], img2bboxes[:, 3]
    bboxes = np.column_stack((xs_left, ys_left, xs_left+ws, ys_left+hs))
    labels = np.array(img2labels)
    scores = np.zeros((len(labels), ncls))
    for i in range(len(labels)):
        scores[i, labels[i]] = 1.0
    return bboxes, scores, labels

def sq_loss(y, logits, targeted=False, loss_type='margin_loss'):
    if loss_type == 'margin_loss':
        preds_correct_class = (logits * y).sum(1, keepdims=True)
        diff = preds_correct_class - logits  # difference between the correct class and all other classes
        diff[y] = np.inf  # to exclude zeros coming from f_correct - f_correct
        margin = diff.min(1, keepdims=True)
        loss = margin * -1 if targeted else margin
    elif loss_type == 'cross_entropy':
        probs = utils.softmax(logits)
        loss = -np.log(probs[y])
        loss = loss * -1 if not targeted else loss
    else:
        raise ValueError('Wrong loss.')
    return loss.flatten()

def filter_scores_labels(scores, labels, objects):
    scores_smooth_adv = list()
    labels_adv = list()
    if len(labels) != 0:
        for i in range(len(labels)):
            if labels[i] in objects:
                scores_smooth_adv.append(scores[i])
                labels_adv.append(labels[i]) 
    scores_smooth_adv = np.array(scores_smooth_adv)
    labels_adv = np.array(labels_adv)
    return scores_smooth_adv, labels_adv

def filter_bboxes_scores_labels(bboxes, scores, labels, objects):
    bboxes_adv = list()
    scores_smooth_adv = list()
    labels_adv = list()
    if len(labels) != 0:
        for i in range(len(labels)):
            if labels[i] in objects:
                bboxes_adv.append(bboxes[i])
                scores_smooth_adv.append(scores[i])
                labels_adv.append(labels[i]) 
    bboxes_adv = np.array(bboxes_adv)
    scores_smooth_adv = np.array(scores_smooth_adv)
    labels_adv = np.array(labels_adv)
    return bboxes_adv, scores_smooth_adv, labels_adv

def img_to_attack_patch(attack_type, xs_t, clean_info, img_metas):
    # xs_t [batch, h, w, c]
    img = xs_t.detach().numpy()
    img_patch_lists = []
    if attack_type == 'bbox':
    # get the predicted border after scaling
        scale_factor = img_metas[0].data[0][0]['scale_factor']
        # bbox [x1, y1, x2, y2]
        bboxes_scaled_clean = (scale_factor * clean_info[0]).astype(int)
        x1 = bboxes_scaled_clean[:, 0]
        x2 = bboxes_scaled_clean[:, 2]
        y1 = bboxes_scaled_clean[:, 1]
        y2 = bboxes_scaled_clean[:, 3]
        for i in range(len(bboxes_scaled_clean)):
            ori_img = img[:, y1[i]:y2[i], x1[i]:x2[i], :][0]
            scale_img = cv2.resize(ori_img, (224, 224))
            img_patch_lists.append(scale_img)
        
    img_patches = np.array(img_patch_lists)
    img_patches_t = torch.from_numpy(img_patches)       
    _shape = list(img_patches_t.shape)
    dim = np.prod(_shape[1:])

    return img_patches_t, _shape, dim

def attack_patch_to_img(attack_type, xs_t, img_patches_t, clean_info, img_metas):
    img = xs_t.detach().numpy()
    img_patches = img_patches_t.detach().numpy()
    if attack_type == 'bbox':
        scale_factor = img_metas[0].data[0][0]['scale_factor'] 
        # bbox [x1, y1, x2, y2]
        bboxes_scaled_clean = (scale_factor * clean_info[0]).astype(int)
        x1 = bboxes_scaled_clean[:, 0]
        x2 = bboxes_scaled_clean[:, 2]
        y1 = bboxes_scaled_clean[:, 1]
        y2 = bboxes_scaled_clean[:, 3]
        for i in range(len(bboxes_scaled_clean)):
            scale_img_patch = img_patches[i]
            ori_h = y2[i]-y1[i]
            ori_w = x2[i]-x1[i]
            ori_img_patch = cv2.resize(scale_img_patch, (ori_w, ori_h))
            img[0][y1[i]:y2[i], x1[i]:x2[i], :] = ori_img_patch
        
        img = torch.from_numpy(img)
        
        return img

def bbox_to_attack_points(h, w, clean_info, img_metas, s=None, get_proposal=False):
    attack_points = list()

    # get the predicted border after scaling
    scale_factor = img_metas[0].data[0][0]['scale_factor']
    # bbox [x1, y1, x2, y2]
    if get_proposal:
        bboxes_scaled_clean = clean_info.astype(int)[:30]
    else:
        bboxes_scaled_clean = (scale_factor * clean_info[0]).astype(int)
    # filter boxes over w-s, h-s
    x1_mask = bboxes_scaled_clean[:, 0] < (w-s)
    y1_mask = bboxes_scaled_clean[:, 1] < (h-s)
    mask = x1_mask & y1_mask
    filter_bboxes = bboxes_scaled_clean[mask]
    
    x1 = filter_bboxes[:, 0]
    x2 = np.clip(filter_bboxes[:, 2], 0, w-s)
    y1 = filter_bboxes[:, 1]
    y2 = np.clip(filter_bboxes[:, 3], 0, h-s)

    for index in range(0, len(x1)):
        for temp_w in range(x1[index], x2[index], 1):
            for temp_h in range(y1[index], y2[index], 1):
                attack_points.append([temp_h, temp_w])

    attack_points = np.array(attack_points)
    # attack_points_unique = unique_rows(attack_points)

    return attack_points

def mask_to_attack_points(clean_info, seg_masks, img_metas, s=None):
    attack_points = np.empty(shape=[0, 2])
    
    object_labels = clean_info[3]
    h, w, _ = img_metas[0].data[0][0]['pad_shape']

    for label in object_labels:
        for mask in seg_masks[int(label)]:
            mask = np.array(mask, dtype='uint8')
            mask_rescale = mmcv.imresize(mask, (w, h))
            mask_rescale = np.mat(mask_rescale)
            temp_h, temp_w = np.nonzero(mask_rescale)
            if s is not None:
                temp_h = np.clip(temp_h, 0, h-s)
                temp_w = np.clip(temp_w, 0, w-s)
            temp_points = np.stack((temp_h, temp_w), axis=-1)
            attack_points = np.append(attack_points, temp_points, axis=0)
            # masks_rescale.append(mmcv.imresize(mask, (w, h)))
    # attack_points_unique = unique_rows(attack_points)
    return attack_points

def get_untargeted_label_dic(gt_labels, ncls):
    dic = dict()

    labels = [i for i in range(ncls)]
    random.shuffle(labels)
    object_labels = np.unique(gt_labels)
    
    for it, object_label in enumerate(object_labels):
        labels.remove(int(object_label))
        dic[object_label] = labels[it]

    return dic
    


def reppoints_to_attack_points(results, h, w, s=None):
    attack_points = np.empty(shape=[0, 2])

    pts_pred_refine_all_layers = results[0]
    topk_inds_all_layers = results[1]
    points_all_layers = results[2]

    assert (len(pts_pred_refine_all_layers) == len(topk_inds_all_layers) == len(points_all_layers))

    for i in range(len(points_all_layers)):
        pts_pred_refine = pts_pred_refine_all_layers[i]
        pts_pred_refine = pts_pred_refine.view(pts_pred_refine.shape[0], pts_pred_refine.shape[1], -1)
        topk_inds = topk_inds_all_layers[i]
        pts_pred_refine = pts_pred_refine[:, :, topk_inds]

        pts_reshape = pts_pred_refine.view(pts_pred_refine.shape[0], -1, 2, *pts_pred_refine.shape[2:])
        pts_y_shift = pts_reshape[:, :, 0, ...] 
        pts_x_shift = pts_reshape[:, :, 1, ...]

        pts_center_y = points_all_layers[i][:, 1]
        pts_center_w = points_all_layers[i][:, 0]
        pts_stride = points_all_layers[i][:, 2]

        temp_h = (pts_y_shift * pts_stride + pts_center_y).cpu().detach().numpy()
        temp_w = (pts_x_shift * pts_stride + pts_center_w).cpu().detach().numpy()
        pts_center_y = pts_center_y.cpu().detach().numpy()
        pts_center_w = pts_center_w.cpu().detach().numpy()

        temp_h = np.array(temp_h, dtype='int64').flatten()
        temp_w = np.array(temp_w, dtype='int64').flatten()
        pts_center_y = np.array(pts_center_y, dtype='int64').flatten()
        pts_center_w = np.array(pts_center_w, dtype='int64').flatten()

        if s is not None:
            temp_h = np.clip(temp_h, 0, h-s)
            temp_w = np.clip(temp_w, 0, w-s)
            pts_center_y = np.clip(pts_center_y, 0, h-s)
            pts_center_w = np.clip(pts_center_w, 0, w-s)

        temp_points = np.stack((temp_h, temp_w), axis=-1)
        temp_center_points = np.stack((pts_center_y, pts_center_w), axis=-1)
        attack_points = np.append(attack_points, temp_points, axis=0)
        attack_points = np.append(attack_points, temp_center_points, axis=0)

    return attack_points

def init_keypoints_model(cfg, checkpoint, fuse_conv_bn):
    cfg.model.pretrained = None
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    keypoints_model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(keypoints_model)
    checkpoint = load_checkpoint(keypoints_model, checkpoint, map_location='cpu')
    if fuse_conv_bn:
        keypoints_model = fuse_conv_bn(keypoints_model)
    
    keypoints_model = MMDataParallel(keypoints_model, device_ids=[0])
    keypoints_model.eval()
    return keypoints_model

class Logger(object):
    level_relations = {
    'debug':logging.DEBUG,
    'info':logging.INFO,
    'warning':logging.WARNING,
    'error':logging.ERROR,
    'crit':logging.CRITICAL
    }
 
    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器

        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)

# if __name__ == '__main__':
#  log = Logger('all.log',level='debug')
#  log.logger.debug('debug')
#  log.logger.info('info')
#  log.logger.warning('警告')
#  log.logger.error('报错')
#  log.logger.critical('严重')
#  Logger('error.log', level='error').logger.error('error')