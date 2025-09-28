import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

import ipdb

import numpy as np
import logging
import torch.backends.cudnn as cudnn
import os
import pickle

from tqdm import tqdm
import torch

from Utils.utils import gpu, decimals
import json


def get_gt(video_metas, query_metas):
    v2t_gt = []
    for vid_id in video_metas:
        v2t_gt.append([])
        for i, query_id in enumerate(query_metas):
            if query_id.split('#', 1)[0] == vid_id:
                v2t_gt[-1].append(i)

    t2v_gt = {}
    for i, t_gts in enumerate(v2t_gt):
        for t_gt in t_gts:
            t2v_gt.setdefault(t_gt, [])
            t2v_gt[t_gt].append(i)

    return v2t_gt, t2v_gt


def eval_q2m(scores, q2m_gts):

    n_q, n_m = scores.shape

    gt_ranks = torch.zeros((n_q), dtype=torch.int32).cuda()
    aps = torch.zeros(n_q).cuda()
    for i in range(n_q):
        s = scores[i]
        sorted_idxs = torch.argsort(s)
        rank = n_m + 1
        tmp_set = []
        for k in q2m_gts[i]:
            tmp = torch.where(sorted_idxs == k)[0][0] + 1
            if tmp < rank:
                rank = tmp

        gt_ranks[i] = rank

    # compute metrics
    r1 = 100.0 * len(torch.where(gt_ranks <= 1)[0]) / n_q
    r5 = 100.0 * len(torch.where(gt_ranks <= 5)[0]) / n_q
    r10 = 100.0 * len(torch.where(gt_ranks <= 10)[0]) / n_q
    r100 = 100.0 * len(torch.where(gt_ranks <= 100)[0]) / n_q

    return (r1, r5, r10, r100)


def cal_perf(t2v_all_errors, t2v_gt):

    # video retrieval
    (t2v_r1, t2v_r5, t2v_r10, t2v_r100) = eval_q2m(t2v_all_errors, t2v_gt)

    logging.info(" * Text to Video:")
    logging.info(" * r_1_5_10_100, medr, meanr: {}".format([round(t2v_r1, 1), round(t2v_r5, 1), round(t2v_r10, 1), round(t2v_r100, 1)]))
    logging.info(" * recall sum: {}".format(round(t2v_r1+t2v_r5+t2v_r10+t2v_r100, 1)))

    return (t2v_r1, t2v_r5, t2v_r10, t2v_r100)
    
def cal_perf_sum(t2v_all_errors, t2v_gt):

    # video retrieval
    (t2v_r1, t2v_r5, t2v_r10, t2v_r100) = eval_q2m(t2v_all_errors, t2v_gt)
    
    t2v_r1, t2v_r5, t2v_r10, t2v_r100 = decimals(t2v_r1), decimals(t2v_r5), decimals(t2v_r10), decimals(t2v_r100)
    
    logging.info(" * Text to Video:")
    logging.info(" * r_1_5_10_100, medr, meanr: {}".format([decimals(t2v_r1), decimals(t2v_r5), decimals(t2v_r10), decimals(t2v_r100)]))
    logging.info(" * recall sum: {}".format(decimals(t2v_r1+t2v_r5+t2v_r10+t2v_r100)))

    return (t2v_r1, t2v_r5, t2v_r10, t2v_r100)

def sinkhorn_knopp(log_sim_matrix, n_iters=4, detach=False):

    if detach:
        log_sim_matrix = log_sim_matrix.detach()

    # m= torch.max(log_sim_matrix)
    m = log_sim_matrix.max()
    _log_sim_matrix = log_sim_matrix - m

    " Subtract the maximum value m from each element to normalize the log-similarity matrix. "
    sim_matrix = torch.exp(_log_sim_matrix)
    b = 1 / sim_matrix.sum(0)

    # Calculate the sum of each column in the sim_matrix, then take the reciprocal of these sums to obtain the initial b vector.

    for _ in range(n_iters):
        a = 1 / (sim_matrix @ b)
        b = 1 / (a @ sim_matrix)

    log_a = a.log()
    log_b = b.log() - m

    return F.log_softmax(log_a, dim=0), F.log_softmax(log_b, dim=0)


class validations(nn.Module):
    def __init__(self, cfg):
        super(validations, self).__init__()

        self.cfg = cfg


    def forward(self, model, context_dataloader, query_eval_loader):

        model.eval()

        context_info = self.compute_context_info(model, context_dataloader)
        query_context_scores, global_query_context_scores, score_sum, query_metas = self.compute_query2ctx_info(model,
                                                             query_eval_loader,
                                                             context_info)
        video_metas = context_info['video_metas']
        
        # text_bias_1, video_bias_1 = sinkhorn_knopp(query_context_scores)
        # text_bias_2, video_bias_2 = sinkhorn_knopp(global_query_context_scores)
        # new_clip_scale_scores = query_context_scores + video_bias_1.unsqueeze(0)
        # new_frame_scale_scores = global_query_context_scores + video_bias_2.unsqueeze(0)

        v2t_gt, t2v_gt = get_gt(video_metas, query_metas)
        print('clip_scale_scores:')
        cal_perf(-1 * query_context_scores, t2v_gt)

        print('frame_scale_scores:')
        cal_perf(-1 * global_query_context_scores, t2v_gt)

        print('score_sum:')
        t2v_r1, t2v_r5, t2v_r10, t2v_r100 = cal_perf(-1 * score_sum, t2v_gt)
        t2v_rsum = 0
        t2v_rsum += (t2v_r1 + t2v_r5 + t2v_r10 + t2v_r100)

        return [t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_rsum]


    def compute_query2ctx_info(self, model, query_eval_loader, ctx_info):

        query_metas = []
        score_sum = []
        clip_scale_scores = []
        frame_scale_scores = []
        for idx, batch in tqdm(enumerate(query_eval_loader), desc="Computing q embedding", total=len(query_eval_loader)):

            batch = gpu(batch)
            query_metas.extend(batch[-1])
            query_feat = batch[0]
            query_mask = batch[1]
            _clip_scale_scores, _frame_scale_scores = model.get_pred_from_raw_query(
                query_feat, query_mask, None, ctx_info["video_proposal_feat"], ctx_info["video_feat"], ctx_info["video_mask"])
            _score_sum = self.cfg['clip_scale_w'] * _clip_scale_scores + self.cfg['frame_scale_w'] * _frame_scale_scores

            score_sum.append(_score_sum)
            clip_scale_scores.append(_clip_scale_scores)
            frame_scale_scores.append(_frame_scale_scores)

        score_sum = torch.cat(score_sum, dim=0)
        clip_scale_scores = torch.cat(clip_scale_scores, dim=0).cpu()
        frame_scale_scores = torch.cat(frame_scale_scores, dim=0).cpu()

        " Test the effect of the sinkhorn_knopp algorithm "
        # text_bias_1, video_bias_1 = sinkhorn_knopp(clip_scale_scores)
        # text_bias_2, video_bias_2 = sinkhorn_knopp(frame_scale_scores)
        # clip_scale_scores = clip_scale_scores + video_bias_1.unsqueeze(0)
        # frame_scale_scores = frame_scale_scores + video_bias_2.unsqueeze(0)

        return clip_scale_scores, frame_scale_scores, score_sum, query_metas


    def compute_context_info(self, model, context_dataloader):

        n_total_vid = len(context_dataloader.dataset)
        bsz = self.cfg['eval_context_bsz']
        metas = []  # list(dicts)
        vid_proposal_feat = []
        frame_feat, frame_mask = [], []
        for idx, batch in tqdm(enumerate(context_dataloader), desc="Computing query2video scores",
                            total=len(context_dataloader)):

            batch = gpu(batch)
            metas.extend(batch[-1])
            clip_video_feat_ = batch[0]
            frame_video_feat_ = batch[1]
            frame_mask_ = batch[2]
            _frame_feat, _video_proposal_feat = model.encode_context(clip_video_feat_, frame_video_feat_, frame_mask_)

            frame_feat.append(_frame_feat)
            frame_mask.append(frame_mask_)

            vid_proposal_feat.append(_video_proposal_feat)

        vid_proposal_feat = torch.cat(vid_proposal_feat, dim=0)

        def cat_tensor(tensor_list):
            if len(tensor_list) == 0:
                return None
            else:
                seq_l = [e.shape[1] for e in tensor_list]
                b_sizes = [e.shape[0] for e in tensor_list]
                b_sizes_cumsum = np.cumsum([0] + b_sizes)
                if len(tensor_list[0].shape) == 3:
                    hsz = tensor_list[0].shape[2]
                    res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l), hsz)
                elif len(tensor_list[0].shape) == 2:
                    res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l))
                else:
                    raise ValueError("Only support 2/3 dimensional tensors")
                for i, e in enumerate(tensor_list):
                    res_tensor[b_sizes_cumsum[i]:b_sizes_cumsum[i+1], :seq_l[i]] = e
                return res_tensor
                
        return dict(
            video_metas=metas,  # list(dict) (N_videos)
            video_proposal_feat=vid_proposal_feat,
            video_feat=cat_tensor(frame_feat),
            video_mask=cat_tensor(frame_mask)
            )