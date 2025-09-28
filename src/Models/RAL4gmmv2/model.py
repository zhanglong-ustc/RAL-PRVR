import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from Models.RAL4gmmv2.model_components import BertAttention, LinearLayer, \
                                            TrainablePositionalEncoding, DyGMMBlock
from Models.RAL4gmmv2.prob_models.pie_model import PIENet
from Models.RAL4gmmv2.prob_models.uncertainty_module import UncertaintyModuleImage
from Models.RAL4gmmv2.prob_models.tensor_utils import l2_normalize, sample_gaussian_tensors
from Models.RAL4gmmv2.prob_models.probemb import MCSoftContrastiveLoss
from Models.RAL4gmmv2.prob_models.until_loss import MILNCELoss_BoF, KLdivergence, con_loss
import ipdb
from scipy.optimize import linear_sum_assignment


class RAL4GMMV2_Net(nn.Module):
    def __init__(self, config):
        super(RAL4GMMV2_Net, self).__init__()
        self.config = config

        self.query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                           hidden_size=config.hidden_size, dropout=config.input_drop)
        self.clip_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                         hidden_size=config.hidden_size, dropout=config.input_drop)
        self.frame_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                          hidden_size=config.hidden_size, dropout=config.input_drop)
        #
        self.query_input_proj = LinearLayer(config.query_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.query_encoder = BertAttention(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop))
        
        self.frame_branch_query_encoder = BertAttention(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop))
        self.frame_branch_query_proj = LinearLayer(config.query_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)

        self.clip_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.clip_encoder = DyGMMBlock(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop, frame_len=32, sft_factor=config.sft_factor))

        self.frame_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                             dropout=config.input_drop, relu=True)
        self.frame_encoder_1 = DyGMMBlock(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop, frame_len=128, sft_factor=config.sft_factor))
                    
        self.modular_vector_mapping = nn.Linear(config.hidden_size, out_features=1, bias=False)
        self.weight_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        "multi-gauss prob"
        self.n_video_samples = 6   # numbers sampling from video distribution 
        self.n_text_samples = 6    # numbers sampling from text distribution

        self.pie_net_video = PIENet(1, config.hidden_size, config.hidden_size, config.hidden_size // 2)
        self.uncertain_net_video = UncertaintyModuleImage(config.hidden_size, config.hidden_size, config.hidden_size // 2)
        
        self.pie_net_text = PIENet(1, config.hidden_size, config.hidden_size, config.hidden_size // 2)
        self.uncertain_net_text = UncertaintyModuleImage(config.hidden_size, config.hidden_size, config.hidden_size // 2)

        self.loss_MIL_fct = MILNCELoss_BoF()
        self.vib_loss = KLdivergence()

        # Performing weight token-wise operation yields the importance level of each token in the query.
        self.text_weight_fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(inplace=True),
            nn.Linear(config.hidden_size, 1))
        
        self.reset_parameters()

    def reset_parameters(self):

        """ Initialize the weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size


    def forward(self, batch):

        clip_video_feat = batch['clip_video_features']
        query_feat = batch['text_feat']
        query_mask = batch['text_mask']
        query_labels = batch['text_labels']

        frame_video_feat = batch['frame_video_features']
        frame_video_mask = batch['videos_mask']

        label_dict = {} #  label_dict[i]: the  list of query index for video i; len(label_dict[i]): the number of annoatation for video i
        for index, label in enumerate(query_labels):
            if label in label_dict:
                label_dict[label].append(index)
            else:
                label_dict[label] = []
                label_dict[label].append(index)

        encoded_frame_feat, vid_proposal_feat = self.encode_context(
            clip_video_feat, frame_video_feat, frame_video_mask)

        clip_scale_scores, clip_scale_scores_, frame_scale_scores, frame_scale_scores_, MIL_loss, vib_loss \
            = self.get_pred_from_raw_query(
            query_feat, query_mask, query_labels, vid_proposal_feat, encoded_frame_feat, frame_video_mask, label_dict, return_query_feats=True)

        # video_query, unit_query = self.encode_query(query_feat, query_mask) # [Nq,d]
        video_query = self.encode_query(query_feat, query_mask) # [Nq,d]

        ######## ot-loss
        total_sim = []
        for k, v in label_dict.items():
            temp_clip_emb = vid_proposal_feat[k]
            temp_text_emb = video_query[v]

            if temp_text_emb.shape[0] == 1:
                continue

            sim = -1. * torch.matmul(F.normalize(temp_clip_emb, dim=-1), F.normalize(temp_text_emb, dim=-1).t()).permute(1, 0)
            indices = linear_sum_assignment(sim.detach().cpu())
            q_idx, c_idx = indices
            for i in range(q_idx.shape[0]):
                total_sim.append(sim[q_idx[i], c_idx[i]])

        total_sim = 1 + torch.stack(total_sim).mean()
        
        return [clip_scale_scores, clip_scale_scores_, label_dict, frame_scale_scores, frame_scale_scores_, video_query, total_sim, MIL_loss, vib_loss]


    def encode_query(self, query_feat, query_mask):
        encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj, self.query_encoder,
                                          self.query_pos_embed)  # (N, Lq, D)
        if query_mask is not None:
            mask = query_mask.unsqueeze(1)
        
        video_query  = self.get_modularized_queries(encoded_query, query_mask)  # (N, D) * 1

        return video_query
    
    def encode_frame_query(self, query_feat, query_mask):
        encoded_query = self.encode_input(query_feat, query_mask, self.frame_branch_query_proj, self.frame_branch_query_encoder,
                                          self.query_pos_embed)  # (N, Lq, D)
        if query_mask is not None:
            mask = query_mask.unsqueeze(1)
        
        return encoded_query


    def encode_context(self, clip_video_feat, frame_video_feat, video_mask=None):

        encoded_clip_feat = self.encode_input(clip_video_feat, None, self.clip_input_proj, self.clip_encoder,
                                               self.clip_pos_embed, self.weight_token)

        if frame_video_feat.shape[1] != 128:
            fix = 128 - frame_video_feat.shape[1]
            temp_feat = 0.0 * frame_video_feat.mean(dim=1, keepdim=True).repeat(1, fix, 1)
            frame_video_feat = torch.cat([frame_video_feat, temp_feat], dim=1)

            temp_mask = 0.0 * video_mask.mean(dim=1, keepdim=True).repeat(1, fix).type_as(video_mask)
            video_mask = torch.cat([video_mask, temp_mask], dim=1)
        
        encoded_frame_feat = self.encode_input(frame_video_feat, video_mask, self.frame_input_proj,
                                                self.frame_encoder_1,
                                                self.frame_pos_embed, self.weight_token)

        encoded_frame_feat = torch.where(video_mask.unsqueeze(-1).repeat(1, 1, encoded_frame_feat.shape[-1]) == 1.0, \
                                                                        encoded_frame_feat, 0. * encoded_frame_feat)

        return encoded_frame_feat, encoded_clip_feat
    
    def weighted_token_wise_intersection(self, text_token, frame_token, attention_mask, video_mask):

        text_weight = self.text_weight_fc(text_token).squeeze(2)  # B x N_t x D -> B x N_t
        text_weight.masked_fill_(torch.tensor((1 - attention_mask), dtype=torch.bool), float("-inf"))
        text_weight = torch.softmax(text_weight, dim=-1)  # B x N_t

        # token-wise interaction
        retrieve_logits = torch.einsum('atd,bvd->abtv', [text_token, frame_token])
        retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, attention_mask])
        retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, video_mask])
        text_sum = attention_mask.sum(-1)
        video_sum = video_mask.sum(-1)

        t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
        t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])

        "Since only part of the video is aligned with the query, using redundant video features fails to accurately assess the ambiguity of the video."
        # v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
        # v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])
        # retrieve_logits = (t2v_logits + v2t_logits) / 2.0
        retrieve_logits = t2v_logits 

        return retrieve_logits
    
    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer, weight_token=None):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            pos_embed_layer: positional embedding layer
        """

        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
        if weight_token is not None:
            return encoder_layer(feat, mask, weight_token)  # (N, L, D_hidden)
        else:
            return encoder_layer(feat, mask)  # (N, L, D_hidden)


    def get_modularized_queries(self, encoded_query, query_mask):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        modular_attention_scores = self.modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)

        return modular_queries.squeeze()
    
    def get_frame_scale_scores(self, text_token, frame_token, text_mask, video_mask):

        norm_query = F.normalize(text_token, dim=-1) # b*l*d
        norm_frame = F.normalize(frame_token, dim=-1) # b*n*d

        f_level_q2v_scores = self.weighted_token_wise_intersection(norm_query, norm_frame, text_mask, video_mask)

        return f_level_q2v_scores
    
    # @staticmethod
    # def get_frame_scale_scores(modularied_query, context_feat):

    #     modularied_query = F.normalize(modularied_query, dim=-1)
    #     context_feat = F.normalize(context_feat, dim=-1)

    #     clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)

    #     query_context_scores, indices = torch.max(clip_level_query_context_scores,
    #                                               dim=1)  # (N, N) diagonal positions are positive pairs
        
    #     return query_context_scores
    
        
    def get_unnormalized_frame_scale_scores(self, text_token, frame_token, text_mask, video_mask):

        output_query_context_scores = self.weighted_token_wise_intersection(text_token, frame_token, text_mask, video_mask)


        return output_query_context_scores

    def probabilistic_video(self, video_pooled, videos):

        output = {}

        "Aggregate the videos via self-attention (self-att), then perform a residual connection with video_pooled."
        out, attn, residual = self.pie_net_video(video_pooled, videos)  # (B 512) (B 12 512)   multiheadatt + fc + sigmoid + (residual) + laynorm
        uncertain_out = self.uncertain_net_video(video_pooled, videos)  # (B 512) (B 12 512)   multiheadatt + fc + (residual)         
        logsigma = uncertain_out['logsigma']

        output['logsigma'] = logsigma       # B 512     可以看作是方差
        output['uncertainty_attention'] = uncertain_out['attention']

        out = l2_normalize(out)     # B 512     l2 normalization后 均值
        output['mean'] = out   
        output['embedding'] = sample_gaussian_tensors(out, logsigma, self.n_video_samples)  # B 7 512    从高斯分布中采样N个embedding  

        return output
    
    def probabilistic_text(self, text_pooled, text_token):
        output = {}

        out, attn, residual = self.pie_net_text(text_pooled, text_token)     
        uncertain_out = self.uncertain_net_text(text_pooled, text_token)     

        logsigma = uncertain_out['logsigma']
        output['logsigma'] = logsigma
        output['uncertainty_attention'] = uncertain_out['attention']

        out = l2_normalize(out)
        output['mean'] = out

        output['embedding'] = sample_gaussian_tensors(out, logsigma, self.n_text_samples)

        return output
    

    def mean_pooling_for_norm_visual(self, visual_frame, video_mask,):

        visual_output = F.normalize(visual_frame, dim=-1)

        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum

        video_out = F.normalize(video_out, dim=-1)

        return video_out
    
    @staticmethod
    def get_clip_scale_scores(modularied_query, context_feat):

        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat = F.normalize(context_feat, dim=-1)

        clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)
        query_context_scores, indices = torch.max(clip_level_query_context_scores,
                                                  dim=1)  # (N, N) diagonal positions are positive pairs
        return query_context_scores
    
    
    @staticmethod
    def get_unnormalized_clip_scale_scores(modularied_query, context_feat):

        query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)
        output_query_context_scores, indices = torch.max(query_context_scores, dim=1)

        return output_query_context_scores
    
    @staticmethod
    def align_matrices(label_dict, video_matrix, query_matrix):
   
        ####### Align the video matrix and the query matrix according to the label_dict. ######
        if video_matrix.dim() == 2:
            leng = query_matrix.shape[0]
            num = video_matrix.shape[1]
            aligned_video_matrix = torch.zeros(leng, num).to(video_matrix.device)

        elif video_matrix.dim() == 3:
            leng = query_matrix.shape[0]
            num = video_matrix.shape[1]
            dim = video_matrix.shape[2]
            aligned_video_matrix = torch.zeros(leng, num, dim).to(video_matrix.device)

        for video_id, query_id in label_dict.items():
            for i in query_id:
                if 0 <= i < leng:  # Ensure that the query_id is within the valid index range of the query matrix.
                    # Ensure the use of correct indexing operations for tensors.
                    aligned_video_matrix[i] = video_matrix[video_id]
            
        return aligned_video_matrix
    
    @staticmethod
    def support_set(label_dict, video_matrix, query_matrix):
   
        ####### label_dict 640*30*128 --> 128*(30*5)*128 ######

        if video_matrix.dim() == 2:
            leng = query_matrix.shape[0]
            bs = video_matrix.shape[0]
            num = (query_matrix.shape[1]) * 26
            support_tensor = torch.zeros(bs, num).to(video_matrix.device)

        elif video_matrix.dim() == 3:
            leng = query_matrix.shape[0]
            bs = video_matrix.shape[0]
            num = (query_matrix.shape[1]) * 26  # Multiply tvr by 5, and multiply act by 26.
            dim = query_matrix.shape[2]
            support_tensor = torch.zeros(bs, num, dim).to(video_matrix.device)

        for i, (video_id, query_id_list) in enumerate(label_dict.items()):
            token_matrix_list = []   # Initialize a list to store the token tensor matrices belonging to the same video_id.
            for query_id in query_id_list: 
                if 0 <= query_id < leng:  # Ensure that the query_id is within the valid index range of the query matrix.
                    token_matrix = query_matrix[query_id]
                    token_matrix_list.append(token_matrix)
                    # token_matrix_list[0].size()   [28, 384]

            if len(token_matrix_list) > 0:
                
                concatenated_matrix = torch.cat(token_matrix_list, dim=0)
                tn = concatenated_matrix.shape[0]
                support_tensor[i,:tn] = concatenated_matrix
                
            else:
                raise ValueError(f"No token matrices found for video_id: {video_id}")
            
        return support_tensor
    
    def mean_pooling_for_sequence(self, sequence_output, attention_mask):

        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)

        return text_out


    def get_pred_from_raw_query(self, query_feat, query_mask, query_labels=None,
                                video_proposal_feat=None, encoded_frame_feat=None, video_mask=None, label_dict=None,
                                return_query_feats=False):

        video_query = self.encode_query(query_feat, query_mask)
        frame_query = self.encode_frame_query(query_feat, query_mask)

        ##### frame_query 2 support_query
        

        # get clip-level retrieval scores
        clip_scale_scores = self.get_clip_scale_scores(
            video_query, video_proposal_feat)

        frame_scale_scores = self.get_frame_scale_scores(
            frame_query, encoded_frame_feat, query_mask, video_mask)
  
        
        if return_query_feats:

            clip_scale_scores_ = self.get_unnormalized_clip_scale_scores(video_query, video_proposal_feat)
            frame_scale_scores_ = self.get_unnormalized_frame_scale_scores(frame_query, encoded_frame_feat, query_mask, video_mask)

            ####################################################################
            ############### probabilistic embedding modeling part ##############

            support_query = self.support_set(label_dict, encoded_frame_feat, frame_query)
            support_query_mask = self.support_set(label_dict, video_mask, query_mask)
            support_query_pooled = self.mean_pooling_for_sequence(support_query,support_query_mask)

            visual_pooled = self.mean_pooling_for_norm_visual(encoded_frame_feat, video_mask)

            # prob_text = self.probabilistic_text(video_query, frame_query)
            prob_text = self.probabilistic_text(support_query_pooled, support_query)
            prob_text_embedding = prob_text['embedding']       # b n 512
            prob_text_logsigma = prob_text['logsigma']   # bs 512
            prob_text_mean = prob_text['mean']       # bs 512

            # prob_video = self.probabilistic_video(visual_pooled, align_frame)
            prob_video = self.probabilistic_video(visual_pooled, encoded_frame_feat)
            prob_video_embedding = prob_video['embedding']  # Sample m embeddings from the distribution.
            prob_video_logsigma = prob_video['logsigma']       
            prob_video_mean = prob_video['mean']

            ############### MILNCELoss_(Supervised Contrastice Loss) ##############

            bs = prob_video_embedding.size(0)
            n_video = self.n_video_samples
            n_text = self.n_text_samples
            dim = prob_video_embedding.size(-1)

            "In prob_video_embedding, the tensor with shape b × n × 512 is reshaped to (b×n) × 512 using view(-1, dim)."
            prob_sim_matrix_from_v = torch.einsum('ad,bd->ab', [prob_video_embedding.view(-1, dim), prob_text_embedding.view(-1, dim)])
            MIL_loss_v = self.loss_MIL_fct(prob_sim_matrix_from_v, bs, n_video, n_text)

            prob_sim_matrix_from_t = torch.einsum('ad,bd->ab', [prob_text_embedding.view(-1, dim), prob_video_embedding.view(-1, dim)])     # 与.t()等价
            MIL_loss_t = self.loss_MIL_fct(prob_sim_matrix_from_t, bs, n_video, n_text)
            MIL_loss = (MIL_loss_v + MIL_loss_t) / 2

            #######  Eliminate the uncertainty of many-to-many matching by means of probabilistic modeling and then using the MIL loss as the loss function.  ########
            vib_loss = self.vib_loss(prob_video_embedding, prob_video_logsigma, prob_text_embedding, prob_text_logsigma)
              
            return clip_scale_scores, clip_scale_scores_, frame_scale_scores, frame_scale_scores_, MIL_loss, vib_loss
        else:

            return clip_scale_scores, frame_scale_scores


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)
