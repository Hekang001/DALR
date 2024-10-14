import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
import torch.nn.functional as F
import math
import numpy as np
from transformers import AutoModel, AutoConfig
from sentence_transformers import util
from model.cross_net import CrossSparseAggrNet_v2
from model.patch_word_align import compute_local_contrastive_loss, patchWordAlignment
from .teachers import Teacher
import copy


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)   # non-linear activation
        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
    
    
class ArcSimilarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp, margin=0.05):
        super().__init__()
        self.temp = temp
        self.margin = margin
        self.cos = nn.CosineSimilarity(dim=-1)
        
    def calculate_arccos1(self, cos_sim, labels=None):
        theta = torch.acos(torch.clamp(cos_sim, -1, 1))
        
        if labels is None:
            labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
        
        num_classes = labels.max().item() + 1
        one_hot_labels = F.one_hot(labels, num_classes)
        
        selected_labels = torch.where(
            torch.gt(theta, math.pi - self.margin),
            torch.zeros_like(one_hot_labels),one_hot_labels)    
        
        
        final_theta = torch.where(selected_labels.bool(),
                                    theta + self.margin,
                                    theta)
        
        return torch.cos(final_theta)
    
    def calculate_arccos2(self, cos_sim, labels=None, slabels=None):
        theta = torch.acos(torch.clamp(cos_sim, -1, 1))
        
        if labels is None:
            labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
        num_classes = labels.max().item() + 1
        one_hot_labels = F.one_hot(labels, num_classes)
        
        selected_labels = torch.where(
            torch.gt(theta, self.margin),
            torch.ones_like(one_hot_labels),one_hot_labels) * torch.abs(one_hot_labels - 1)
        
        if slabels is None:
            final_theta = torch.where(selected_labels.bool(),
                                    theta - self.margin,
                                    theta)
            
        else:
            final_theta = torch.where(selected_labels.bool(),
                                    theta - (1-slabels)*self.margin,
                                    theta)
            
        return torch.cos(final_theta)

    def forward(self, x, y, slabels=None):
        return self.calculate_arccos2(self.cos(x, y), slabels=slabels) / self.temp


class ListNet(nn.Module):
    """
    ListNet objective for ranking distillation; minimizes the cross entropy between permutation [top-1] probability distribution and ground truth obtained from teacher
    """
    def __init__(self, tau, gamma_):
        super(ListNet, self).__init__()
        self.teacher_temp_scaled_sim = Similarity(tau / 2)
        self.student_temp_scaled_sim = Similarity(tau)
        self.gamma_ = gamma_

    def forward(self, teacher_top1_sim_pred, student_top1_sim_pred):
        p = F.log_softmax(student_top1_sim_pred.fill_diagonal_(float('-inf')), dim=-1)
        q = F.softmax(teacher_top1_sim_pred.fill_diagonal_(float('-inf')), dim=-1)
        loss = -(q*p).nansum()  / q.nansum()
        return self.gamma_ * loss 

class ListMLE(nn.Module):
    """
    ListMLE objective for ranking distillation; maximizes the liklihood of the ground truth permutation (sorted indices of the ranking lists obtained from teacher) 
    """
    def __init__(self, tau, gamma_):
        super(ListMLE, self).__init__()
        self.temp_scaled_sim = Similarity(tau)
        self.gamma_ = gamma_ 
        self.eps = 1e-7

    def forward(self, teacher_top1_sim_pred, student_top1_sim_pred):

        y_pred = student_top1_sim_pred 
        y_true = teacher_top1_sim_pred

        # shuffle for randomised tie resolution
        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)
        mask = y_true_sorted == -1
        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
        preds_sorted_by_true[mask] = float('-inf')
        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
        observation_loss = torch.log(cumsums + self.eps) - preds_sorted_by_true_minus_max
        observation_loss[mask] = 0.0

        return self.gamma_ * torch.mean(torch.sum(observation_loss, dim=1))


class AlignLoss(nn.Module):

    def __init__(self, args, margin=0.2, max_violation=False):
        super(AlignLoss, self).__init__()
        self.args = args
        self.margin = margin
        self.max_violation = max_violation
        self.mask_repeat = True

        self.false_hard = []

    def max_violation_on(self):
        self.max_violation = True

    def max_violation_off(self):
        self.max_violation = False

    def forward(self, im, s, img_ids=None, scores=None):

        # compute image-sentence score matrix
        if scores is None:
            scores = im.mm(s.t()) 
        
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval, i->t
        cost_s = (self.margin + scores - d1).clamp(min=0)

        # compare every diagonal score to scores in its row
        # image retrieval t->i
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        if not self.mask_repeat:
            mask = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)
        else:
            mask = (img_ids.unsqueeze(1) == img_ids.unsqueeze(0))
            # repeat = len(img_ids) - len(torch.unique(img_ids))

        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s, idx_s = cost_s.max(1)
            cost_im, idx_im = cost_im.max(0)

        loss = cost_s.sum() + cost_im.sum()

        return loss


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.args = model_kargs['model_args']
        self.bert = BertModel(config)
        self.pooler = MLPLayer(config.hidden_size, config.hidden_size)
        self.init_weights()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        pooler_output = self.pooler(outputs.last_hidden_state[:, 0])

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )


class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.args = model_kargs['model_args']
        self.roberta = RobertaModel(config)
        self.pooler = MLPLayer(config.hidden_size, config.hidden_size)
        self.init_weights()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        pooler_output = self.pooler(outputs.last_hidden_state[:, 0])

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )


class ResNetVisnModel(nn.Module):
    def __init__(self, feature_dim,  proj_dim):
        super().__init__()
        self.mlp = MLPLayer(feature_dim, proj_dim) # visual features -> grounding space

    def forward(self, x):
        x = self.mlp(x)
        x = x / x.norm(2, dim=-1, keepdim=True)
        return x
    

class ClipVisnModel(nn.Module):
    def __init__(self, feature_dim,  proj_dim):
        super().__init__()
        self.vmlp = nn.Linear(feature_dim, proj_dim)  # visual features -> grounding space
        self.tmlp = nn.Linear(feature_dim, proj_dim) # textual features -> grounding space
        self.logit_scale = torch.tensor(np.log(1 / 0.05))
        self.loss_fct = nn.CrossEntropyLoss()

    def logit(self, image_features, text_features):
        device = image_features.device
        
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        #logits_per_image, logits_per_text = self.logit(images, texts)
        ground_truth = torch.arange(logits_per_image.size(0)).to(device)
        total_loss = (self.loss_fct(logits_per_image,ground_truth) + self.loss_fct(logits_per_text,ground_truth))/2
        
        return total_loss

    def forward(self, visn_feat, text_feat):
        visn_feat = self.vmlp(visn_feat)
        visn_feat = visn_feat / visn_feat.norm(2, dim=-1, keepdim=True)
        
        #text_feat = self.vmlp(text_feat) 2
        text_feat = self.tmlp(text_feat)
        text_feat = text_feat / text_feat.norm(2, dim=-1, keepdim=True)
        
        return visn_feat, text_feat, None #self.logit(visn_feat, text_feat)
    


class MSE(nn.Module):
    def __init__(self, lang_model, visn_model, args, yaml_config:dict):
        super().__init__()
        self.args = args
        self.device = args.device
        # self.tau=self.args.tau
        self.lang_model = lang_model
        self.visn_model = visn_model
        self.vision_model_m = ClipVisnModel(args.hidden_size, args.proj_dim)
        # self.teacher_model = teacher_model
        # self.sparse_ratio = 0.8
        # self.aggr_ratio = 0.4
        self.visn_text_clip_proj = ClipVisnModel(args.hidden_size, args.proj_dim)
        self.embed_dim = args.hidden_size
        self.proj_dim = args.proj_dim
        self.loss_config = yaml_config['loss_config']
        if self.args.distillation_loss == "listnet":
            self.distillation_loss_fct = ListNet(self.args.tau2, self.args.gamma_)
        elif self.args.distillation_loss == "listmle":
            self.distillation_loss_fct = ListMLE(self.args.tau2, self.args.gamma_)
        else:
            raise NotImplementedError
        
        # self.cross_net = CrossSparseAggrNet_v2(args=args)

        self.teacher_proj = ClipVisnModel(args.hidden_size, args.proj_dim)
        self.grounding = MLPLayer(args.hidden_size, args.proj_dim)

        self.vision_proj = nn.Linear(args.vision_width,args.proj_dim)
        self.text_proj = nn.Linear(args.hidden_size, args.proj_dim)
        
        self.sim_vl = ArcSimilarity(temp=self.args.temp, margin=args.margin1)
        self.sim = Similarity(temp=self.args.temp)
        self.contrastive =  AlignLoss(args, margin=0.2, max_violation=True)
        # self.sim_vl = ArcSimilarity(temp=self.args.temp_vl, margin=args.margin2)
        
        self.loss_fct = nn.CrossEntropyLoss()

        if self.args.cross_softlabel:
            self.ln_cross_image_projection = nn.LayerNorm(self.embed_dim)
            self.ln_cross_text_projection = nn.LayerNorm(self.embed_dim)
            self.cross_image_projection = nn.Linear(self.embed_dim, self.proj_dim)
            self.cross_text_projection = nn.Linear(self.embed_dim, self.proj_dim)
        if self.args.uni_softlabel:
            self.ln_uni_image_projection = nn.LayerNorm(self.embed_dim)
            self.ln_uni_text_projection = nn.LayerNorm(self.embed_dim)
            self.uni_image_projection = nn.Linear(self.embed_dim, self.proj_dim)
            self.uni_text_projection = nn.Linear(self.embed_dim, self.proj_dim)
        

        # set tau
        if self.args.contrastive:
            self.__init_tau = self.loss_config['contrastive']['tau']
            self.tau = nn.Parameter(torch.tensor(self.__init_tau, device=self.device))

        if self.args.cross_softlabel:
            self.__init_cross_image_tau = self.loss_config['cross_softlabel']['image_tau']
            self.__init_cross_text_tau = self.loss_config['cross_softlabel']['text_tau']
            self.__init_cross_tau = (self.__init_cross_image_tau + self.__init_cross_text_tau) / 2.0
            self.__init_cross_the_softlabel_image_tau = self.loss_config['cross_softlabel']['the_softlabel_image_tau']
            self.__init_cross_the_softlabel_text_tau = self.loss_config['cross_softlabel']['the_softlabel_text_tau']
            self.__init_cross_the_softlabel_tau = (self.__init_cross_the_softlabel_image_tau + self.__init_cross_the_softlabel_text_tau) / 2.0
            if self.is_each_cross_soft_mode():
                if self.loss_config['cross_softlabel']['use_same_tau']:
                    self.cross_tau = nn.Parameter(torch.tensor(self.__init_cross_tau, device=self.device))
                else:
                    self.cross_tau_image = nn.Parameter(torch.tensor(self.__init_cross_image_tau, device=self.device))
                    self.cross_tau_text = nn.Parameter(torch.tensor(self.__init_cross_text_tau, device=self.device))
                if self.loss_config['cross_softlabel']['use_same_softlabel_tau']:
                    self.cross_the_softlabel_tau = nn.Parameter(torch.tensor(self.__init_cross_the_softlabel_tau, device=self.device))
                else:
                    self.cross_the_softlabel_tau_image = nn.Parameter(torch.tensor(self.__init_cross_the_softlabel_image_tau, device=self.device))
                    self.cross_the_softlabel_tau_text = nn.Parameter(torch.tensor(self.__init_cross_the_softlabel_text_tau, device=self.device))
            else:
                self.cross_tau = nn.Parameter(torch.tensor(self.__init_cross_tau, device=self.device))
                self.cross_the_softlabel_tau = nn.Parameter(torch.tensor(self.__init_cross_the_softlabel_tau, device=self.device))

        if self.is_mode_on("uni_softlabel"):
            self.__init_uni_image_tau = self.loss_config['uni_softlabel']['image_tau']
            self.__init_uni_text_tau = self.loss_config['uni_softlabel']['text_tau']
            self.__init_uni_tau = (self.__init_uni_image_tau + self.__init_uni_text_tau) / 2.0
            self.__init_uni_the_softlabel_image_tau = self.loss_config['uni_softlabel']['the_softlabel_image_tau']
            self.__init_uni_the_softlabel_text_tau = self.loss_config['uni_softlabel']['the_softlabel_text_tau']
            self.__init_uni_the_softlabel_tau = (self.__init_uni_the_softlabel_image_tau + self.__init_uni_the_softlabel_text_tau) / 2.0
            
            if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_tau']:
                if self.loss_config['uni_softlabel']['use_same_tau']:
                    self.uni_tau = nn.Parameter(torch.tensor(self.__init_uni_tau, device=self.device))
                else:
                    self.uni_tau_image = nn.Parameter(torch.tensor(self.__init_uni_image_tau, device=self.device))
                    self.uni_tau_text = nn.Parameter(torch.tensor(self.__init_uni_text_tau, device=self.device))

            if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_softlabel_tau']:
                if self.loss_config['uni_softlabel']['use_same_softlabel_tau']:
                    self.uni_the_softlabel_tau = nn.Parameter(torch.tensor(self.__init_uni_the_softlabel_tau, device=self.device))
                else:
                    self.uni_the_softlabel_tau_image = nn.Parameter(torch.tensor(self.__init_uni_the_softlabel_image_tau, device=self.device))
                    self.uni_the_softlabel_tau_text = nn.Parameter(torch.tensor(self.__init_uni_the_softlabel_text_tau, device=self.device))

        self.initialize_parameters()

        
        self.using_threshhold = args.using_threshhold
        if self.using_threshhold:
            print("USING THRESHOLD")

    
    def is_mode_on(self, modeName: str) -> bool:
        return self.loss_config[modeName]['is_on']

    
    def is_mean_contrastive_loss_mode(self, lossName):
        return self.loss_config[lossName]['contrastive_loss_mode'] == "mean"

    def is_sum_contrastive_loss_mode(self, lossName):
        return self.loss_config[lossName]['contrastive_loss_mode'] == "sum"
    
    def is_each_cross_soft_mode(self):
        """check if each softlabel"""
        return self.args.cross_softlabel and self.loss_config['cross_softlabel']['cross_softlabel_mode'] == "each"

    
    def is_add_cross_soft_mode(self):
        """check if add softlabel"""
        return self.args.cross_softlabel and self.loss_config['cross_softlabel']['cross_softlabel_mode'] == "add"

    def is_dot_cross_soft_mode(self):
        """check if dot softlabel"""
        return self.args.cross_softlabel and self.loss_config['cross_softlabel']['cross_softlabel_mode'] == "dot"


    def initialize_parameters(self):
        """Initialize the model parameters."""
        if self.is_mode_on('contrastive') or self.is_mode_on('cross_softlabel'):
            nn.init.normal_(self.cross_image_projection.weight, std=0.02)
            nn.init.normal_(self.cross_text_projection.weight, std=0.02)
        if self.is_mode_on('uni_softlabel'):
            nn.init.normal_(self.uni_image_projection.weight, std=0.02)
            nn.init.normal_(self.uni_text_projection.weight, std=0.02)

        if self.is_mode_on('contrastive'):
            if self.loss_config['contrastive']['is_block_tau']:
                self.tau.requires_grad_(False)

        if self.is_mode_on('cross_softlabel'):
            if self.loss_config['cross_softlabel']['is_block_tau']:
                if hasattr(self, "cross_tau"):
                    self.cross_tau.requires_grad_(False)
                else:
                    self.cross_tau_image.requires_grad_(False)
                    self.cross_tau_text.requires_grad_(False)
            if self.loss_config['cross_softlabel']['is_block_softlabel_tau']:
                if hasattr(self, "cross_the_softlabel_tau"):
                    self.cross_the_softlabel_tau.requires_grad_(False)
                else:
                    self.cross_the_softlabel_tau_image.requires_grad_(False)
                    self.cross_the_softlabel_tau_text.requires_grad_(False)

        if self.is_mode_on('uni_softlabel'):
            if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_tau']:
                if self.loss_config['uni_softlabel']['is_block_tau']:
                    if hasattr(self, "uni_tau"):
                        self.uni_tau.requires_grad_(False)
                    else:
                        self.uni_tau_image.requires_grad_(False)
                        self.uni_tau_text.requires_grad_(False)
            if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_softlabel_tau']:
                if self.loss_config['uni_softlabel']['is_block_softlabel_tau']:
                    if hasattr(self, "uni_the_softlabel_tau"):
                        self.uni_the_softlabel_tau.requires_grad_(False)
                    else:
                        self.uni_the_softlabel_tau_image.requires_grad_(False)
                        self.uni_the_softlabel_tau_text.requires_grad_(False)

    @torch.no_grad()
    def clamp_tau(self):
        # clip tau to prevent overflow
        if self.is_mode_on("contrastive"):
            self.tau.clamp_(min=self.loss_config['contrastive']['tau_min'], max=self.loss_config['contrastive']['tau_max'])

        if self.is_mode_on("cross_softlabel"):
            if hasattr(self, "cross_tau"):
                self.cross_tau.clamp_(min=(self.loss_config['cross_softlabel']['image_tau_min']+self.loss_config['cross_softlabel']['text_tau_min'])/2.0,
                                      max=(self.loss_config['cross_softlabel']['image_tau_max']+self.loss_config['cross_softlabel']['text_tau_max'])/2.0)
            else:
                self.cross_tau_image.clamp_(min=self.loss_config['cross_softlabel']['image_tau_min'],
                                            max=self.loss_config['cross_softlabel']['image_tau_max'])
                self.cross_tau_text.clamp_(min=self.loss_config['cross_softlabel']['text_tau_min'],
                                           max=self.loss_config['cross_softlabel']['text_tau_max'])
            if hasattr(self, "cross_the_softlabel_tau"):
                self.cross_the_softlabel_tau.clamp_(min=(self.loss_config['cross_softlabel']['the_softlabel_image_tau_min']+self.loss_config['cross_softlabel']['the_softlabel_text_tau_min'])/2.0,
                                                    max=(self.loss_config['cross_softlabel']['the_softlabel_image_tau_max']+self.loss_config['cross_softlabel']['the_softlabel_text_tau_max'])/2.0)
            else:
                self.cross_the_softlabel_tau_image.clamp_(min=self.loss_config['cross_softlabel']['the_softlabel_image_tau_min'],
                                                          max=self.loss_config['cross_softlabel']['the_softlabel_image_tau_max'])
                self.cross_the_softlabel_tau_text.clamp_(min=self.loss_config['cross_softlabel']['the_softlabel_text_tau_min'],
                                                         max=self.loss_config['cross_softlabel']['the_softlabel_text_tau_max'])

        if self.is_mode_on("uni_softlabel"):
            if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_tau']:
                if hasattr(self, "uni_tau"):
                    self.uni_tau.clamp_(min=(self.loss_config['uni_softlabel']['image_tau_min']+self.loss_config['uni_softlabel']['text_tau_min'])/2.0,
                                        max=(self.loss_config['uni_softlabel']['image_tau_max']+self.loss_config['uni_softlabel']['text_tau_max'])/2.0)
                else:
                    self.uni_tau_image.clamp_(min=self.loss_config['uni_softlabel']['image_tau_min'],
                                            max=self.loss_config['uni_softlabel']['image_tau_max'])
                    self.uni_tau_text.clamp_(min=self.loss_config['uni_softlabel']['text_tau_min'],
                                            max=self.loss_config['uni_softlabel']['text_tau_max'])
            if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_softlabel_tau']:
                if hasattr(self, "uni_the_softlabel_tau"):
                    self.uni_the_softlabel_tau.clamp_(min=(self.loss_config['uni_softlabel']['the_softlabel_image_tau_min']+self.loss_config['uni_softlabel']['the_softlabel_text_tau_min'])/2.0,
                                                      max=(self.loss_config['uni_softlabel']['the_softlabel_image_tau_max']+self.loss_config['uni_softlabel']['the_softlabel_text_tau_max'])/2.0)
                else:
                    self.uni_the_softlabel_tau_image.clamp_(min=self.loss_config['uni_softlabel']['the_softlabel_image_tau_min'],
                                                            max=self.loss_config['uni_softlabel']['the_softlabel_image_tau_max'])
                    self.uni_the_softlabel_tau_text.clamp_(min=self.loss_config['uni_softlabel']['the_softlabel_text_tau_min'],
                                                           max=self.loss_config['uni_softlabel']['the_softlabel_text_tau_max'])
    
        
    def _encode_image_features(self, image_features, cross_modal=True):
        """encode from clip model"""
        if cross_modal and self.args.cross_softlabel:
            # image_features = self.ln_cross_image_projection(image_features)
            image_features = self.cross_image_projection(image_features)
        elif (not cross_modal) and self.args.uni_softlabel:
            # image_features = self.ln_uni_image_projection(image_features)
            image_features = self.uni_image_projection(image_features)
        return image_features
    
    def _encode_text_features(self, text_features, cross_modal=True):
        """encode from clip model"""
        if cross_modal and self.args.cross_softlabel:
            # text_features = self.ln_cross_text_projection(text_features)
            text_features = self.cross_text_projection(text_features)
        elif (not cross_modal) and self.args.uni_softlabel:
            # text_features = self.ln_uni_text_projection(text_features)
            text_features = self.uni_text_projection(text_features)
        return text_features

    def get_similarity(self, image_features, text_features, cross_modal=True):
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        if cross_modal:
            """if cross-modal alignment, return the similarity between image and text"""
            logits_per_image = image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            return logits_per_image, logits_per_text
        else:
            """if uni-modal alignment, return the similarity between image and image, text and text"""
            logits_image_image = image_features @ image_features.t()
            logits_text_text = text_features @ text_features.t()
            return logits_image_image, logits_text_text

    def ContrastiveLoss(self, logits_per_image, logits_per_text, idx=None):
        # contrastive loss
        if idx is None:
            sim_targets = torch.eye(logits_per_image.shape[0], device=self.device)
        else:
            idx = idx.view(-1, 1)
            pos_idx = torch.eq(idx, idx.t()).float()
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        if self.args.contrastive_loss_mode == "mean" and self.is_mean_contrastive_loss_mode("contrastive"):
            loss_i2t = -torch.mean(F.log_softmax(logits_per_image / self.tau, dim=1) * sim_targets, dim=1).mean()
            loss_t2i = -torch.mean(F.log_softmax(logits_per_text / self.tau, dim=1) * sim_targets, dim=1).mean()
        elif self.args.contrastive_loss_mode == "sum" and self.is_sum_contrastive_loss_mode("contrastive"):
            loss_i2t = -torch.sum(F.log_softmax(logits_per_image / self.tau, dim=1) * sim_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits_per_text / self.tau, dim=1) * sim_targets, dim=1).mean()
        else:
            raise ValueError("contrastive loss mode error")
        contrastive_loss = loss_i2t + loss_t2i

        return contrastive_loss

    def KLContrastiveSimLoss(self, logits, softlabel, tau, softlabel_tau, use_loss="kl"):
        """
        KL divergence loss
        make logits and softlabel have the same distribution
        logits to softlabel
        """
        # softmax for softlabel
        sim_targets = F.softmax(softlabel / softlabel_tau, dim=1)

        # log softmax
        logit_inputs = F.log_softmax(logits / tau, dim=1)

        if use_loss == "kl":
            # KL divergence
            loss = F.kl_div(logit_inputs, sim_targets, reduction='batchmean')
        elif use_loss == "contrastive":
            # Switch to the same loss as ContrastiveLoss, but sim_targets is soft
            if self.args.contrastive_loss_mode == "mean":
                loss = -torch.mean(logit_inputs * sim_targets, dim=1).mean()
            elif self.args.contrastive_loss_mode == 'sum':
                loss = -torch.sum(logit_inputs * sim_targets, dim=1).mean()
            else:
                raise ValueError("contrastive loss mode error")
        else:
            raise ValueError("loss mode error")

        return loss    
    


    def forward(self, batch, cal_inter=False):

        teacher = None
        if self.args.second_teacher_name_or_path is None:
            teacher_pooler = ("cls_before_pooler" if ("simcse" in self.args.first_teacher_name_or_path or "RankCSE" in self.args.first_teacher_name_or_path) else "avg")
            teacher = Teacher(model_name_or_path=self.args.first_teacher_name_or_path, pooler=teacher_pooler)
        else:
            first_pooler = ("cls_before_pooler" if ("simcse" in self.args.first_teacher_name_or_path or "RankCSE" in self.args.first_teacher_name_or_path) else "avg")
            first_teacher = Teacher(model_name_or_path=self.args.first_teacher_name_or_path, pooler=first_pooler)
            second_pooler = ("cls_before_pooler" if ("simcse" in self.args.second_teacher_name_or_path or "RankCSE" in self.args.second_teacher_name_or_path) else "avg")
            second_teacher = Teacher(model_name_or_path=self.args.second_teacher_name_or_path, pooler=second_pooler)

        with torch.no_grad():

            # Read batch inputs
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            token_type_ids = None
            if "token_type_ids" in batch:
                token_type_ids = batch["token_type_ids"]

            batch_size = input_ids.size(0)
            num_sent = input_ids.size(1)

            # Flatten input for encoding by the teacher - (bsz * num_sent, len)
            input_ids = input_ids.view((-1, input_ids.size(-1))) 
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) 
            attention_mask = attention_mask.view((-1, attention_mask.size(-1)))

            teacher_inputs = copy.deepcopy(batch)
            teacher_inputs["input_ids"] = input_ids
            teacher_inputs["attention_mask"] = attention_mask
            teacher_inputs["token_type_ids"] = token_type_ids

            # Encode, unflatten, and pass to student
            if teacher is not None: 
                # Single teacher
                embeddings = teacher.encode(teacher_inputs)
                embeddings = embeddings.view((batch_size, num_sent, -1))

                z1T, z2T = embeddings[:,0], embeddings[:,1]

                if num_sent==3:
                    z3T = embeddings[:,2]
                
                if self.args.fp16:
                    if num_sent==3:
                        z3T = z3T.to(torch.float16)
                        
                    z1T = z1T.to(torch.float16)
                    z2T = z2T.to(torch.float16)

                cos = nn.CosineSimilarity(dim=-1)

                teacher_top1_sim_pred = cos(z1T.unsqueeze(1), z2T.unsqueeze(0)) / self.args.tau2
                if num_sent == 3:
                    z1_z3_cos = cos(z1T.unsqueeze(1), z3T.unsqueeze(0)) / self.args.tau2
                    teacher_top1_sim_pred = torch.cat([teacher_top1_sim_pred, z1_z3_cos], 1)
           
            else:
                # Weighted average of two teachers
                embeddings1 = first_teacher.encode(teacher_inputs)
                embeddings2 = second_teacher.encode(teacher_inputs)
                embeddings1 = embeddings1.view((batch_size, num_sent, -1))
                embeddings2 = embeddings2.view((batch_size, num_sent, -1))
                first_teacher_z1, first_teacher_z2 = embeddings1[:,0], embeddings1[:,1]
                second_teacher_z1, second_teacher_z2 = embeddings2[:,0], embeddings2[:,1]
                
                if num_sent==3:
                    first_teacher_z3 = embeddings1[:,2]
                    second_teacher_z3 = embeddings2[:,2]

                if self.args.fp16:
                    first_teacher_z1 = first_teacher_z1.to(torch.float16)
                    first_teacher_z2 = first_teacher_z2.to(torch.float16)
                    second_teacher_z1 = second_teacher_z1.to(torch.float16)
                    second_teacher_z2 = second_teacher_z2.to(torch.float16)

                    if num_sent==3:
                        first_teacher_z3 = first_teacher_z3.to(torch.float16)
                        second_teacher_z3 = second_teacher_z3.to(torch.float16)

                cos = nn.CosineSimilarity(dim=-1)
                first_teacher_top1_sim = cos(first_teacher_z1.unsqueeze(1), first_teacher_z2.unsqueeze(0)) / self.args.tau2
                second_teacher_top1_sim = cos(second_teacher_z1.unsqueeze(1), second_teacher_z2.unsqueeze(0)) / self.args.tau2
                if num_sent == 3:
                    first_teacher_z1_z3_cos = cos(first_teacher_z1.unsqueeze(1), first_teacher_z3.unsqueeze(0)) / self.args.tau2
                    second_teacher_z1_z3_cos = cos(second_teacher_z1.unsqueeze(1), second_teacher_z3.unsqueeze(0)) / self.args.tau2
                    first_teacher_top1_sim = torch.cat([first_teacher_top1_sim, first_teacher_z1_z3_cos], 1)
                    second_teacher_top1_sim = torch.cat([second_teacher_top1_sim, second_teacher_z1_z3_cos], 1)
                teacher_top1_sim_pred = (self.args.alpha_ * first_teacher_top1_sim) + ((1.0 - self.args.alpha_) * second_teacher_top1_sim)



        self.clamp_tau()

        lang_output = self.lang_model(input_ids=batch['input_ids'],
                                      attention_mask=batch['attention_mask'],
                                      token_type_ids=batch['token_type_ids'] if 'position_ids' in batch.keys() else None,
                                      position_ids=batch['position_ids'] if 'position_ids' in batch.keys() else None)

        batch_size = batch['input_ids'].size(0)
        num_sent = batch['input_ids'].size(1)
        seq_length = lang_output.last_hidden_state.size(1)
        reshape_lang_output = lang_output.last_hidden_state.view(batch_size, num_sent, seq_length,-1)
        cap_emb = reshape_lang_output[:,0,:,:]

        # [bs*2, hidden] -> [bs, 2, hidden]
        lang_pooled_output = lang_output.last_hidden_state[:, 0].view((batch_size, num_sent, -1))
        lang_projection = lang_output.pooler_output.view((batch_size, num_sent, -1))  # [bs, 2,  hidden],  output of additional MLP layer

        # Separate representation
        z1, z2 = lang_projection[:, 0], lang_projection[:, 1]  # (bs, hidden)
        # cap_emb = self.text_proj(lang_output.last_hidden_state)
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))  # (bs, bs)

        labels = torch.arange(cos_sim.size(0)).long().to(self.args.device)  # [0, 1, bs-1]  (bs)
        
        contra_loss = self.loss_fct(cos_sim, labels)  # unsup: bs-1 negatives

        sentence_features =  lang_output.last_hidden_state.view(batch_size, num_sent, seq_length, -1)  # [bs, seq, hidden]
        sentence_global_feat = self.text_proj(sentence_features[:,0])


        if not cal_inter:
                
            return contra_loss
        
        else:

           
            v, t, _ = self.vision_model_m(batch['img'], batch['clip_text_feat'])  # [bs, proj_dim]
            
            l2v_proj = self.grounding(lang_pooled_output)  # [bs, 2, proj_dim],  output for vision grounding
            l2v_proj = l2v_proj / l2v_proj.norm(2, dim=-1, keepdim=True)

            p1, p2 = l2v_proj[:, 0], l2v_proj[:, 1]  # (bs, proj)
            cos_sim_p0_v = self.sim_vl(p1.unsqueeze(1), v.unsqueeze(0), slabels=batch['vv_scores'])  # (bs, bs)
            cos_sim_p1_v = self.sim_vl(p2.unsqueeze(1), v.unsqueeze(0), slabels=batch['vv_scores'])
            
            
            p1, p2 = l2v_proj[:, 0], l2v_proj[:, 1]  # (bs, proj)
            cos_sim_p0_t = self.sim_vl(p1.unsqueeze(1), t.unsqueeze(0), slabels=batch['cc_scores'])  # (bs, bs)
            cos_sim_p1_t = self.sim_vl(p2.unsqueeze(1), t.unsqueeze(0), slabels=batch['cc_scores'])
            
            if self.using_threshhold:
                cos_sim_p0_v = cos_sim_p0_v + batch['cv_slabels']
                cos_sim_p1_v = cos_sim_p1_v + batch['cv_slabels'] 
                cos_sim_p0_t = cos_sim_p0_t + batch['vc_slabels']
                cos_sim_p1_t = cos_sim_p1_t + batch['vc_slabels'] 
            
            inter_loss1 = (self.loss_fct(cos_sim_p0_v, labels) + self.loss_fct(cos_sim_p1_v, labels)) / 2
            inter_loss2 = (self.loss_fct(cos_sim_p0_t, labels) + self.loss_fct(cos_sim_p1_t, labels)) / 2

            inter_loss = inter_loss1 + inter_loss2

            # return loss, inter_loss

            teacher_image_features =batch['img']
            teacher_text_features = batch['clip_text_feat']
                      
            image_embeds = self.visn_model(batch['image']) 

            image_global_feat = F.normalize(image_embeds, dim=-1)
            image_global_feat = self.vision_proj(F.normalize(image_embeds, dim=-1))
            # print("========image_global_feat.size()==========")
            # print(image_global_feat.size())
            
            # print("========sentence_global_feat.size()==========")
            # print(sentence_global_feat.size())
            
            # S_prime, P_prime = patchWordAlignment(image_global_feat, sentence_global_feat,tau=0.05)
            # add_loss = compute_local_contrastive_loss(S_prime, P_prime, sentence_global_feat, image_global_feat)

            image_feat = F.normalize(image_embeds[:,0,:],dim=-1) 
            text_features = lang_projection[:, 0] # [bs, hidden]

            if self.args.contrastive or self.args.cross_softlabel:
                cross_image_features, cross_text_features= self._encode_image_features(
                    image_feat, cross_modal=True), self._encode_text_features(text_features, cross_modal=True)
                # 类似于软标签的做法
                logits_per_image, logits_per_text = self.get_similarity(cross_image_features, cross_text_features, cross_modal=True)
            

            if self.args.uni_softlabel:
                uni_image_features, uni_text_features = self._encode_image_features(
                    image_feat, cross_modal=False), self._encode_text_features(text_features, cross_modal=False)
                # 类似于软标签的做法
                logits_image_image, logits_text_text = self.get_similarity(uni_image_features, uni_text_features, cross_modal=False)
            
            if self.args.cross_softlabel or self.args.uni_softlabel:
                with torch.no_grad():


                    softlabel_image_sim = util.cos_sim(teacher_image_features, teacher_image_features)
                    softlabel_text_sim = util.cos_sim(teacher_text_features, teacher_text_features)
                  
                    if self.is_mode_on("cross_softlabel"): 
                        if self.is_add_cross_soft_mode():
                            # Average two similarities
                            softlabel_all_sim = (softlabel_image_sim + softlabel_text_sim) / 2.0
                        elif self.is_dot_cross_soft_mode():
                            # Dot two similarities
                            softlabel_image_sim_copy = softlabel_image_sim.clone()
                            softlabel_text_sim_copy = softlabel_text_sim.clone()
                            softlabel_image_sim_copy[softlabel_image_sim_copy < 0.0] = 0.0
                            softlabel_text_sim_copy[softlabel_text_sim_copy < 0.0] = 0.0
                            softlabel_all_sim = softlabel_image_sim_copy * softlabel_text_sim_copy
                            softlabel_all_sim = torch.sqrt(softlabel_all_sim)
                        elif self.is_each_cross_soft_mode():
                            pass
                        else:         
                            raise ValueError("softlabel mode error")

            cross_modal_loss, uni_modal_loss, contrastive_loss = torch.tensor(0.0, device=self.device), torch.tensor(
                0.0, device=self.device), torch.tensor(0.0, device=self.device)
            
            if self.args.cross_softlabel:
                # for cross-modal alignment (similarity)
                # image-text and image-image softlabel
                # text-image and text-text softlabel
                softlabel_image_sim_loss = softlabel_image_sim
                softlabel_text_sim_loss = softlabel_text_sim
                if not self.is_each_cross_soft_mode():
                    softlabel_image_sim_loss = softlabel_all_sim
                    softlabel_text_sim_loss = softlabel_all_sim

                if hasattr(self, "cross_tau"):
                    cross_tau_loss_image = self.cross_tau
                    cross_tau_loss_text = self.cross_tau
                else:
                    cross_tau_loss_image = self.cross_tau_image
                    cross_tau_loss_text = self.cross_tau_text

                if hasattr(self, "cross_the_softlabel_tau"):
                    cross_the_softlabel_tau_loss_image = self.cross_the_softlabel_tau
                    cross_the_softlabel_tau_loss_text = self.cross_the_softlabel_tau
                else:
                    cross_the_softlabel_tau_loss_image = self.cross_the_softlabel_tau_image
                    cross_the_softlabel_tau_loss_text = self.cross_the_softlabel_tau_text

                teacher_image_features = self.grounding(teacher_image_features)
                teacher_text_features = self.grounding(teacher_text_features)

                cos_sim_image2image = self.sim_vl(cross_image_features.unsqueeze(1), teacher_image_features.unsqueeze(0), slabels=softlabel_text_sim)  # (bs, bs)
                cos_sim_image2text = self.sim_vl(cross_image_features.unsqueeze(1), teacher_text_features.unsqueeze(0), slabels=softlabel_text_sim)  # (bs, bs)
                
                cos_sim_text2image_1 = self.sim_vl(cross_text_features.unsqueeze(1), teacher_image_features.unsqueeze(0),  slabels=softlabel_text_sim)  # (bs, bs)
                cos_sim_text2text_1 = self.sim_vl(cross_text_features.unsqueeze(1), teacher_text_features.unsqueeze(0), slabels=softlabel_text_sim)  # (bs, bs)

                cross_loss = (self.loss_fct(cos_sim_image2text, labels) + self.loss_fct(cos_sim_image2image, labels) + \
                              self.loss_fct(cos_sim_text2image_1, labels) + self.loss_fct(cos_sim_text2text_1, labels)) / 5


                cross_modal_loss = self.KLContrastiveSimLoss(logits_per_image, softlabel_image_sim_loss, cross_tau_loss_image,
                                                            cross_the_softlabel_tau_loss_image, use_loss=self.loss_config['cross_softlabel']['use_loss'])
                cross_modal_loss += self.KLContrastiveSimLoss(logits_per_text, softlabel_text_sim_loss, cross_tau_loss_text,
                                                            cross_the_softlabel_tau_loss_text, use_loss=self.loss_config['cross_softlabel']['use_loss'])
                cross_modal_loss /= 2.0
                # cross_loss = cross_loss + cross_modal_loss

            
            if self.args.uni_softlabel:
                if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_tau']:
                    if hasattr(self, "uni_tau"):
                        uni_tau_image_loss = self.uni_tau
                        uni_tau_text_loss = self.uni_tau
                    else:
                        uni_tau_image_loss = self.uni_tau_image
                        uni_tau_text_loss = self.uni_tau_text
                else:
                    if hasattr(self, "cross_tau"):
                        uni_tau_image_loss = self.cross_tau
                        uni_tau_text_loss = self.cross_tau
                    else:
                        uni_tau_image_loss = self.cross_tau_image
                        uni_tau_text_loss = self.cross_tau_text

                if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_softlabel_tau']:
                    if hasattr(self, "uni_the_softlabel_tau"):
                        uni_the_softlabel_tau_image_loss = self.uni_the_softlabel_tau
                        uni_the_softlabel_tau_text_loss = self.uni_the_softlabel_tau
                    else:
                        uni_the_softlabel_tau_image_loss = self.uni_the_softlabel_tau_image
                        uni_the_softlabel_tau_text_loss = self.uni_the_softlabel_tau_text
                else:
                    if hasattr(self, "cross_the_softlabel_tau"):
                        uni_the_softlabel_tau_image_loss = self.cross_the_softlabel_tau
                        uni_the_softlabel_tau_text_loss = self.cross_the_softlabel_tau
                    else:
                        uni_the_softlabel_tau_image_loss = self.cross_the_softlabel_tau_image
                        uni_the_softlabel_tau_text_loss = self.cross_the_softlabel_tau_text
                # fro uni-modal alignment (similarity)
                # image-image and image-image softlabel
                # text-text and text-text softlabel
                uni_modal_loss = self.KLContrastiveSimLoss(logits_image_image, softlabel_image_sim,uni_tau_image_loss, uni_the_softlabel_tau_image_loss,
                                                        use_loss=self.args.use_loss)
                uni_modal_loss += self.KLContrastiveSimLoss(logits_text_text, softlabel_text_sim, uni_tau_text_loss, uni_the_softlabel_tau_text_loss,
                                                            use_loss=self.args.use_loss)
                uni_modal_loss /= 2.0
                # uni_modal_loss = uni_modal_loss * 0.5
            
            if self.args.contrastive:
                # the simplest contrastive loss
                # image-text and text-image
                contrastive_loss = self.ContrastiveLoss(logits_per_image, logits_per_text)
                contrastive_loss /= 2.0
                # contrastive_loss = contrastive_loss * 1.0
            
            return contra_loss, cross_modal_loss, uni_modal_loss, contrastive_loss, cross_loss, inter_loss


        

        # return lang_pooled_output, lang_projection

    # def compute_loss(self, batch, cal_inter=False):
    #     l_pool, l_proj = self.forward(batch)

    #     # Separate representation
    #     z1, z2 = l_proj[:, 0], l_proj[:, 1]  # (bs, hidden)
    #     cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))  # (bs, bs)

    #     labels = torch.arange(cos_sim.size(0)).long().to(self.args.device)  # [0, 1, bs-1]  (bs)
    #     loss = self.loss_fct(cos_sim, labels)  # unsup: bs-1 negatives

    #     if not cal_inter:
    #         return loss

    #     else:
            # v, t, _ = self.teacher_proj(batch['img'], batch['clip_text_feat'])  # [bs, proj_dim]
            # l2v_proj = self.grounding(l_pool)  # [bs, 2, proj_dim],  output for vision grounding
            # l2v_proj = l2v_proj / l2v_proj.norm(2, dim=-1, keepdim=True)

            # p1, p2 = l2v_proj[:, 0], l2v_proj[:, 1]  # (bs, proj)
            # cos_sim_p0_v = self.sim_vl(p1.unsqueeze(1), v.unsqueeze(0), slabels=batch['vv_scores'])  # (bs, bs)
            # cos_sim_p1_v = self.sim_vl(p2.unsqueeze(1), v.unsqueeze(0), slabels=batch['vv_scores'])
            
            
            # p1, p2 = l2v_proj[:, 0], l2v_proj[:, 1]  # (bs, proj)
            # cos_sim_p0_t = self.sim_vl(p1.unsqueeze(1), t.unsqueeze(0), slabels=batch['cc_scores'])  # (bs, bs)
            # cos_sim_p1_t = self.sim_vl(p2.unsqueeze(1), t.unsqueeze(0), slabels=batch['cc_scores'])
            
            # if self.using_threshhold:
            #     cos_sim_p0_v = cos_sim_p0_v + batch['cv_slabels']
            #     cos_sim_p1_v = cos_sim_p1_v + batch['cv_slabels'] 
            #     cos_sim_p0_t = cos_sim_p0_t + batch['vc_slabels']
            #     cos_sim_p1_t = cos_sim_p1_t + batch['vc_slabels'] 
            
            # inter_loss1 = (self.loss_fct(cos_sim_p0_v, labels) + self.loss_fct(cos_sim_p1_v, labels)) / 2
            # inter_loss2 = (self.loss_fct(cos_sim_p0_t, labels) + self.loss_fct(cos_sim_p1_t, labels)) / 2

            # inter_loss = inter_loss1 + inter_loss2

            # return loss, inter_loss