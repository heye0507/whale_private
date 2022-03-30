import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from fastai.vision.learner import _update_first_layer, num_features_model, has_pool_type
from fastai.layers import AdaptiveConcatPool2d, Flatten
from fastai.torch_core import apply_init

class ArcMarginProduct(nn.Module):
    r"""
        Source: https://www.kaggle.com/alibaba19/fastai-arcface-pipeline-training
        Modified: HH
        Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, inp, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(inp), F.normalize(self.weight.to(inp.device)))
        sine = torch.sqrt(1.0 - torch.pow(cosine,2)).to(cosine.dtype) #needed for to_fp16()
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size()).to(cosine.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class BinaryHead(nn.Module):
    def __init__(self, num_class, emb_size, s = 8.0):
        super(BinaryHead,self).__init__()
        self.s = s
        self.fc = nn.Sequential(nn.Linear(emb_size, num_class))

    def forward(self, fea):
        fea = F.normalize(fea)
        logit = self.fc(fea)*self.s
        return logit

class Eff_Arc_Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_outputs = config['num_classes']
        if 'channel_size' in config:
            channel_size = config['channel_size']
        else:
            channel_size = 2048

        self.backbone = self.create_timm_backbone(config)
        in_features = num_features_model(nn.Sequential(*self.backbone.children())) * 2
        self.head = nn.Sequential(AdaptiveConcatPool2d(1), Flatten(), nn.BatchNorm1d(in_features),
                                 nn.Dropout2d(config['drop_rate'], inplace=True),
                                 nn.Linear(in_features, channel_size),
                                 nn.BatchNorm1d(channel_size))
        apply_init(self.head, nn.init.kaiming_normal_)
        self.head_arc = ArcMarginProduct(channel_size, num_outputs, 
                                         s=config['s'], m=config['m'],
                                         easy_margin = config['easy_margin'],
                                         ls_eps = config['ls_eps']
                                        )
        if config['binary_head']:
            self.binary_head = BinaryHead(num_outputs, channel_size)

    def forward(self, images, labels=None):
        features = self.backbone(images)
        features = self.head(features)
        if labels is not None:
            return self.head_arc(features, labels), features, self.binary_head(features)
        return features
    
    def create_timm_backbone(self,config):
        model = timm.create_model(config['model_name'],pretrained=config['pretrained'],
                                  num_classes=0)
        _update_first_layer(model, config['inp_channel'], config['pretrained']) # update first layer inp channel
        if config['cut'] is None:
            ll = list(enumerate(model.children()))
            cut = next(i for i,o in reversed(ll) if has_pool_type(o))
        if isinstance(cut, int): 
            return nn.Sequential(*list(model.children())[:cut])