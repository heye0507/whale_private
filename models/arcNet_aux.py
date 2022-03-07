import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from fastai.vision.learner import _update_first_layer, num_features_model, has_pool_type
from fastai.layers import AdaptiveAvgPool, Flatten
from fastai.torch_core import apply_init
from fastai.callback.hook import hook_outputs

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




class Eff_Arc_Aux_Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_outputs = config['num_classes']
        if 'channel_size' in config:
            channel_size = config['channel_size']
        else:
            channel_size = 2048

        backbone = self.create_timm_backbone(config)
        self.backbone_1 = backbone[0:3]
        self.backbone_2 = backbone[3][:5]
        self.backbone_3 = backbone[3][5:6]
        self.backbone_4 = backbone[3][6:]
        self.backbone_5 = backbone[4:]
        
#         self.sfs = hook_outputs([self.backbone[3][idx][-1].bn3 for idx in [4,5,6]]) #b7 last 3 layers in conv part
        in_features = 3808 #num_features_model(nn.Sequential(*self.backbone.children()))
        self.avg_pool = AdaptiveAvgPool()
        self.head = nn.Sequential(Flatten(),
                                 nn.Dropout2d(config['drop_rate'], inplace=True),
                                 nn.Linear(in_features, channel_size),
                                 nn.BatchNorm1d(channel_size))
        apply_init(self.head, nn.init.kaiming_normal_)
        self.head_arc = ArcMarginProduct(channel_size, num_outputs, 
                                         s=config['s'], m=config['m'],
                                         easy_margin = config['easy_margin'],
                                         ls_eps = config['ls_eps']
                                        )

    def forward(self, images, labels=None):
        features = self.backbone_1(images)
        features = self.backbone_2(features)
        f1 = features.clone().detach()
        f1 = self.avg_pool(f1).to(images.device)
        features = self.backbone_3(features)
        f2 = features.clone().detach()
        f2 = self.avg_pool(f2).to(images.device)
        features = self.backbone_4(features)
        f3 = features.clone().detach()
        f3 = self.avg_pool(f3).to(images.device)
        features = self.backbone_5(features)
        features = self.avg_pool(features).to(images.device)
#         print(f1.shape,f2.shape,f3.shape, features.shape)
        assert f1.shape[0] == f2.shape[0] == f3.shape[0] == features.shape[0], f'{f1.shape[0]}, {f2.shape[0]}, {f3.shape[0]}, {features.shape[0]}'
        features = torch.cat([f1, f2, f3, features], dim=1)
#         print(features.shape)
        features = self.head(features)
        if labels is not None:
            return self.head_arc(features, labels), features
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