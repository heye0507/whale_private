import torch
from fastai.callback.core import Callback
from fastai.torch_core import params
from tqdm import tqdm
import numpy as np
import pandas as pd



class ArcfaceCallback(Callback):
    def before_batch(self):
        self.learn.xb = (self.x['image'], self.y)
        
    def after_pred(self):
        self.learn.pred = self.pred[0]

class ArcfaceFocalOHEM(Callback):
    def before_batch(self):
        self.learn.xb = (self.x['image'], self.y)
        
    def after_pred(self):
        if not self.training:
            self.learn.pred = self.pred[0]
        else:
            self.learn.pred = (self.pred, self.epoch)

# Metric
def map5kfast(preds, targs, k=10):
    predicted_idxs = preds.sort(descending=True)[1]
    top_5 = predicted_idxs[:, :5]
    scores = torch.zeros(len(preds), k).float()
    for kk in range(k):
        scores[:,kk] = (top_5[:,kk] == targs).float() / float((kk+1))
    return scores.max(dim=1)[0].mean()

def map5(preds,targs):
    if type(preds) is list:
        raise Exception('Not Implemented... ')
    return map5kfast(preds,targs, 5)

def _apk(actual, predicted, k=5):
        """
        Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
        """
        if len(predicted) > k:
            predicted = predicted[:k]
        actual = [actual]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not actual:
            return 0.0

        ret = score / min(len(actual), k)
        return ret


def _mapk(actual, predicted, k=5):
        """
        Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
        """
        return np.mean([_apk(a, p, k) for a, p in zip(actual, predicted)])

    

def _splitter(model):
    return [params(model.backbone), params(torch.nn.Sequential(model.head, model.head_arc))]

def _check_freeze(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        
def inference(m, dataloader, test=False, **kwargs):
    '''
        get feat embedding of all images, 
        
        return dict, key is identies, value is list of its features
    '''
    m.eval()
#     ret_dict = defaultdict(list)
    
    if 'device' in kwargs.keys():
        device = kwargs['device']
        
    m.to(device)
    
    ret_features, ret_labels = [], []
    filename = []
    
    with torch.no_grad():
        for xb, yb in tqdm(dataloader):
            images = xb['image'].to(device)
            if not test:
                fnames = xb['fname']
            else: fnames = '0'
            id_list = yb
#             if not test:
#                 id_list = xb['fn']
#             else: id_list = yb
            feats, _ = m(images)
            feats = torch.nn.functional.normalize(feats)
            ret_features.extend(feats.cpu().numpy())
            ret_labels.extend(id_list)
            filename.extend(fnames)
#             for key, feat in zip(id_list, feats.cpu().numpy()):
#                 ret_dict[key].append(feat)
    return ret_features, ret_labels, filename

import nmslib

def create_index(a):
    index = nmslib.init(space='cosinesimil')
    index.addDataPointBatch(a)
    index.createIndex()
    return index

def get_knns(index, vecs):
     return zip(*index.knnQueryBatch(vecs, k=200, num_threads=4))

def get_knn(index, vec): return index.knnQuery(vec, k=200)


def get_top5_nms(scores, indices, labels, threshold, k=5):
            used = set()
            ret_labels = []
            ret_scores = []
            ret_ids = []

            for i,s in zip(indices, scores):
                l = labels[i].item()
                s = 1 - s
                if l in used:
                    continue

                if -1 not in used and s < threshold:
                    used.add(-1)
                    ret_labels.append(-1)
                    ret_scores.append(threshold)
                    
                
                used.add(l)
                ret_labels.append(l)
                ret_scores.append(s)
                if len(ret_labels) >= k:
                    break
            return ret_labels[:5], ret_scores[:5]
        
def process_valid_with_new(lbls, df):
    ret_lbls = []
    gt_lbls = df['final_category'].tolist()
    
    for lbl, gt_lbl in tqdm(zip(lbls, gt_lbls), total=len(gt_lbls)):
        lbl = lbl.item()
        if gt_lbl != -1: assert lbl == gt_lbl
        ret_lbls.append(gt_lbl)
            
    return ret_lbls

def create_submission(pred_labels, df_train, df_test):
    fns = df_test['image'].tolist()
    result = {'image': [], 'predictions' : []}
    
    for f, lbls in tqdm(zip(fns, pred_labels),total=len(fns)):
        row_lbls = []
        for l in lbls:
            if l == -1: 
                row_lbls.append('new_individual')
            else:
                row_lbls.append(df_train.loc[df_train['final_category'] == l, 'individual_id'].iloc[0])
        result['image'].extend([f])
        result['predictions'].extend([' '.join(i for i in row_lbls)])
    return pd.DataFrame(result)
     