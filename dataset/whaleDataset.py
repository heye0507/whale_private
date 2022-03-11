import cv2
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
import random
import copy
import numpy as np


class WhaleDataset(Dataset):
    def __init__(self,path,df,transforms=None, test=False, **kwargs):
        super().__init__()
        self.path = path
        self.df = df
        self.transforms = transforms
        self.test = test
        if 'crop_method' in kwargs.keys():
            self.crop_method = kwargs['crop_method']
        else:
            self.crop_method = None
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        fname = self.df.iloc[idx]['image']
        if not self.test:
            label = self.df.iloc[idx]['final_category']
            Id = self.df.iloc[idx]['individual_id']
        image = cv2.imread(f'{self.path}/{fname}', cv2.IMREAD_COLOR)
        if self.crop_method == 'body':
            left, top, right, bottom = map(int, self.df.iloc[idx]['body_bbox'].split())
        elif self.crop_method == 'fin':
            left, top, right, bottom = map(int, self.df.iloc[idx]['bbox'].split())
        if self.crop_method != None and left != -1:
            image = image[top:bottom, left:right]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            aug = self.transforms(
                image = image,
            )
            image = aug['image']
        if self.test:
            return {'image': image,
                   }, fname
        else:
            return {'image': image,'id': Id}, label
        
class WhaleTripletSampler(Sampler):
    '''
        Randomly sample N identies, for each identity,
        Randomly sample K images, the batch size should be N*K
        
        Args:
        - source: df_train
        - batch_size: total batch size 
        - num_instance: number of images per indentity in a batch (K)
        
        Note:
        - Therefore the number of identity in a batch is equal to batch_size // K
    '''
    
    def __init__(self, source, batch_size, num_instances):
        assert (batch_size % num_instances) == 0, "identity cannot be divided"
        self.bs = batch_size
        self.num_instances = num_instances
        self.id_per_batch = int(batch_size // num_instances)
        self.index_dict, self.ids = self._process_df(source)
        
        # estimate length of examples in one epoch
        self.length = 0
        for Id in self.ids:
            idxs = self.index_dict[Id]
            num = len(idxs)
            if num < self.num_instances: # for indenties have less then K images
                num = self.num_instances
            self.length += num - num % self.num_instances
            
        print(self.length)
    
    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        
        for Id in self.ids:
            idxs = copy.deepcopy(self.index_dict[Id])
            
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            
            random.shuffle(idxs)
            batch_idxs = []
            
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[Id].append(batch_idxs)
                    batch_idxs = []
                    
        avai_ids = copy.deepcopy(self.ids)
        final_idxs = []
        
        while len(avai_ids) >= self.id_per_batch:
            select_ids = random.sample(avai_ids, self.id_per_batch)
            for Id in select_ids:
                batch_idxs = batch_idxs_dict[Id].pop(0)
                final_idxs.extend(batch_idxs)
                
                if len(batch_idxs_dict[Id]) == 0:
                    avai_ids.remove(Id)
                    
        return iter(final_idxs)
    
    def __len__(self):
        return self.length
    
    def _process_df(self,df):
        '''
            Get Id to index mapping and number of unique ids
        '''
        index_dict = defaultdict(list)
        
#         image_fns = df['image'].tolist()
        ids = df['final_category'].tolist()
        for idx, Id in enumerate(ids):
            index_dict[Id].append(idx)
        ids = df['final_category'].unique().tolist()
        assert len(ids) == len(index_dict.keys()) #sanity check
        return index_dict, ids
        
