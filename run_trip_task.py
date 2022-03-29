import sys
sys.path.insert(1, '/mnt/home/hheat/USERDIR/kaggle/kaggle_whale')

import os
from src.config import CFG
os.environ["CUDA_VISIBLE_DEVICES"] = CFG["CUDA_VISIBLE_DEVICES"]

from src.dataset.whaleDataset import WhaleDataset, WhaleTripletSampler
from src.loss.triplet_loss import CombLoss
from src.models.arcNet import Eff_Arc_Net
from src.utils import map5, _splitter, _check_freeze#, ArcfaceCallback
# from src.transforms.transform import get_train_transform, get_valid_transform

from fastai.vision.all import *
from fastai.distributed import *
from glob import glob

SEED = CFG['seed']

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

PATH = '/mnt/home/hheat/USERDIR/kaggle/kaggle_whale'
train_data = f'{PATH}/data/train_images'
df_all = pd.read_csv(f'{PATH}/data/train_v8.csv')
df_train = df_all[(df_all['species'] != 'beluga') & 
                  (df_all['species'] != 'gray_whale') &
                  (df_all['species'] != 'southern_right_whale')
                 ].copy()

df_train = df_train[df_train['count'] > 1]

CFG['train_batch_size'] *= torch.cuda.device_count()
CFG['valid_batch_size'] *= 2*torch.cuda.device_count()

print(torch.cuda.device_count())

if not os.path.exists(CFG['save_root']):
    os.makedirs(CFG['save_root'])


##### Setup Data pipeline #####

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_train_transform(config):
    return A.Compose([
        A.Resize(config['img_height'], config['img_size']),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
        A.Rotate(limit=30, p=0.5),
        A.Blur(blur_limit=3,p=0.2),
        A.ToGray(p=0.5),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)

def get_valid_transform(config):
    return A.Compose([
        A.Resize(config['img_height'], config['img_size']),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)

train_dataset = WhaleDataset(
    path = train_data,
    df = df_train[df_train[CFG['valid_set']] != CFG['fold_number']],
    transforms=get_train_transform(CFG),
    crop_method = CFG['crop_method'],
)

validation_dataset = WhaleDataset(
    path = train_data,
    df = df_train[df_train[CFG['valid_set']] == CFG['fold_number']],
    transforms=get_valid_transform(CFG),
    crop_method = CFG['crop_method'],
)

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler = WhaleTripletSampler(df_train[df_train['spec_fold'] != CFG['fold_number']], CFG['train_batch_size'], 4),
        batch_size=CFG['train_batch_size'],
        pin_memory=False,
        drop_last=False,
        num_workers=CFG['num_workers'],
)

val_loader = torch.utils.data.DataLoader(
    validation_dataset, 
    batch_size=CFG['valid_batch_size'],
    num_workers=CFG['num_workers'],
    shuffle=False,
    sampler=SequentialSampler(validation_dataset),
    pin_memory=False,
)

dls = DataLoaders(train_loader, val_loader)


##### callbacks ######
class ArcfaceCallback(Callback):
    def before_batch(self):
        self.learn.xb = (self.x['image'], self.y)
        
    def after_pred(self):
        if not self.training:
            self.learn.pred = self.pred[0]
            
class SaveModelCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model's best during training and loads it at the end."
    order = 99
    def __init__(self, monitor='valid_loss', comp=None, min_delta=0., fname='model', every_epoch=False, at_end=False,
                 with_opt=False, reset_on_fit=True):
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit)
        assert not (every_epoch and at_end), "every_epoch and at_end cannot both be set to True"
        # keep track of file path for loggers
        self.last_saved_path = None
        store_attr('fname,every_epoch,at_end,with_opt')

    def _save(self, name): self.last_saved_path = self.learn.save(name, with_opt=self.with_opt)

    def after_epoch(self):
        "Compare the value monitored to its best score and save if best."
        if self.every_epoch:
            if (self.epoch%self.every_epoch) == 0: self._save(f'{self.fname}_{self.epoch}')
        else: #every improvement
            super().after_epoch()
            if self.new_best:  
                print(f'Better model found at epoch {self.epoch} with {self.monitor} value: {self.best}.')
                torch.save(self.learn.model.state_dict(), 
                           f'{CFG["save_root"]}/{CFG["folder"]}-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{CFG["save_root"]}/{CFG["folder"]}*epoch.bin'))[:-3]:
                    os.remove(path)
                
    def after_fit(self, **kwargs):
        "Load the best model."
        self._save(f'{self.fname}')

##### Train Loop #####

learn = Learner(dls,Eff_Arc_Net(CFG),loss_func=BaseLoss(CombLoss, flatten=False),
                cbs=[ArcfaceCallback, 
                     SaveModelCallback(min_delta=0.001, fname=CFG['save_root'] + '/' + CFG['folder']), 
                     CSVLogger(fname=CFG['save_root'] + '/' + CFG['log_path']),
                     ParallelTrainer(device_ids=CFG['device_id'])],
                metrics = [map5,top_k_accuracy],
                splitter = _splitter,
               ).to_fp16()

##### Phase 1 warmup head #####
learn.freeze()
print(f"Params trainable: {_check_freeze(learn.model)}")

learn.fit_one_cycle(CFG['epochs_p1'],CFG['lr'])

##### Phase 2 training #####

learn.unfreeze()
print(f"Params trainable: {_check_freeze(learn.model)}")
learn.fit_one_cycle(CFG['epochs_p2'],1e-4)

# torch.save(learn.model.state_dict(), f'CFG["save_root"] + "/" + final.bin')


