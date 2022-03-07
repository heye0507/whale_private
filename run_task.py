import sys
sys.path.insert(1, '/mnt/home/hheat/USERDIR/kaggle/kaggle_whale')

import os
from src.config import CFG
os.environ["CUDA_VISIBLE_DEVICES"] = CFG["CUDA_VISIBLE_DEVICES"]

from src.dataset.whaleDataset import WhaleDataset
from src.models.arcNet_aux import Eff_Arc_Aux_Net
from src.utils import map5, _splitter, _check_freeze, ArcfaceCallback
from src.transforms.transform import get_train_transform, get_valid_transform

from fastai.vision.all import *
from fastai.distributed import *

PATH = CFG['PATH']
train_data = CFG['train_data']
df_train = pd.read_csv(f'{CFG["PATH"]}/data/train_v3.csv')
CFG['train_batch_size'] *= torch.cuda.device_count()
CFG['valid_batch_size'] *= 2*torch.cuda.device_count()

print(torch.cuda.device_count())

if not os.path.exists(CFG['save_root']):
    os.makedirs(CFG['save_root'])

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


##### Setup Data pipeline #####

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
        sampler=RandomSampler(train_dataset),
        batch_size=CFG['train_batch_size'],
        pin_memory=False,
        drop_last=True,
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

##### Train Loop #####

learn = Learner(dls,Eff_Arc_Aux_Net(CFG),loss_func=CrossEntropyLossFlat(),
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

torch.save(learn.model.state_dict(), f'CFG["save_root"] + "/" + final.bin')


