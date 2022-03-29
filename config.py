CFG = {
    # File locations
    "name": "test_trip",
    "PATH": '/mnt/home/hheat/USERDIR/kaggle/kaggle_whale',
    "train_data": f'/mnt/home/hheat/USERDIR/kaggle/kaggle_whale/data/train_images',
    "save_root": f'/mnt/home/hheat/USERDIR/kaggle/kaggle_whale/model/test_trip',
    "folder": 'best',
    "log_path": 'log.csv',
    "crop_method": 'fin', #fin, body
    "valid_set": "spec_fold",
    # Train Hyperparams
    "channel_size": 2048,
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "device_id": [0,1,2,3],
    "seed": 42,
    "epochs_p1": 1,
    "epochs_p2": 21,
    "img_size": 784,
    "img_height": 392,
    "model_name": "tf_efficientnet_b7_ns",
    "num_classes": 15587,
    "pretrained": True,
    "drop_rate": 0.2,
    "train_batch_size": 12,
    "valid_batch_size": 12,
    "num_workers": 12,
    "lr": 1e-3,
    "inp_channel": 3,
    "cut": None,
    "n_fold": 5,
    "fold_number": 0,
    # ArcFace Hyperparameters
    "s": 30.0, 
    "m": 0.3,
    "ls_eps": 0.0,
    "easy_margin": False
    
    
}