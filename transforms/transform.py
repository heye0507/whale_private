import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_train_transform(config):
    return A.Compose([
        A.Resize(config['img_size'], config['img_size']),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
        A.Rotate(limit=30, p=0.5),
        A.Blur(blur_limit=3,p=0.2),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)

# def get_train_transform(config):
#     return A.Compose([
#         A.Resize(config['img_size'], config['img_size']),
# #         A.HorizontalFlip(p=0.5),
# #         A.VerticalFlip(p=0.5),
#         A.Rotate(limit=30, p=0.5),
#         A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
#         A.Blur(blur_limit=3,p=0.2),
#         A.Normalize(
#                 mean=[0.485, 0.456, 0.406], 
#                 std=[0.229, 0.224, 0.225], 
#                 max_pixel_value=255.0, 
#                 p=1.0
#             ),
#         ToTensorV2()], p=1.)

def get_valid_transform(config):
    return A.Compose([
        A.Resize(config['img_size'], config['img_size']),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)