import cv2
import random
import albumentations as albu
import albumentations.pytorch as albu_pt

# Default ImageNet mean and std
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Normalize into [-1, 1] range
# MEAN = (0.5, 0.5, 0.5)
# STD = (0.5, 0.5, 0.5)


def get_aug(aug_type="val", size=512):
    """Return augmentations by type
    Args:
        aug_type (str): one of `val`, `test`, `light`, `medium`
        size (int): final size of the crop
    """

    NORM_TO_TENSOR = albu.Compose([albu.Normalize(mean=MEAN, std=STD), albu_pt.ToTensorV2()])

    # CROP_AUG = albu.Compose([albu.RandomCrop(size, size)])

    VAL_AUG = albu.Compose([
        # albu.CenterCrop(size, size),
        NORM_TO_TENSOR
    ])

    LIGHT_AUG = albu.Compose(
        [
            albu.Flip(),
            albu.Cutout(num_holes=12, max_h_size=size // 16, max_w_size=size // 16, fill_value=0, p=0.5),
            NORM_TO_TENSOR
        ],
        p=1.0,
    )

    # aug from github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/
    HARD_AUG = albu.Compose(
        [
            albu.Flip(p=0.5),
            albu.IAAAffine(rotate=0.2, shear=0.2, mode='constant', p=0.5),
            albu.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            albu.MotionBlur(p=0.5),
            albu.CLAHE(p=0.5),
            albu.Cutout(num_holes=16, max_h_size=size // 16, max_w_size=size // 16, fill_value=0, p=0.5),
            RandomCropThenScaleToOriginalSize(limit=0.2, p=1.0),
            NORM_TO_TENSOR,
        ],
        p=1.0,
    )

    types = {
        "val": VAL_AUG,
        "test": VAL_AUG,
        "light": LIGHT_AUG,
        "hard": HARD_AUG,
    }

    return types[aug_type]


class RandomCropThenScaleToOriginalSize(albu.ImageOnlyTransform):
    """Crop a random part of the input and rescale it to some size.
    Args:
        limit (float): maximum factor range for cropping region size.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        pad_value (int): pixel value for padding.
        p (float): probability of applying the transform. Default: 1.
    """

    def __init__(self, limit=0.1, interpolation=cv2.INTER_LINEAR, pad_value=0, p=1.0):
        super(RandomCropThenScaleToOriginalSize, self).__init__(p)
        self.limit = limit
        self.interpolation = interpolation
        self.pad_value = pad_value

    def apply(self, img, height_scale=1.0, width_scale=1.0, h_start=0, w_start=0, interpolation=cv2.INTER_LINEAR,
              pad_value=0, pad_loc_seed=None, **params):
        img_height, img_width = img.shape[:2]
        crop_height, crop_width = int(img_height * height_scale), int(img_width * width_scale)
        crop = self.random_crop(img, crop_height, crop_width, h_start, w_start, pad_value, pad_loc_seed)
        return albu.augmentations.functional.resize(crop, img_height, img_width, interpolation)

    def get_params(self):
        height_scale = 1.0 + random.uniform(-self.limit, self.limit)
        width_scale = 1.0 + random.uniform(-self.limit, self.limit)
        return {'h_start': random.random(),
                'w_start': random.random(),
                'height_scale': height_scale,
                'width_scale': width_scale,
                'pad_loc_seed': random.random()}

    def update_params(self, params, **kwargs):
        if hasattr(self, 'interpolation'):
            params['interpolation'] = self.interpolation
        if hasattr(self, 'pad_value'):
            params['pad_value'] = self.pad_value
        params.update({'cols': kwargs['image'].shape[1], 'rows': kwargs['image'].shape[0]})
        return params

    @staticmethod
    def random_crop(img, crop_height, crop_width, h_start, w_start, pad_value=0, pad_loc_seed=None):
        height, width = img.shape[:2]

        if height < crop_height or width < crop_width:
            img = _pad_const(img, crop_height, crop_width, value=pad_value, center=False, pad_loc_seed=pad_loc_seed)

        y1 = max(int((height - crop_height) * h_start), 0)
        y2 = y1 + crop_height
        x1 = max(int((width - crop_width) * w_start), 0)
        x2 = x1 + crop_width
        img = img[y1:y2, x1:x2]
        return img


def _pad_const(x, target_height, target_width, value=255, center=True, pad_loc_seed=None):
    random.seed(pad_loc_seed)
    height, width = x.shape[:2]

    if height < target_height:
        if center:
            h_pad_top = int((target_height - height) / 2.0)
        else:
            h_pad_top = random.randint(a=0, b=target_height - height)
        h_pad_bottom = target_height - height - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if width < target_width:
        if center:
            w_pad_left = int((target_width - width) / 2.0)
        else:
            w_pad_left = random.randint(a=0, b=target_width - width)
        w_pad_right = target_width - width - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    x = cv2.copyMakeBorder(x, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right,
                           cv2.BORDER_CONSTANT, value=value)
    return x
