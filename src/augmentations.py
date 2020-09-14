import albumentations as albu
import albumentations.pytorch as albu_pt

# Default ImageNet mean and std
# MEAN = (0.485, 0.456, 0.406)
# STD = (0.229, 0.224, 0.225)

# Normalize into [-1, 1] range
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def get_aug(aug_type="val", size=512):
    """Return augmentations by type
    Args:
        aug_type (str): one of `val`, `test`, `light`, `medium`
        size (int): final size of the crop
    """

    NORM_TO_TENSOR = albu.Compose([albu.Normalize(mean=MEAN, std=STD), albu_pt.ToTensorV2()])

    CROP_AUG = albu.Compose([albu.RandomCrop(size, size)])

    VAL_AUG = albu.Compose([albu.CenterCrop(size, size), NORM_TO_TENSOR])
    # VAL_AUG = albu.Compose([albu.Resize(size, size), NORM_TO_TENSOR])

    LIGHT_AUG = albu.Compose([CROP_AUG, albu.Flip(), NORM_TO_TENSOR])

    # aug from good performing public kernel
    HARD_AUG = albu.Compose(
        [
            albu.RandomResizedCrop(size, size, scale=(0.6, 1.0), ratio=(0.8, 1.2), p=1.0),
            albu.RandomRotate90(p=0.5),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.Cutout(num_holes=8, max_h_size=16, max_w_size=16, fill_value=0, p=0.5),
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
