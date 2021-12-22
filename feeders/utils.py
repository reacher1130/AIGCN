from .tools import *
from torchvision import transforms, datasets


def aug_transfrom(aug_name, args_list, norm, norm_aug, selected_frames):
    aug_name_list = aug_name.split("_")
    transform_aug = [aug_look('selectFrames', selected_frames)]

    if aug_name_list[0] != 'None':
        for i, aug in enumerate(aug_name_list):
            transform_aug.append(aug_look(aug, args_list[i * 2], args_list[i * 2 + 1]))

    if norm == 'normalizeC':
        transform_aug.extend([Skeleton2Image(), ToTensor(), norm_aug, Image2skeleton()])
    elif norm == 'normalizeCV':
        transform_aug.extend([ToTensor(), norm_aug])
    else:
        transform_aug.extend([ToTensor(), ])
    transform_aug = transforms.Compose(transform_aug)
    return transform_aug


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


