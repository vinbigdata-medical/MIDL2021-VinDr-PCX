import numpy as np
import cv2
from albumentations import HorizontalFlip


def border_pad(image, cfg):
    h, w, c = image.shape
    if cfg.border_pad == 'zero':
        image = np.pad(image, ((0, cfg.image_size - h),
                               (0, cfg.image_size - w), (0, 0)),
                       mode='constant',
                       constant_values=0.0)
    elif cfg.border_pad == 'pixel_mean':
        image = np.pad(image, ((0, cfg.image_size - h),
                               (0, cfg.image_size - w), (0, 0)),
                       mode='constant',
                       constant_values=cfg.pixel_mean)
    else:
        image = np.pad(image, ((0, cfg.image_size - h),
                               (0, cfg.image_size - w), (0, 0)),
                       mode=cfg.border_pad)

    return image

def fix_ratio(image, cfg):
    h, w, c = image.shape

    if h >= w:
        ratio = h * 1.0 / w
        h_ = cfg.image_size
        w_ = round(h_ / ratio)
    else:
        ratio = w * 1.0 / h
        w_ = cfg.image_size
        h_ = round(w_ / ratio)

    image = cv2.resize(image, dsize=(w_, h_), interpolation=cv2.INTER_LINEAR)
    image = border_pad(image, cfg)

    return image

def transform(image, cfg):
    assert image.ndim == 2, "image must be gray image"
    if cfg.use_equalizeHist:
        image = cv2.equalizeHist(image)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = fix_ratio(image, cfg)

    # normalization
    # image = image.astype(np.float32) - cfg.pixel_mean
    # # vgg and resnet do not use pixel_std, densenet and inception use.
    # if cfg.pixel_std:
    #     image /= cfg.pixel_std
    # # normal image tensor :  H x W x C
    # # torch image tensor :   C X H X W
    image = image.transpose((2, 0, 1))

    return image

def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def test_time_aug(image):
    img1 = image.copy()
    img2 = HorizontalFlip(p=1)(image=image)['image']
    img3 = rotate_image(img1, 5)
    img4 = rotate_image(img1, -5)
    img5 = rotate_image(img2, 5)
    img6 = rotate_image(img2, -5)
    return [img1, img2, img3, img4, img5, img6]