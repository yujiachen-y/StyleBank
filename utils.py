from __future__ import division

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


#######################
# image classes and functions
#######################
class ZeroPadding(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, old_im):
        w, h = old_im.size
        mx = max(w, h) * 1.
        radio = self.output_size / mx
        new_size = int(w * radio), int(h * radio)
        old_im.thumbnail(new_size, Image.ANTIALIAS)

        new_im = Image.new('RGB', (self.output_size, self.output_size))
        new_im.paste(old_im, (int((self.output_size - new_size[0])/2),
                              int((self.output_size - new_size[1])/2)))
        return new_im

def tensor_normalizer():
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )


def recover_image(img):
    return (
        (
            img *
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)) +
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        ).transpose(0, 2, 3, 1) *
        255.
    ).clip(0, 255).astype(np.uint8)


def save_test_image(dir, file_name, images, titles):
    if not os.path.exists(dir) :
        os.makedirs(dir)

    for i, image in enumerate(images):
        images[i] = Image.fromarray(recover_image(images[i].cpu().numpy())[0])

    num = len(images)
    row = 2
    col = (num + 1) // row

    plt.figure()
    for i, image, title in enumerate(zip(images, titles)):
        plt.subplot(row, col, i)
        plt.axis('off')
        plt.title(title)
        plt.imshow(image)
    plt.savefig(os.path.join(dir, file_name))


########
# others
########
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram
