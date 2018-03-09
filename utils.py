import os
import numpy as np
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
        mx = max(w, h)
        radio = self.output_size / mx
        new_size = w * radio, h * radio
        old_im.thumbnail(new_size, Image.ANTIALIAS)

        new_im = Image.new('RGB', (self.output_size, self.output_size))
        new_im.paste(old_im, ((self.output_size - new_size[0])/2,
                              (self.output_size - new_size[1])/2))
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


def save_test_image(dir, file_name, images):
    if not os.path.exists(dir) :
        os.makedirs(dir)

    for i, image in enumerate(images):
        images[i] = Image.fromarray(recover_image(images[i].cpu().numpy())[0])

    num = len(images)
    w, h = images[0].size[: 2]
    result = Image.new('RGB', (w * num + 5 * (num - 1), h))

    for i, image in enumerate(images):
        result.paste(image, (i * (5 + w), 0))

    result.save(os.path.join(dir, file_name))


########
# others
########
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram
