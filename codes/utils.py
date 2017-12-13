import os
from PIL import Image
import numpy as np
from torchvision import transforms


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


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


def save_test_image(dir, file_name, original_image, stylize_image, output_image):
    if not os.path.exists(dir) :
        os.makedirs(dir)

    original_image = Image.fromarray(recover_image(original_image.cpu().numpy())[0])
    stylize_image = Image.fromarray(recover_image(stylize_image.cpu().numpy())[0])
    output_image = Image.fromarray(recover_image(output_image.cpu().numpy())[0])

    # ratio = 64. / max(stylize_image.size)
    # h, w = stylize_image.size
    # new_size = (ratio * h, ratio * w)
    stylize_image.thumbnail((64, 64), Image.ANTIALIAS)

    result = Image.new('RGB', (original_image.size[0] * 2 + 5, original_image.size[1]))
    result.paste(original_image, (0, 0))
    result.paste(stylize_image, (0, 0))
    result.paste(output_image, (original_image.size[0] + 5, 0))

    result.save(os.path.join(dir, file_name))
