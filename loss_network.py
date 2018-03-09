import torch
import torchvision
from collections import namedtuple
from torch.autograd import Variable

mse_loss = torch.nn.MSELoss()

LossOutput = namedtuple('LossOutput',
                        ['relu1_2', 'relu2_2', 'relu3_2','relu4_2'])
netdir = '/home/jiachen/.torch/models/vgg16.pth'


# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
class LossNetwork(torch.nn.Module):
    def __init__(self):
        super(LossNetwork, self).__init__()

        vgg_model = torchvision.models.vgg16(pretrained=True)
        # pram = torch.load(netdir)
        # vgg_model.load_state_dict(pram)

        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '13': 'relu3_2',
            '20': 'relu4_2'
        }

    def forward(self, x):
        output = {}
        count = 0
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                # print name
                output[self.layer_name_mapping[name]] = x
                count = count + 1
            if count == len(self.layer_name_mapping):
                break
        return LossOutput(**output)


def get_content_loss(loss_network, content_image, output_features):
    content_image = Variable(content_image.data, volatile=True)
    content_features = loss_network(content_image)
    content_features = Variable(content_features[3].data, requires_grad=False)
    return mse_loss(output_features[3], content_features)


def get_style_loss(loss_network, style_image, output_features):
    style_image = Variable(style_image.data, volatile=True)
    style_features = loss_network(style_image)
    style_loss = 0
    for i in range(4):
        temp_features = Variable(style_features[i].data, requires_grad=False)
        style_loss += mse_loss(output_features[i], temp_features)
    return style_loss


def get_regularization_loss(output_image):
    return torch.sum(torch.abs(output_image[:, :, :, :-1] - output_image[:, :, :, 1:])) + \
      torch.sum(torch.abs(output_image[:, :, :-1, :] - output_image[:, :, 1:, :]))
