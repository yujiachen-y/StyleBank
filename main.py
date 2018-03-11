import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
import time
import numpy as np

from dataloader import ImageLoader, ContentStyleDataset
from utils import tensor_normalizer, ZeroPadding, save_test_image
from loss_network import LossNetwork, get_content_loss, get_regularization_loss, get_style_loss, mse_loss
from training_network import StyleBankNet, Discriminator

######################################################################
# set a SEED for random first so that we can get a reproducible result
######################################################################
SEED = 1080
np.random.seed(SEED)
torch.manual_seed(SEED)

##################################
# build transforms for ImageLoader
##################################
IMAGE_SIZE = 256
transform_list = []
# (1) cut out the extra parts of pictures
transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                transforms.CenterCrop(IMAGE_SIZE),
                                transforms.ToTensor(),
                                tensor_normalizer()])
transform_list.append(transform)
# (2) padding zeros in the edges of picture
transform = transforms.Compose([ZeroPadding(IMAGE_SIZE),
                                transforms.ToTensor(),
                                tensor_normalizer()])
transform_list.append(transform)
print('transforms done')
####################################
# construct ImageLoaders and dataloader
####################################
DATADIR = os.path.join('..', 'Datasets')
BATCHSIZE = 4

dataset_list = ['AR', 'CUHK', 'CUHK_FERET', 'XM2VTS',
                'VIPSL0', 'VIPSL1', 'VIPSL2', 'VIPSL3', 'VIPSL4']
imageloaders = [ImageLoader(os.path.join(DATADIR, dname, 'photos'),
                            os.path.join(DATADIR, dname, 'sketches'))
                for dname in dataset_list]

cs_dataset = ContentStyleDataset(imageloaders, transform_list)
dataloader = DataLoader(cs_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=4)
print('dataloader done')
############################################################
# instantiate neural networks, loss functions and optimizers
############################################################
netG = StyleBankNet(len(dataset_list))
netD = Discriminator()

loss_network = LossNetwork()
bce_loss = nn.BCELoss()

use_gpu = torch.cuda.is_available()
if use_gpu:
    netG = netG.cuda()
    netD = netD.cuda()
    loss_network = loss_network.cuda()

LR = 1e-2
optimizerG = Adam(netG.parameters(), LR)
optimizerD = Adam(netD.parameters(), LR)
scheduler = StepLR(optimizerG, step_size=30, gamma=0.2)

netG.train()
netD.train()
print('networks done')
######################
# training the network
######################
CONTENT_WEIGHT = 100
STYLE_WEIGHT = 10000
REG_WEIGHT = 1e-5
LOG_INTERVAL = 200
EPOCH = 300
T = 2
LAMBDA = 1

count, t = 0, 0

print('\ntraining start at', time.strftime('%Y_%m_%d_%H_%M', time.localtime()))
for epoch in range(EPOCH):
    Loss_D, Loss_G, D_x, D_G_z1, D_G_z2 = 0, 0, 0, 0, 0
    Loss_I, Loss_S, Loss_C, Loss_R = 0, 0, 0, 0
    epoch_count = 0

    scheduler.step()

    for batch_id, (style_id, content_image, stylized_image) in enumerate(dataloader):
        optimizerG.zero_grad()
        optimizerD.zero_grad()
        
        batch_size = len(style_id)
        count += batch_size
        epoch_count += batch_size

        label = torch.FloatTensor(batch_size).fill_(1)
        
        content_image = Variable(content_image)
        stylized_image = Variable(stylized_image)
        if use_gpu:
            content_image = content_image.cuda()
            stylized_image = stylized_image.cuda()
            label = label.cuda()
        labelv = Variable(label)

        output_image = netG(content_image, style_id)
        output_features = loss_network(output_image)

        #############################################################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # the Discriminator training part is referenced from
        # https://github.com/pytorch/examples/blob/master/dcgan/main.py
        # at 2018/3/6
        #############################################################

        # train with real
        p_real = netD(stylized_image)
        errD_real = bce_loss(p_real, labelv)
        errD_real.backward()
        D_x = D_x + p_real.data.mean()

        # train with fake
        labelv = Variable(label.fill_(0))
        p_fake = netD(output_image.detach())
        errD_fake = bce_loss(p_fake, labelv)
        errD_fake.backward()
        D_G_z1 = D_G_z1 + p_fake.data.mean()
        Loss_D = Loss_D + errD_fake.data.mean() + errD_real.data.mean()
        optimizerD.step()

        #############################################
        # (2) Update G network: maximize log(D(G(z)))
        #############################################
        labelv = Variable(label.fill_(1)) # G want to generator real pictures
        p_fake_true = netD(output_image)
        errG = bce_loss(p_fake_true, labelv)
        # errG will backward later
        D_G_z2 = D_G_z2 + p_fake_true.data.mean()
        Loss_G = Loss_G + errG
        optimizerG.step()

        ##################################################################
        # (3) Update StyleBank(G) network: minimize content and style loss
        ##################################################################
        content_loss = CONTENT_WEIGHT * get_content_loss(loss_network, content_image, output_features)
        style_loss = STYLE_WEIGHT * get_style_loss(loss_network, stylized_image, output_features)
        reg_loss = REG_WEIGHT * get_regularization_loss(output_image)

        total_loss = content_loss + style_loss + reg_loss + errG
        total_loss.backward()
        optimizerG.step()

        Loss_C = Loss_C + content_loss.data.mean()
        Loss_S = Loss_S + style_loss.data.mean()
        Loss_R = Loss_R + reg_loss.data.mean()

        #########################################################
        # (4) Update StyleBank(G) network: minimize identity loss
        #########################################################
        t = t + 1
        if t == T + 1:
            t = 0
            sizes, sums = 0., 0.
            for param in netG.named_parameters():
                if  'style' in param[0]: continue
                sums += param[1].grad.abs().sum()

                p = 1
                for d in param[1].size(): p *= d
                sizes += p
            norm_style = sums / sizes
            
            optimizerG.zero_grad()
            
            output_image = netG(content_image)
            identity_loss = mse_loss(output_image, content_image)
            identity_loss.backward()
            Loss_I = Loss_I + identity_loss.data.mean()
            
            sizes,sums = 0.,0.
            for param in netG.named_parameters():
                if 'style' in param[0]: continue
                sums += param[1].grad.abs().sum()

                p = 1
                for d in param[1].size(): p *= d
                sizes += p
            norm_encoder = sums/sizes
            
            for param in netG.named_parameters():
                if 'style' in param[0]: continue
                param[1].grad = LAMBDA * norm_style / norm_encoder * param[1].grad 

            optimizerG.step()

        if batch_id % LOG_INTERVAL == 0:
            Loss_D, Loss_G, D_x, D_G_z1, D_G_z2 = [x / epoch_count
                                                   for x in (Loss_D, Loss_G, D_x, D_G_z1, D_G_z2)]
            Loss_I, Loss_S, Loss_C, Loss_R = [x / epoch_count
                                              for x in (Loss_I, Loss_S, Loss_C, Loss_R)]
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f\nLoss_C: %.4f Loss_S: %.4f Loss_R: %.4f Loss_I: %.4f'
                % (epoch, EPOCH, batch_id, len(dataloader),
                   Loss_D, Loss_G, D_x, D_G_z1, D_G_z2,
                   Loss_C, Loss_S, Loss_R, Loss_I))
            Loss_D, Loss_G, D_x, D_G_z1, D_G_z2 = 0, 0, 0, 0, 0
            Loss_I, Loss_S, Loss_C, Loss_R = 0, 0, 0, 0
            epoch_count = 0

            netG.eval()
            cs_dataset.test()
            _, content_image, stylized_image = cs_dataset.random_sample()
            content_image = content_image.unsqueeze(0)
            if use_gpu:
                content_image = content_image.cuda()
                stylized_image = stylized_image.cuda()
            content_image = Variable(content_image)
            images = [content_image.data, stylized_image.data]
            for i in range(len(dataset_list)):
                images.append(netG(content_image, i+1).data)
            save_test_image('result', '%04d_%05d.png'%(epoch, count),
                            images, ['original', 'stylized'] + dataset_list)
            netG.train()
            cs_dataset.train()

print('\ntraining end at', time.strftime('%Y_%m_%d_%H_%M', time.localtime()))
current_time = time.strftime('%Y_%m_%d_%H_%M', time.localtime())
torch.save(netG, os.path.join('models', current_time + '.pkl'))
