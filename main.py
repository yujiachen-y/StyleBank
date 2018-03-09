import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
import time

from dataloader import ImageLoader, ContentStyleDataset
from utils import tensor_normalizer, ZeroPadding, save_test_image
from loss_network import LossNetwork, get_content_loss, get_regularization_loss, get_style_loss, mse_loss
from training_network import StyleBankNet, Discriminator

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

dataloader = DataLoader(ContentStyleDataset(imageloaders, transform_list),
                        batch_size=BATCHSIZE, shuffle=True, num_workers=4)

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

for epoch in range(EPOCH):
    Loss_D, Loss_G, Loss_D_x, Loss_D_G_z1, Loss_D_G_z2 = 0, 0, 0, 0, 0
    Loss_I, Loss_S, Loss_C, Loss_R = 0, 0, 0, 0
    scheduler.step()

    for batch_id, (style_id, content_image, stylized_image) in enumerate(dataloader):

        optimizerG.zero_grad()
        optimizerD.zero_grad()
        
        batch_size = len(style_id)
        count += batch_size

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
        #############################################################

        # train with real
        p_real = netD(stylized_image)
        errD_real = bce_loss(p_real, labelv)
        errD_real.backward()
        D_x = p_real.data.mean()

        # train with fake
        labelv = Variable(label.fill_(0))
        p_fake = netD(output_image.detach())
        errD_fake = bce_loss(p_fake, labelv)
        errD_fake.backward()
        Loss_D_G_z1 = Loss_D_G_z1 + p_fake.data.mean()

        Loss_D = Loss_D + errD_fake.data.mean() + errD_real.data.mean()
        optimizerD.step()

        #############################################
        # (2) Update G network: maximize log(D(G(z)))
        #############################################
        labelv = Variable(label.fill_(1)) # G want to generator real pictures
        p_fake_true = netD(output_image)
        errG = bce_loss(p_fake_true, labelv)
        errG.backward()
        Loss_D_G_z2 = Loss_D_G_z2 + p_fake_true.data.mean()
        optimizerG.step()

        ##################################################################
        # (3) Update StyleBank(G) network: minimize content and style loss
        ##################################################################
        content_loss = CONTENT_WEIGHT * get_content_loss(loss_network, content_image, output_features)
        style_loss = STYLE_WEIGHT * get_style_loss(loss_network, stylized_image, output_features)
        reg_loss = REG_WEIGHT * get_regularization_loss(output_image)

        total_loss = content_loss + style_loss + reg_loss
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

            Loss_I = Loss_I + identity_loss.data.mean()

        if batch_id % LOG_INTERVAL == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f\nLoss_C: %.4f Loss_S: %.4f Loss_R: %.4f Loss_I: %.4f'
                % (epoch, EPOCH, batch_id, len(dataloader),
                    Loss_D, Loss_G, Loss_D_x, Loss_D_G_z1, Loss_D_G_z2,
                    Loss_C, Loss_S, Loss_R, Loss_I))
            Loss_D, Loss_G, Loss_D_x, Loss_D_G_z1, Loss_D_G_z2 = 0, 0, 0, 0, 0
            Loss_I, Loss_S, Loss_C, Loss_R = 0, 0, 0, 0

            netG.eval()
            output_image = netG(content_image, style_id)
            save_test_image('result',
                            '{}_{}_{}.png'.format(epoch, count, dataset_list[style_id[0]]),
                            [content_image.data, stylized_image.data, output_image.data])
            netG.train()

        current_time = time.strftime('%Y_%m_%d_%H_%M', time.localtime())
        torch.save(netG, os.path.join('models', current_time + '.pkl'))
