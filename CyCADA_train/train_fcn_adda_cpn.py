import logging
import os
import os.path
from collections import deque
import itertools
from datetime import datetime

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter

from PIL import Image
from torch.autograd import Variable

from cycada.data.adda_datasets import AddaDataLoader
from cycada.models import get_model
from cycada.models.models import models
from cycada.models import VGG16_FCN8s, Discriminator
from cycada.util import config_logging
from cycada.util import to_tensor_raw
from cycada.tools.util import make_variable
from UNet_CPN import UNet_CPN

def check_label(label, num_cls):
    "Check that no labels are out of range"
    label_classes = np.unique(label.numpy().flatten())
    label_classes = label_classes[label_classes < 255]
    if len(label_classes) == 0:
        print('All ignore labels')
        return False
    class_too_large = label_classes.max() > num_cls
    if class_too_large or label_classes.min() < 0:
        print('Labels out of bound')
        print(label_classes)
        return False
    return True


def forward_pass(net, discriminator, im, requires_grad=False, discrim_feat=False):
    if discrim_feat:
        score, feat = net(im)
        dis_score = discriminator(feat)
    else:
        score = net(im)
        dis_score = discriminator(score)
    if not requires_grad:
        score = Variable(score.data, requires_grad=False)

    return score, dis_score

def supervised_loss(score, label, weights=None):
    loss_fn_ = torch.nn.NLLLoss(weight=weights, size_average=True, 
            ignore_index=255)
    loss = loss_fn_(F.log_softmax(score, dim=1), label)
    return loss
   
def discriminator_loss(score, target_val, lsgan=False):
    if lsgan:
        loss = 0.5 * torch.mean((score - target_val)**2)
    else:
        _,_,h,w = score.size()
        target_val_vec = Variable(target_val * torch.ones(1,h,w),requires_grad=False).long().cuda()
        loss = supervised_loss(score, target_val_vec)
    return loss

def one_hot(label, num_classes):
    a,b,c = label.shape
    label_new = torch.cuda.FloatTensor(a, num_classes, b, c).fill_(0)
    for i in range(a):
        for j in range(num_classes):
            label_new[i,j:,:] = (label[i,:,:]==j)
    return label_new

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def seg_accuracy(score, label, num_cls):
    _, preds = torch.max(score.data, 1)
    hist = fast_hist(label.cpu().numpy().flatten(),
                     preds.cpu().numpy().flatten(), num_cls)
    intersections = np.diag(hist)
    unions = (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-8) * 100
    acc = np.diag(hist).sum() / hist.sum()
    return intersections, unions, acc

@click.command()
@click.argument('output')
@click.option('--dataset', required=True, multiple=True)
@click.option('--datadir', default="", type=click.Path(exists=True))
@click.option('--lr', '-l', default=0.0001)
@click.option('--momentum', '-m', default=0.9)
@click.option('--batch', default=1)
@click.option('--snapshot', '-s', default=5000)
@click.option('--downscale', type=int)
@click.option('--crop_size', default=None, type=int)
@click.option('--half_crop', default=None)
@click.option('--cls_weights', type=click.Path(exists=True))
@click.option('--weights_discrim', type=click.Path(exists=True))
@click.option('--weights_init', type=click.Path(exists=True))
@click.option('--model', default='fcn8s', type=click.Choice(models.keys()))
@click.option('--lsgan/--no_lsgan', default=False)
@click.option('--num_cls', type=int, default=19)
@click.option('--gpu', default='0')
@click.option('--max_iter', default=10000)
@click.option('--lambda_d', default=1.0)
@click.option('--lambda_g', default=1.0)
@click.option('--train_discrim_only', default=False)
@click.option('--discrim_feat/--discrim_score', default=False)
@click.option('--weights_shared/--weights_unshared', default=False)



def main(output, dataset, datadir, lr, momentum, snapshot, downscale, cls_weights, gpu,
        weights_init, num_cls, lsgan, max_iter, lambda_d, lambda_g,
        train_discrim_only, weights_discrim, crop_size, weights_shared,
        discrim_feat, half_crop, batch, model):

    # So data is sampled in consistent way
    np.random.seed(1337)
    torch.manual_seed(1337)
    logdir = 'runs/{:s}/{:s}_to_{:s}/lr{:.1g}_ld{:.2g}_lg{:.2g}'.format(model, dataset[0],
            dataset[1], lr, lambda_d, lambda_g)
    if weights_shared:
        logdir += '_weightshared'
    else:
        logdir += '_weightsunshared'
    if discrim_feat:
        logdir += '_discrimfeat'
    else:
        logdir += '_discrimscore'
    logdir += '/' + datetime.now().strftime('%Y_%b_%d-%H:%M')
    writer = SummaryWriter(log_dir=logdir)


    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    config_logging()
    print(model)
    net = get_model(model, num_cls=num_cls, pretrained=True, weights_init=weights_init,
                    output_last_ft=discrim_feat)

    loader = AddaDataLoader(net.transform, dataset, datadir, downscale,
                            crop_size=crop_size, half_crop=half_crop,
                            batch_size=batch, shuffle=True, num_workers=2)
    print('dataset', dataset)

    # Class weighted loss?
    weights = None

    # setup optimizers
    opt_rep = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)

    iteration = 0
    num_update_g = 0
    last_update_g = -1
    losses_super_s = deque(maxlen=100)
    losses_super_t = deque(maxlen=100)
    intersections = np.zeros([100, num_cls])
    unions = np.zeros([100, num_cls])
    accuracy = deque(maxlen=100)
    print('max iter:', max_iter)

    net.train()
    cpn = UNet_CPN()
    cpn.load_state_dict(torch.load('CPN_cycada.pth'))
    cpn.cuda()
    cpn.eval()
    for param in cpn.parameters():
        param.requires_grad = False

    while iteration < max_iter:
        for im_s, im_t, label_s, label_t in loader:

            if iteration > max_iter:
                break
            info_str = 'Iteration {}: '.format(iteration)

            if not check_label(label_s, num_cls):
                continue

            ###########################
            # 1. Setup Data Variables #
            ###########################
            im_s = make_variable(im_s, requires_grad=False)
            label_s = make_variable(label_s, requires_grad=False)
            im_t = make_variable(im_t, requires_grad=False)
            label_t = make_variable(label_t, requires_grad=False)

            #############################
            # 2. Optimize Discriminator #
            #############################

            # zero gradients for optimizer
            opt_rep.zero_grad()
            score_s = net(im_s)

            _, fake_label_s = torch.max(score_s, 1)
            _, fake_label_s = torch.max( cpn(im_t, one_hot(fake_label_s, 19)),1)
            loss_supervised_s = supervised_loss(score_s, label_s, weights=weights) + 0.5*supervised_loss(score_s, fake_label_s, weights=weights)

            score_t = net(im_t)

            _, fake_label_t = torch.max(score_t, 1)
            _, fake_label_t = torch.max(cpn(im_t, one_hot(fake_label_t,19)),1)
            loss_supervised_t = supervised_loss(score_t, fake_label_t, weights=weights)

            loss = loss_supervised_s + loss_supervised_t

            losses_super_t.append(loss_supervised_t.item())
            info_str += ' clsT:{:.2f}'.format(np.mean(losses_super_t))
            writer.add_scalar('loss/supervised/target', np.mean(losses_super_t), iteration)

            losses_super_s.append(loss_supervised_s.item())
            info_str += ' clsS:{:.2f}'.format(np.mean(losses_super_s))
            writer.add_scalar('loss/supervised/source', np.mean(losses_super_s), iteration)

            loss.backward()
            # optimize target net
            opt_rep.step()

            ###########################
            # Log and compute metrics #
            ###########################)
            if iteration % 1 == 0 and iteration > 0:
                # compute metrics
                intersection, union, acc = seg_accuracy(score_t, label_t.data, num_cls)
                intersections = np.vstack([intersections[1:, :], intersection[np.newaxis, :]])
                unions = np.vstack([unions[1:, :], union[np.newaxis, :]])
                accuracy.append(acc.item() * 100)
                acc = np.mean(accuracy)
                mIoU = np.mean(np.maximum(intersections, 1) / np.maximum(unions, 1)) * 100

                info_str += ' acc:{:0.2f}  mIoU:{:0.2f}'.format(acc, mIoU)
                writer.add_scalar('metrics/acc', np.mean(accuracy), iteration)
                writer.add_scalar('metrics/mIoU', np.mean(mIoU), iteration)
                logging.info(info_str)
            iteration += 1

            ################
            # Save outputs #
            ################

            # every 500 iters save current model
            if iteration % 500 == 0:
                os.makedirs(output, exist_ok=True)
                torch.save(net.state_dict(),
                           '{}/net-itercurr.pth'.format(output))

    writer.close()


if __name__ == '__main__':
    main()
