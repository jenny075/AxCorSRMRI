import numpy as np

import torch.nn.functional as F
from torch.nn.functional import adaptive_avg_pool2d

import torch
import torch.nn as nn

from scipy import linalg
import piq

def calc_edge(fake_images, real_images, loss_fn, type_sobel='x'):
    if type_sobel == 'x':
        sobel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    else:
        sobel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    sobel_kernel = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(1, 1, 3, 3).cuda()
    dev_real = F.conv2d(real_images, sobel_kernel, padding=[1,1])
    dev_fake = F.conv2d(fake_images, sobel_kernel, padding=[1,1])
    loss = loss_fn(dev_real, dev_fake)
    return loss


def calc_edge_loss(fake_images, real_images, loss_fn):
    loss_x = calc_edge(fake_images, real_images, loss_fn, type_sobel='x')
    loss_y = calc_edge(fake_images, real_images, loss_fn, type_sobel='y')
    tot_loss = 0.5 * (loss_x + loss_y)
    return tot_loss

def SelfSupervisedLoss_torch(input,output,criterion,grad = False,):
    ds_volume = torch.nn.functional.interpolate(output,scale_factor=0.25, mode='bilinear', align_corners=True,
                                                 recompute_scale_factor=None)
    up_sample = nn.UpsamplingBilinear2d(scale_factor=4)
    up_volume = up_sample(ds_volume)
    if grad == False:
        loss = criterion(input.detach(),up_volume)
    else:
        loss = calc_edge_loss(input.detach(),up_volume,criterion)
    return loss


def calculate_activation_statistics(images, model, batch_size=250, dims=2048,
                                    cuda=True,MSID = False):
    model.eval()
    act_tot = []
    batches = int(len(images)/batch_size)
    last_batch = batch_size
    if batches > 0 and batch_size*batches < len(images):
        last_batch = len(images) - batch_size*batches
        batches = batches + 1
    if batches == 0:
        batches = 1
        last_batch = len(images)
    for i in range(batches):
        if i == batches - 1:
            act = np.empty((last_batch, dims))
        else:
            act = np.empty((batch_size, dims))

        if cuda:
            batch = images[i*batch_size:(i+1)*batch_size].cuda()
        else:
            batch = images
        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        act = pred.cpu().data.numpy().reshape(pred.size(0), -1)
        act_tot.append(act)
    act = np.concatenate(act_tot, 0)
    if MSID:
        return act

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def calculate_fretchet(images_real, images_fake, model):


    mu_1, std_1 = calculate_activation_statistics(images_real.unsqueeze(dim=1).repeat(1,3,1,1), model, cuda=True)
    mu_2, std_2 = calculate_activation_statistics(images_fake.unsqueeze(dim=1).repeat(1,3,1,1), model, cuda=True)

    """get fretched distance"""
    fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
    return fid_value

def calculate_KID(images_real, images_fake, model,ind=15):
    torch.manual_seed(ind)
    feature_real = calculate_activation_statistics(images_real.unsqueeze(dim=1).repeat(1,3,1,1), model, cuda=True,MSID=True)
    feture_fake = calculate_activation_statistics(images_fake.unsqueeze(dim=1).repeat(1, 3, 1, 1), model, cuda=True,MSID=True)

    KID = piq.KID()
    KID_value = KID(torch.from_numpy(feature_real),torch.from_numpy(feture_fake))
    return KID_value

def calculate_FID(images_real, images_fake, model):
    feature_real = calculate_activation_statistics(images_real.unsqueeze(dim=1).repeat(1,3,1,1), model, cuda=True,MSID=True)
    feture_fake = calculate_activation_statistics(images_fake.unsqueeze(dim=1).repeat(1, 3, 1, 1), model, cuda=True,
                                                  MSID=True)
    """get MSID value"""
    msid = piq.FID()
    FID_value = msid(torch.from_numpy(feature_real),torch.from_numpy(feture_fake))
    return FID_value