import torch
import torch.nn as nn
from torch.optim import Adam, Adagrad, Adadelta

import numpy as np
import os

from data import Dataset
from config import device
from model import Generator, Discriminator

# pylint: disable=E1101,E1102

g = Generator().to(device)
g_opt = Adam(g.parameters())
d = Discriminator().to(device)
d_opt = Adagrad(d.parameters())
loss_func = nn.BCELoss()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

g.apply(weights_init_normal)
d.apply(weights_init_normal)



def generate(save_dir='output', save_png=False, save_mid=False):
    z = torch.randn(64, 64).to(device)
    gen_x = g(z)
    gen_x = gen_x.detach().reshape(64, 64, 64).to('cpu').numpy()
    os.makedirs(save_dir, exist_ok=True)
    if save_png:
        from PIL import Image
        imgs = (gen_x * 255).astype(np.uint8)
        path = os.path.join(save_dir, 'all.png')
        Image.fromarray(np.concatenate(imgs, 0).T).save(path)
    if save_mid:
        from notes import array_to_pm
        path = os.path.join(save_dir, 'all.mid')
        array_to_pm(np.concatenate(gen_x, 0)).write(path)


def load(path):
    state = torch.load(path)
    g.load_state_dict(state['g'])
    g_opt.load_state_dict(state['g_opt'])
    d.load_state_dict(state['d'])
    d_opt.load_state_dict(state['d_opt'])


def save(path):
    torch.save({
        'g': g.state_dict(), 'g_opt': g_opt.state_dict(),
        'd': d.state_dict(), 'd_opt': d_opt.state_dict()
    }, path)


def train(dataset_path, ratio=(1, 1)):
    g_loss_sum = d_loss_sum = 0

    for i, batch in enumerate(Dataset(dataset_path).batches(64)):
        real = torch.ones(64, 1).to(device)
        fake = torch.zeros(64, 1).to(device)

        z = torch.randn(64, 64, requires_grad=True).to(device)
        gen_x = g(z)
        gen_y = d(gen_x)

        g_loss = loss_func(gen_y, real)
        g_opt.zero_grad()
        (g_loss * ratio[0]).backward()
        g_opt.step()

        real_x = batch.reshape(64, 1, 64, 64)
        real_x = torch.tensor(real_x, requires_grad=True).to(device)
        real_y = d(real_x)
        fake_y = d(gen_x.detach())

        d_loss = (loss_func(real_y, real) + loss_func(fake_y, fake)) / 2
        d_opt.zero_grad()
        (d_loss * ratio[1]).backward()
        d_opt.step()

        g_loss_sum += g_loss.item()
        d_loss_sum += d_loss.item()

        if (i + 1) % 100 == 0:
            print(f'g: {g_loss_sum / 100}, d: {d_loss_sum / 100}')
            g_loss_sum = d_loss_sum = 0
