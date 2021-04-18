# -*- coding:utf-8 -*-
__author__ = 'Mingqi Yuan'
"""
Implementation of the end-to-end curiosity module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

def process(tensor, normalize=False, range=None, scale_each=False):
    """Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    return tensor

class CuriosityNet(nn.Module):
    def __init__(self, kwargs):
        super(CuriosityNet, self).__init__()

        self.conv1 = nn.Conv2d(kwargs['input_dim'][0], 32, 8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.flatten = nn.Flatten()
        self.linear_decoder = nn.Linear(32 * 7 * 7 + kwargs['input_dim'][1], 32*11*11)

        self.conv4 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.ConvTranspose2d(32, kwargs['output_dim'], 3, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)
        nn.init.xavier_uniform_(self.conv6.weight)

    def forward(self, state, action):
        x = self.bn1(self.conv1(state))
        x = self.leaky_relu(x)
        x = self.bn2(self.conv2(x))
        x = self.leaky_relu(x)
        x = self.bn3(self.conv3(x))
        x = self.leaky_relu(x)
        x = self.flatten(x)
        x = self.linear_decoder(torch.cat([x, action], dim=1))
        x = self.bn4(self.conv4(x.view(-1, 32, 11, 11), output_size=(21, 21)))
        x = self.leaky_relu(x)
        x = self.bn5(self.conv5(x, output_size=(42, 42)))
        x = self.leaky_relu(x)
        output = self.conv6(x, output_size=(84, 84))

        return output

class ECM:
    def __init__(
            self,
            envs,
            device,
            lr,
            batch_size
    ):
        self.envs = envs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

        ecm_kwargs = {
            'input_dim': [envs.observation_space.shape[0], envs.action_space.n],
            'output_dim': envs.observation_space.shape[0]
        }
        self.curiosity = CuriosityNet(kwargs=ecm_kwargs)
        self.optimizer_cm = optim.Adam(self.curiosity.parameters(), self.lr)

    def train(self, gt_loader, epoch, epislon=0.1):
        updates = len(gt_loader)
        total_cm_loss = 0.
        for idx, g_data in enumerate(gt_loader):
            self.optimizer_cm.zero_grad()

            g_states, g_actions, g_next_states = g_data
            g_states = g_states.to(self.device)
            g_actions = g_actions.to(self.device)
            g_next_states = g_next_states.to(self.device)
            ''' epislon greedy '''
            g_actions = torch.where(
                torch.rand_like(g_actions.float()) < epislon,
                torch.randint_like(g_actions, low=0, high=self.envs.action_space.n),
                g_actions
            )

            ''' reconstruct the transition '''
            g_actions = F.one_hot(g_actions.squeeze(1), self.envs.action_space.n)
            pred_next_states = self.curiosity(g_states, g_actions)
            cm_loss = 0.5 * F.mse_loss(pred_next_states, g_next_states)
            cm_loss.backward()
            self.optimizer_cm.step()

            total_cm_loss += cm_loss.item()

        total_cm_loss = total_cm_loss / updates

        return total_cm_loss


    def get_irs(self, next_states, pred_next_states):
        intrinsic_rewards = F.mse_loss(
            process(next_states, normalize=True, range=(-1, 1)),
            process(pred_next_states, normalize=True, range=(-1, 1)),
            reduction='mean')

        return intrinsic_rewards

