# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist
from torch._six import inf


def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def load_pretrained(config, model, logger):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}

    save_path = os.path.join(config.OUTPUT, f'model_best.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

def save_checkpoint_llm(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger):
    model_state_dict = model.state_dict()
    trainable = {}
    for n, p in model_state_dict.items():
        if 'adapter' in n:
            trainable[n] = p
        if 'backbone' in n:
            trainable[n] = p
        if 'output' in n:
            trainable[n] = p
        if 'prompt' in n:
            trainable[n] = p

    save_state = {'model': trainable,
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}

    save_path = os.path.join(config.OUTPUT, f'model_best.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
import torch
import torch.nn as nn
from torch.nn import Conv2d
import torch.nn.functional as F
import torchvision.transforms as transforms
import math
import numpy as np
import random
import collections
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
class RandConvModule(nn.Module):
    def __init__(self, kernel_size=3, in_channels=3, hidden_channels=2,
                 mixing=True,
                 identity_prob=0.5,
                 ):
        """

        :param mixing: "random": output = (1-alpha)*input + alpha* randconv(input) where alpha is a random number sampled
                            from a distribution defined by res_dist
        """
        super(RandConvModule, self).__init__()
        self.randconv = RandConv2d(in_channels=in_channels, hidden_channels=hidden_channels, kernel_size=kernel_size)
        # mixing mode
        self.mixing = mixing # In the mixing mode, a mixing connection exists between input and output of random conv layer
        if self.mixing:
            self.alpha = random.random()  # sample mixing weights from uniform distributin (0, 1)

        self.identity_prob = identity_prob  # the probability that use original input
        self.tr=transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    def forward(self, input):
        """assume that the input is whightened"""
        ######## random conv ##########
        if not (self.identity_prob > 0 and torch.rand(1) < self.identity_prob):
            # whiten input and go through randconv
            output = self.randconv(input)
            output = (self.alpha*output + (1-self.alpha)*input)
            b,c,h,w=output.shape
            output=output.view(output.size(0), -1)
            input=input.view(input.size(0), -1)
            output = F.normalize(output, dim=1)*torch.norm(input, dim=1, keepdim=True)
            output =output.view(b, c, h, w)
            output=self.tr(output)
        else:
            output = input
        return output

    def parameters(self, recurse=True):
        return self.randconv.parameters()

    def trainable_parameters(self, recurse=True):
        return self.randconv.trainable_parameters()

    def whiten(self, input):
        return (input - self.data_mean) / self.data_std

    def dewhiten(self, input):
        return input * self.data_std + self.data_mean

    def randomize(self):
        self.randconv.randomize()
        self.alpha = random.random()

    def set_test_res_weight(self, w):
        self.res_test_weight = w

# 对模型的权重进行高斯初始化
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=1)
        nn.init.constant_(m.bias, 0)

class RandConv2d(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size):
        super(RandConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.leaklyrelu=nn.LeakyReLU()

    def randomize(self):
        self.apply(weights_init)

    def forward(self, input):
        output = self.conv2(self.leaklyrelu(self.conv1(input)))
        return output


import torch.distributed as dist
from PIL import ImageFilter, ImageOps

from PIL import Image, ImageChops, ImageMath
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
class GaussianBlurimgs(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img1, img2):
        do_it = random.random() <= self.prob
        if not do_it:
            return img1, img2

        radius = random.uniform(self.radius_min, self.radius_max)
        return img1.filter(ImageFilter.GaussianBlur(radius=radius)), img2
class GaussianBlurMaskimgs(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img1, img2):
        do_it = random.random() <= self.prob
        if not do_it:
            return img1, img2

        radius = random.uniform(self.radius_min, self.radius_max)
        img11=img1.filter(ImageFilter.GaussianBlur(radius=radius))
        mask_np=np.expand_dims(np.array(img2)/255., axis=2)
        mask_img11=np.array(img11)*(1-mask_np)+np.array(img11)*(mask_np)
        result_image = Image.fromarray(mask_img11.astype(np.uint8))
        return result_image, img2
class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
class random_bbox_maskimgs(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p, block_size = 20, num_blocks = 10):
        self.p = p
        self.block_size=block_size
        self.num_blocks=num_blocks

    def __call__(self, img1, img2):
        do_it = random.random() <= self.p
        if not do_it:
            return img1, img2
        width, height = img1.size
        mask = np.ones((height, width, 1), dtype=np.uint8)
        mask_np = np.expand_dims(np.array(img2) / 255., axis=2)

        for _ in range(self.num_blocks):
            x = random.randint(0, width - self.block_size)
            y = random.randint(0, height - self.block_size)
            mask[y:y + self.block_size, x:x + self.block_size, :] = 0
            mask = mask * (1 - mask_np) + mask_np
            img1 = np.array(img1) * (mask)
        result_image = Image.fromarray(img1.astype(np.uint8))
        return result_image, img2
class Solarizationimgs(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img1, img2):
        if random.random() < self.p:
            return ImageOps.solarize(img1), img2
        else:
            return img1, img2
def random_bbox(config, batch_size):
    """Generate a random tlhw with configuration.
    Args:
        config: Config should have configuration including img

    Returns:
        tuple: (top, left, height, width)

    """
    img_height, img_width, _ = config['image_shape']
    h, w = config['mask_shape']
    margin_height, margin_width = config['margin']
    maxt = img_height - margin_height - h
    maxl = img_width - margin_width - w
    bbox_list = []
    if config['mask_batch_same']:
        t = np.random.randint(margin_height, maxt)
        l = np.random.randint(margin_width, maxl)
        bbox_list.append((t, l, h, w))
        bbox_list = bbox_list * batch_size
    else:
        for i in range(batch_size):
            t = np.random.randint(margin_height, maxt)
            l = np.random.randint(margin_width, maxl)
            bbox_list.append((t, l, h, w))
    return torch.tensor(bbox_list, dtype=torch.int64)
def bbox2mask(bboxes, height, width, max_delta_h, max_delta_w):
    batch_size = bboxes.size(0)
    mask = torch.zeros((batch_size, 1, height, width), dtype=torch.float32)
    for i in range(batch_size):
        bbox = bboxes[i]
        delta_h = np.random.randint(max_delta_h // 2 + 1)
        delta_w = np.random.randint(max_delta_w // 2 + 1)
        mask[i, :, bbox[0] + delta_h:bbox[0] + bbox[2] - delta_h, bbox[1] + delta_w:bbox[1] + bbox[3] - delta_w] = 1.
    return mask



def test_bbox2mask():
    image_shape = [256, 256, 3]
    mask_shape = [128, 128]
    margin = [0, 0]
    max_delta_shape = [32, 32]
    bbox = random_bbox(image_shape)
    mask = bbox2mask(bbox, image_shape[0], image_shape[1], max_delta_shape[0], max_delta_shape[1])
    return mask