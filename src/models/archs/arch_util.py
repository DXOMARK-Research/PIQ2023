import math
import collections.abc
import numpy as np
import torch
import os
from typing import Tuple
from urllib.parse import urlparse
from itertools import repeat
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
from torch.hub import download_url_to_file, get_dir


# --------------------------------------------
# IQA utils
    # Code taken from: https://github.com/chaofengc/IQA-PyTorch
# --------------------------------------------

def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.

    Ref: https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file

def dist_to_mos(dist_score: torch.Tensor) -> torch.Tensor:
    """Convert distribution prediction to mos score.
    For datasets with detailed score labels, such as AVA

    Args:
        dist_score (tensor): (*, C), C is the class number

    Output:
        mos_score (tensor): (*, 1)
    """
    num_classes = dist_score.shape[-1]
    mos_score = dist_score * torch.arange(1, num_classes + 1).to(dist_score)
    mos_score = mos_score.sum(dim=-1, keepdim=True)
    return mos_score


# --------------------------------------------
# Common utils
# --------------------------------------------


def load_pretrained_network(net, model_path, strict=True, weight_keys=None):
    if model_path.startswith('https://') or model_path.startswith('http://'):
        model_path = load_file_from_url(model_path)
    print(f'Loading pretrained model {net.__class__.__name__} from {model_path}')
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    if weight_keys:
        state_dict = state_dict[weight_keys]
    net.load_state_dict(state_dict, strict=strict)


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    r"""Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0.
        kwargs (dict): Other arguments for initialization function.

    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def symm_pad(im: torch.Tensor, padding: Tuple[int, int, int, int]):
    """Symmetric padding same as tensorflow.
    Ref: https://discuss.pytorch.org/t/symmetric-padding/19866/3
    """
    h, w = im.shape[-2:]
    left, right, top, bottom = padding
 
    x_idx = np.arange(-left, w+right)
    y_idx = np.arange(-top, h+bottom)
 
    def reflect(x, minx, maxx):
        """ Reflects an array around two points making a triangular waveform that ramps up
        and down,  allowing for pad lengths greater than the input length """
        rng = maxx - minx
        double_rng = 2*rng
        mod = np.fmod(x - minx, double_rng)
        normed_mod = np.where(mod < 0, mod+double_rng, mod)
        out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
        return np.array(out, dtype=x.dtype)

    x_pad = reflect(x_idx, -0.5, w-0.5)
    y_pad = reflect(y_idx, -0.5, h-0.5)
    xx, yy = np.meshgrid(x_pad, y_pad)
    return im[..., yy, xx]


def exact_padding_2d(x, kernel, stride=1, dilation=1, mode='same'):
    assert len(x.shape) == 4, f'Only support 4D tensor input, but got {x.shape}'
    kernel = to_2tuple(kernel)
    stride = to_2tuple(stride)
    dilation = to_2tuple(dilation)
    b, c, h, w = x.shape
    h2 = math.ceil(h / stride[0])
    w2 = math.ceil(w / stride[1])
    pad_row = (h2 - 1) * stride[0] + (kernel[0] - 1) * dilation[0] + 1 - h
    pad_col = (w2 - 1) * stride[1] + (kernel[1] - 1) * dilation[1] + 1 - w
    pad_l, pad_r, pad_t, pad_b = (pad_col // 2, pad_col - pad_col // 2, pad_row // 2, pad_row - pad_row // 2)

    mode = mode if mode != 'same' else 'constant'
    if mode != 'symmetric':
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode=mode)
    elif mode == 'symmetric':
        x = symm_pad(x, (pad_l, pad_r, pad_t, pad_b)) 

    return x


class ExactPadding2d(nn.Module):
    r"""This function calculate exact padding values for 4D tensor inputs,
    and support the same padding mode as tensorflow.

    Args:
        kernel (int or tuple): kernel size.
        stride (int or tuple): stride size.
        dilation (int or tuple): dilation size, default with 1.
        mode (srt): padding mode can be ('same', 'symmetric', 'replicate', 'circular')

    """

    def __init__(self, kernel, stride=1, dilation=1, mode='same'):
        super().__init__()
        self.kernel = to_2tuple(kernel)
        self.stride = to_2tuple(stride)
        self.dilation = to_2tuple(dilation)
        self.mode = mode

    def forward(self, x):
        return exact_padding_2d(x, self.kernel, self.stride, self.dilation, self.mode)
