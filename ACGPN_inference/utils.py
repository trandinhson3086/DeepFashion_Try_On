import torch.nn.init as init
import torch
import torch.nn as nn
from torch import distributed
import torch.nn.functional as F
from torch.utils import data
import sys
from itertools import cycle

def normalize(x):
    x = ((x+1)/2).clamp(0,1)
    return x

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun

class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCEWithLogitsLoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()
        elif self.type == 'wgan-gp':
            if is_real:
                outputs = -outputs
            return outputs.mean()
        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            if self.type == 'nsgan':
                outputs = torch.sigmoid(outputs)
            loss = self.criterion(outputs, labels)
            return loss

def compute_perceptual_loss(x, y, criterion):
    loss = 0
    for a,b in zip(x, y):
        loss += criterion(a, b)
    return loss

def compute_gram(x):
    b, ch, h, w = x.size()
    f = x.view(b, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (h * w * ch)
    return G

def compute_style_loss(x, y, criterion):
    loss = 0
    for a,b in zip(x, y):
        loss += criterion(compute_gram(a), compute_gram(b))
    return loss

def compute_hole_valid_loss(output, gt, mask, criterion):
    hole_loss = criterion((1 - mask) * output, (1 - mask) * gt)
    valid_loss = criterion(mask * output, mask * gt)
    return hole_loss, valid_loss

def compute_transform_loss(warp_num, gt, mask, transforms, transformed_products, beta=10):
    transform_loss = 0
    for k in range(1, warp_num + 1):
        transform_loss += torch.mean(torch.abs(transformed_products[k-1] - ((1 - mask) * gt + mask)))
        warp_losses = []
        for i in range(k):
            foreground_l1_loss = torch.abs(transformed_products[i] - ((1 - mask) * gt + mask)) * (1 - mask) * beta
            # merge spatial dimension to one
            foreground_l1_loss = foreground_l1_loss.view(gt.size(0), -1).unsqueeze(2)
            warp_losses.append(foreground_l1_loss)

        warp_losses = torch.cat(warp_losses, dim=2)
        per_pixel_min_loss = torch.min(warp_losses, dim=2)[0]
        transform_loss += torch.mean(per_pixel_min_loss)

    transform_l2_loss = 0
    if warp_num > 1:
        for k in range(warp_num - 1):
            transform_l2_loss += F.mse_loss(transforms[0], transforms[k+1])

    total_loss = (transform_loss + 1e-2*transform_l2_loss) / warp_num

    return total_loss

def handler(signal_received, frame):
    # Handle any cleanup here
    print(signal_received)
    distributed.destroy_process_group()
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    sys.exit(0)
    
def single_gpu_flag(distributed, local_rank):
    return not distributed or (distributed and local_rank % torch.cuda.device_count() == 0)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)
    
def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)