import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from data.aligned_dataset import AlignedDataset
from models.models import create_model
import util.util as util
import os
import numpy as np
import torch
from torch.autograd import Variable
import cv2
import torch.nn as nn
from resnet import Embedder
from unet import UNet, VGGExtractor, Discriminator
from tryon_net import G
import torch.nn.init as init
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.utils import make_grid

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
import utils

def normalize(x):
    x = ((x+1)/2).clamp(0,1)
    return x

SIZE = 320
NC = 14

lambdas_vis_reg = {'l1': 1.0, 'prc': 0.05, 'style': 100.0}
lambdas = {'adv': 0.25, 'identity': 500, 'mse': 50, 'vis_reg': .5, 'consist': 50}

def generate_label_plain(inputs):
    size = inputs.size()
    pred_batch = []
    for input in inputs:
        input = input.view(1, NC, 256, 192)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_batch = pred_batch.view(size[0], 1, 256, 192)

    return label_batch

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

def generate_label_color(inputs):
    label_batch = []
    for i in range(len(inputs)):
        label_batch.append(util.tensor2label(inputs[i], opt.label_nc))
    label_batch = np.array(label_batch)
    label_batch = label_batch * 2 - 1
    input_label = torch.from_numpy(label_batch)

    return input_label


def complete_compose(img, mask, label):
    label = label.cpu().numpy()
    M_f = label > 0
    M_f = M_f.astype(np.int)
    M_f = torch.FloatTensor(M_f).cuda()
    masked_img = img * (1 - mask)
    M_c = (1 - mask.cuda()) * M_f
    M_c = M_c + torch.zeros(img.shape).cuda()  ##broadcasting
    return masked_img, M_c, M_f


def compose(label, mask, color_mask, edge, color, noise):
    # check=check>0
    # print(check)
    masked_label = label * (1 - mask)
    masked_edge = mask * edge
    masked_color_strokes = mask * (1 - color_mask) * color
    masked_noise = mask * noise
    return masked_label, masked_edge, masked_color_strokes, masked_noise


def changearm(old_label):
    label = old_label
    arm1 = torch.FloatTensor((data['label'].cpu().numpy() == 11).astype(np.int))
    arm2 = torch.FloatTensor((data['label'].cpu().numpy() == 13).astype(np.int))
    noise = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.int))
    label = label * (1 - arm1) + arm1 * 4
    label = label * (1 - arm2) + arm2 * 4
    label = label * (1 - noise) + noise * 4
    return label


opt = TrainOptions().parse()

os.makedirs(os.path.join("checkpoints", opt.name), exist_ok=True)

iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
opt.distributed = n_gpu > 1
local_rank = opt.local_rank

if opt.distributed:
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    synchronize()

opt.batchSize = 16
opt.data_list = "test_files/one_person_different_cloth.txt"
dataset = AlignedDataset()
dataset.initialize(opt)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.nThreads))
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

total_steps = (start_epoch - 1) * dataset_size + epoch_iter

l1_criterion = nn.L1Loss()
mse_criterion = nn.MSELoss()
vgg_extractor = VGGExtractor().cuda().eval()
adv_criterion = utils.AdversarialLoss('lsgan').cuda()

opt.batch_size = opt.batchSize

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return
    model.load_state_dict(torch.load(checkpoint_path))
    model.cuda()

prev_model = create_model(opt)
prev_model.cuda()

model = G()
model.cuda()

if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
    load_checkpoint(model, opt.checkpoint)

model.eval()

step = 0

test_files_dir = "result_files_dir/" + opt.name
os.makedirs(test_files_dir, exist_ok=True)

pbar = tqdm(enumerate(data_loader), total=len(data_loader))

for i, data in pbar:
    iter_start_time = time.time()
    total_steps += opt.batchSize
    epoch_iter += opt.batchSize

    with torch.no_grad():

        t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float))
        mask_clothes = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int))
        mask_fore = torch.FloatTensor((data['label'].cpu().numpy() > 0).astype(np.int))
        img_fore = data['image'] * mask_fore
        img_fore_wc = img_fore * mask_fore
        all_clothes_label = changearm(data['label'])

        ############## Forward Pass ######################
        losses, transfer_1, prod_image, input_label, L1_loss, style_loss, clothes_mask, CE_loss, rgb, alpha = prev_model(
            Variable(data['label'].cuda()), Variable(data['edge'].cuda()), Variable(img_fore.cuda()),
            Variable(mask_clothes.cuda())
            , Variable(data['color'].cuda()), Variable(all_clothes_label.cuda()), Variable(data['image'].cuda()),
            Variable(data['pose'].cuda()), Variable(data['image'].cuda()), Variable(mask_fore.cuda()))

        _, transfer_2, prod_image_2, _, _, _, clothes_mask_2, _, gt_rgb_2, _ = prev_model(
            Variable(data['label'].cuda()), Variable(data['edge2'].cuda()), Variable(img_fore.cuda()),
            Variable(mask_clothes.cuda())
            , Variable(data['color2'].cuda()), Variable(all_clothes_label.cuda()), Variable(data['image'].cuda()),
            Variable(data['pose'].cuda()), Variable(data['image'].cuda()), Variable(mask_fore.cuda()))

        consistent_mask = (torch.abs(clothes_mask_2 - clothes_mask) < 0.1).float()

        gt_residual = ((torch.mean(data['image'].cuda(), dim=1) - torch.mean(transfer_2, dim=1)).unsqueeze(1)) * consistent_mask
        output_1 = model(transfer_1.detach(), gt_residual.detach())
        output_2 = model(transfer_2.detach(), gt_residual.detach())

    ### display output images
    seg_label_1 = generate_label_color(generate_label_plain(input_label)).float().cuda()
    prod_1 = prod_image.float().cuda()
    prod_1_mask = torch.cat([clothes_mask, clothes_mask, clothes_mask], 1)
    prod_2 = prod_image_2.float().cuda()

    # visuals = [[prod_1_mask.cpu(), torch.cat([clothes_mask_2, clothes_mask_2, clothes_mask_2], 1).cpu(),  data['image']],
    #            [data['color'], data['color2'], torch.cat([gt_residual, gt_residual, gt_residual], dim=1)],
    #            [transfer_1, output_1, (output_1 - transfer_1) / 2],
    #            [transfer_2, output_2, (output_2 - transfer_2) / 2]]

    output_residual = torch.cat([normalize(gt_residual), normalize(gt_residual), normalize(gt_residual)], dim=1).cpu()

    grid = make_grid(torch.cat(
        [output_residual.cpu(), normalize(transfer_1.cpu()), normalize(output_1.cpu()), normalize((output_1.cpu() - transfer_1.cpu()))], dim=0), nrow=transfer_1.shape[0])

    save_image(grid, os.path.join(test_files_dir, "test_{}.png".format(i)))