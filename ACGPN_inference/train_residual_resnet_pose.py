import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from visualization import board_add_images
import cv2
import torch.nn as nn
from resnet import Embedder
from tryon_net import G
from unet import VGGExtractor, Discriminator
import torch.nn.init as init
from tqdm import tqdm
import torch.nn.functional as F

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
import utils


SIZE = 320
NC = 14

lambdas_vis_reg = {'l1': 1.0, 'prc': 0.05, 'style': 100.0}
# lambdas = {'adv': 0.1, 'identity': 100, 'match_gt': 50, 'vis_reg': .5, 'consist': 50}
# lambdas = {'adv': 0.25, 'identity': 500, 'mse': 50, 'vis_reg': .5, 'consist': 50}
# lambdas = {'adv': 0.1, 'identity': 500, 'match_gt': 50, 'vis_reg': .1, 'consist': 50}
lambdas = {'adv': 0.1, 'identity': 500, 'match_gt': 50, 'vis_reg': .1, 'consist': 50}


def single_gpu_flag(args):
    return not args.distributed or (args.distributed and args.local_rank % torch.cuda.device_count() == 0)

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

if single_gpu_flag(opt):
    os.makedirs(os.path.join("checkpoints", opt.name), exist_ok=True)

if opt.distributed:
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    synchronize()

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

total_steps = (start_epoch - 1) * dataset_size + epoch_iter

l1_criterion = nn.L1Loss()
mse_criterion = nn.MSELoss()
vgg_extractor = VGGExtractor().cuda().eval()
adv_criterion = utils.AdversarialLoss('lsgan').cuda()

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return
    model.load_state_dict(torch.load(checkpoint_path))
    model.cuda()

def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.state_dict(), save_path)


if single_gpu_flag(opt):
    board = SummaryWriter(os.path.join('runs', opt.name))


prev_model = create_model(opt)
prev_model.cuda()


embedder_model = Embedder()
load_checkpoint(embedder_model, "../../cp-vton/checkpoints/identity_train_64_dim/step_020000.pth")
image_embedder = embedder_model.embedder_b.cuda()

model = G(input_dim=22)
model.apply(weights_init('kaiming'))
model.cuda()

if opt.use_gan:
    discriminator = Discriminator()
    discriminator.apply(utils.weights_init('gaussian'))
    discriminator.cuda()

if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
    load_checkpoint(model, opt.checkpoint)
    if opt.use_gan:
        load_checkpoint(discriminator, opt.checkpoint.replace("step_", "step_disc_"))

model_module = model
if opt.use_gan:
    discriminator_module = discriminator
if opt.distributed:
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank,
                                                      find_unused_parameters=True)
    model_module = model.module
    if opt.use_gan:
        discriminator = torch.nn.parallel.DistributedDataParallel(discriminator,
                                                                  device_ids=[local_rank],
                                                                  output_device=local_rank,
                                                                  find_unused_parameters=True)
        discriminator_module = discriminator.module

image_embedder.eval()
model.train()

# optimizer
optimizer = torch.optim.Adam(list(model.parameters()), lr=opt.lr, betas=(0.5, 0.999), weight_decay=1e-4)
if opt.use_gan:
    D_optim = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=1e-4)

step = 0

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    pbar = enumerate(dataset, start=epoch_iter)
    if single_gpu_flag(opt):
        pbar = tqdm(pbar)

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

        gt_residual = ((torch.mean(data['image'].cuda(), dim=1) - torch.mean(transfer_1, dim=1)).unsqueeze(1)) * consistent_mask
        output_1 = model(transfer_1.detach(), torch.cat([gt_residual.detach(), data['pose'].cuda()], dim=1))
        output_2 = model(transfer_2.detach(), torch.cat([gt_residual.detach(), data['pose'].cuda()], dim=1))

        embedding_1 = image_embedder(output_1)
        embedding_2 = image_embedder(output_2)
        embedding_1_t = image_embedder(transfer_1)
        embedding_2_t = image_embedder(transfer_2)

        if opt.use_gan:
            # train discriminator
            real_L_logit, real_L_cam_logit, real_G_logit, real_G_cam_logit = discriminator(F.interpolate(data['image'].cuda(), size=(256, 256), mode='bilinear'))
            fake_L_logit_1, fake_L_cam_logit_1, fake_G_logit_1, fake_G_cam_logit_1 = discriminator(F.interpolate(output_1, size=(256, 256), mode='bilinear').detach())
            fake_L_logit_2, fake_L_cam_logit_2, fake_G_logit_2, fake_G_cam_logit_2 = discriminator(F.interpolate(output_2, size=(256, 256), mode='bilinear').detach())

            D_true_loss = adv_criterion(real_L_logit, True) + \
                     adv_criterion(real_G_logit, True) + \
                     adv_criterion(real_L_cam_logit, True) + \
                     adv_criterion(real_G_cam_logit, True)
            D_fake_loss =  adv_criterion(torch.cat([fake_L_cam_logit_1, fake_L_cam_logit_2], dim=0), False) + \
                     adv_criterion(torch.cat([fake_G_cam_logit_1, fake_G_cam_logit_2], dim=0), False) + \
                     adv_criterion(torch.cat([fake_L_logit_1, fake_L_logit_2], dim=0), False) + \
                     adv_criterion(torch.cat([fake_G_logit_1, fake_G_logit_2], dim=0), False)

            D_loss = D_true_loss + D_fake_loss
            D_optim.zero_grad()
            D_loss.backward()
            D_optim.step()

            # train generator
            fake_L_logit_1, fake_L_cam_logit_1, fake_G_logit_1, fake_G_cam_logit_1 = discriminator(F.interpolate(output_1, size=(256, 256), mode='bilinear'))
            fake_L_logit_2, fake_L_cam_logit_2, fake_G_logit_2, fake_G_cam_logit_2 = discriminator(F.interpolate(output_2, size=(256, 256), mode='bilinear'))

            G_adv_loss = adv_criterion(torch.cat([fake_L_logit_1, fake_L_logit_2], dim=0), True) + \
                         adv_criterion(torch.cat([fake_G_logit_1, fake_G_logit_2], dim=0), True) + \
                         adv_criterion(torch.cat([fake_L_cam_logit_1, fake_L_cam_logit_2], dim=0), True) + \
                         adv_criterion(torch.cat([fake_G_cam_logit_1, fake_G_cam_logit_2], dim=0), True)

        # identity loss
        identity_loss = mse_criterion(embedding_1, embedding_1_t) + mse_criterion(embedding_2, embedding_2_t)

        # vis reg loss
        output_1_feats = vgg_extractor(output_1)
        transfer_1_feats = vgg_extractor(transfer_1)
        output_2_feats = vgg_extractor(output_2)
        transfer_2_feats = vgg_extractor(transfer_2)
        # gt_feats = vgg_extractor(data['image'].cuda())

        style_reg = utils.compute_style_loss(output_1_feats, transfer_1_feats, l1_criterion) + utils.compute_style_loss(output_2_feats, transfer_2_feats, l1_criterion)
        perceptual_reg = utils.compute_perceptual_loss(output_1_feats, transfer_1_feats, l1_criterion) + utils.compute_perceptual_loss(output_2_feats, transfer_2_feats, l1_criterion)
        l1_reg = l1_criterion(output_1, transfer_1) + l1_criterion(output_2, transfer_2)

        vis_reg_loss = l1_reg #* lambdas_vis_reg["l1"] + style_reg * lambdas_vis_reg["style"] + perceptual_reg * lambdas_vis_reg["prc"]

        # match gt loss
        match_gt_loss = l1_criterion(output_1, data['image'].cuda()) #* lambdas_vis_reg["l1"] + utils.compute_style_loss(output_1_feats, gt_feats, l1_criterion) * lambdas_vis_reg["style"] + utils.compute_perceptual_loss(output_1_feats, gt_feats, l1_criterion) * lambdas_vis_reg["prc"]

        # consistency loss
        consistency_loss = mse_criterion(transfer_1 - output_1, transfer_2 - output_2)


        ### display output images
        seg_label_1 = generate_label_color(generate_label_plain(input_label)).float().cuda()
        prod_1 = prod_image.float().cuda()
        prod_1_mask = torch.cat([clothes_mask, clothes_mask, clothes_mask], 1)
        prod_2 = prod_image_2.float().cuda()

        total_loss = lambdas['identity'] * identity_loss + \
                     lambdas['match_gt'] * match_gt_loss + \
                     lambdas['vis_reg'] * vis_reg_loss + \
                     lambdas['consist'] * consistency_loss

        if opt.use_gan:
            total_loss += lambdas['adv'] * G_adv_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if single_gpu_flag(opt):

            if (step + 1) % opt.display_freq == 0:
                visuals = [[prod_1_mask.cpu(), torch.cat([clothes_mask_2, clothes_mask_2, clothes_mask_2], 1).cpu(),  data['image']],
                           [data['color'], data['color2'], torch.cat([gt_residual, gt_residual, gt_residual], dim=1)],
                           [transfer_1, output_1, (output_1 - transfer_1)],
                           [transfer_2, output_2, (output_2 - transfer_2)]]
                # combine=c[0].squeeze()
                # cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
                # if step % 1 == 0:
                #     writer.add_image('combine', (combine.data + 1) / 2.0, step)
                #     rgb = (cv_img * 255).astype(np.uint8)
                #     bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                #     n = str(step) + '.jpg'
                #     cv2.imwrite(tmp_img_dir + data['name'][0], bgr)
                board_add_images(board, str(step + 1), visuals, step + 1)
            board.add_scalar('loss/total', total_loss.item(), step + 1)
            board.add_scalar('loss/identity', identity_loss.item(), step + 1)
            board.add_scalar('loss/vis_reg', vis_reg_loss.item(), step + 1)
            board.add_scalar('loss/match_gt', match_gt_loss.item(), step + 1)
            board.add_scalar('loss/consist', consistency_loss.item(), step + 1)
            if opt.use_gan:
                board.add_scalar('loss/Dadv', D_loss.item(),  step + 1)
                board.add_scalar('loss/Gadv', G_adv_loss.item(),  step + 1)

            pbar.set_description('step: %8d, loss: %.4f, identity: %.4f, vis_reg: %.4f, mse: %.4f, consist: %.4f'
                  % (step + 1, total_loss.item(), identity_loss.item(),
                     vis_reg_loss.item(), match_gt_loss.item(), consistency_loss.item()))

        if (step+1) % opt.save_count == 0 and single_gpu_flag(opt):
            save_checkpoint(model_module, os.path.join("checkpoints", opt.name, 'step_%06d.pth' % (step + 1)))
            if opt.use_gan:
                save_checkpoint(discriminator_module, os.path.join("checkpoints", opt.name, 'step_disc_%06d.pth' % (step + 1)))
        step += 1