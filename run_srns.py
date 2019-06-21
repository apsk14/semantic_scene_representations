import argparse
import os, time, datetime

import torch
from torch import nn
import torchvision
import numpy as np
import cv2

from dataio import *
from torch.utils.data import DataLoader

from srns import *
from losses import *

from tensorboardX import SummaryWriter
from data_util import *
import util

import sys
torch.backends.cudnn.benchmark = True

# params
parser = argparse.ArgumentParser()

# data paths
parser.add_argument('--train_test', type=str, required=True, help='path to file list of h5 train data')
parser.add_argument('--data_root', required=True, help='path to file list of h5 train data')
parser.add_argument('--val_root', required=False, help='path to file list of h5 train data')
parser.add_argument('--logging_root', type=str, default='/media/staging/deep_sfm/',
                    required=False, help='path to file list of h5 train data')

# train params
parser.add_argument('--max_epoch', type=int, default=1501, help='number of epochs to train for')
parser.add_argument('--max_steps', type=int, default=None, help='number of epochs to train for')

# Intersection testing
parser.add_argument('--tracing_steps', type=int, default=10, help='Number of steps of intersection tester')

parser.add_argument('--experiment_name', type=str, default='', help='path to file list of h5 train data')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate, default=0.001')
parser.add_argument('--gan_weight', type=float, default=0., help='learning rate, default=0.001')
parser.add_argument('--l1_weight', type=float, default=200, help='learning rate, default=0.001')
parser.add_argument('--kl_weight', type=float, default=1, help='learning rate, default=0.001')
parser.add_argument('--proxy_weight', type=float, default=0, help='learning rate, default=0.001')
parser.add_argument('--reg_weight', type=float, default=1e-3, help='learning rate, default=0.001')

parser.add_argument('--steps_til_ckpt', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--steps_til_val', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--no_validation', action='store_true', default=False, help='#images')

# model params
parser.add_argument('--use_images', action='store_true', default=False, help='start epoch')
parser.add_argument('--img_sidelength', type=int, default=128, required=False, help='start epoch')

parser.add_argument('--num_val_objects', type=int, default=10, required=False, help='start epoch')
parser.add_argument('--num_val_images', type=int, default=10, required=False, help='start epoch')
parser.add_argument('--num_images', type=int, default=-1, required=False, help='start epoch')
parser.add_argument('--num_samples', type=int, default=64**2, required=False, help='start epoch')
parser.add_argument('--embedding_size', type=int, required=True, help='start epoch')
parser.add_argument('--mode', type=str, required=True, help='#images')

parser.add_argument('--dir_name', type=str, default=None, required=False, help='#images')

parser.add_argument('--overwrite_embeddings', action='store_true', default=False, help='#images')
parser.add_argument('--single_conditional', action='store_true', default=False, help='#images')
parser.add_argument('--freeze_rendering', action='store_true', default=False, help='#images')
parser.add_argument('--freeze_var', action='store_true', default=False, help='#images')

parser.add_argument('--no_preloading', action='store_true', default=False, help='#images')
parser.add_argument('--no_latent_reg_schedule', action='store_true', default=True, help='#images')
parser.add_argument('--num_objects', type=int, default=-1, help='start epoch')
parser.add_argument('--has_params', action='store_true', default=False, help='start epoch')

parser.add_argument('--no_gan', action='store_true', default=False, help='start epoch')
parser.add_argument('--gt_depth', action='store_true', default=False, help='start epoch')
parser.add_argument('--depth_supervision', action='store_true', default=False, help='start epoch')

parser.add_argument('--orthographic', action='store_true', default=False, help='start epoch')

# For retraining
parser.add_argument('--checkpoint', default=None, help='model to load')
parser.add_argument('--start_step', type=int, default=0, help='start step')
parser.add_argument('--batch_size', type=int, default=4, help='start epoch')

parser.add_argument('--renderer', type=str, default='fc', help='start epoch')

# 2d/3d
parser.add_argument('--implicit_nf', type=int, default=256, help='start epoch')


opt = parser.parse_args()
print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

device = torch.device('cuda')


def train(model, dataset, val_dataset):
    collate_fn = dataset.collate_fn
    dataloader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            drop_last=True,
                            collate_fn=collate_fn)

    if val_dataset is not None:
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=2,
                                    shuffle=False,
                                    drop_last=True,
                                    collate_fn=collate_fn)

    model.train()
    model.cuda()

    if not opt.no_gan:
        discriminator = PatchDiscriminator(input_nc=3, ndf=28, n_layers=3)
        criterionGAN = GANCriterion()

        discriminator.train()
        discriminator.cuda()
        criterionGAN.cuda()
    else:
        discriminator = None


    # directory structure: month_day/
    if opt.dir_name is None:
        dir_name = os.path.join(datetime.datetime.now().strftime('%m_%d'),
                                datetime.datetime.now().strftime('%H-%M-%S_')) + params_to_filename(opt)[:200]
    else:
        dir_name = opt.dir_name

    print(dir_name)

    log_dir = os.path.join(opt.logging_root, 'logs', dir_name)
    run_dir = os.path.join(opt.logging_root, 'runs', dir_name)
    util.cond_mkdir(log_dir)
    util.cond_mkdir(run_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    if opt.checkpoint is not None:
        print("Loading model from %s"%opt.checkpoint)
        util.custom_load(model, path=opt.checkpoint,
                         discriminator=discriminator,
                         optimizer=optimizer,
                         overwrite_embeddings=opt.overwrite_embeddings)


    if discriminator is not None :
        optimizerD = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

    writer = SummaryWriter(run_dir)
    iter = opt.start_step
    start_epoch = iter // len(dataloader)
    step = 0

    writer.add_scalar('Learning rate', opt.lr)
    writer.add_scalar('l1 weight', opt.l1_weight)

    if not opt.no_latent_reg_schedule:
        kl_weight_sched = util.LinearSchedule(begin=0., end=opt.kl_weight, num_steps=100000)

    with torch.autograd.set_detect_anomaly(True):
        print('Beginning training...')
        for epoch in range(start_epoch, opt.max_epoch):
            for model_input, ground_truth in dataloader:
                model_outputs = model(model_input)

                optimizer.zero_grad()

                dist_loss = model.get_distortion_loss(model_outputs, ground_truth)
                reg_loss = model.get_regularization_loss(model_outputs, ground_truth)
                proxy_loss = model.get_proxy_losses(model_outputs, ground_truth)
                var_loss = model.get_variational_loss()

                #####
                # Compute the GAN loss
                if discriminator is not None:
                    optimizerD.zero_grad()

                    pred_gan_in, real_gan_in = model.prep_for_gan(model_outputs, ground_truth)

                    # Fake forward step
                    pred_fake = discriminator.forward(pred_gan_in.detach())  # Detach to make sure no gradients go into generator
                    loss_d_fake = criterionGAN(pred_fake, False)

                    # Real forward step
                    pred_real = discriminator.forward(real_gan_in)
                    loss_d_real = criterionGAN(pred_real, True)

                    # Combined Loss
                    loss_d_gan = (loss_d_fake + loss_d_real) * 0.5

                    # Try to fake discriminator
                    pred_fake = discriminator.forward(pred_gan_in)
                    loss_g_gan = criterionGAN(pred_fake, True)

                    loss_d_gan.backward()
                    optimizerD.step()
                else:
                    loss_g_gan = 0
                    loss_d_gan = 0

                weighted_proxy_loss = opt.proxy_weight * proxy_loss
                # weighted_proxy_loss = opt.proxy_weight * max(0, ((opt.max_epoch/2 - epoch)/(opt.max_epoch/2))) * proxy_loss
                # weighted_kl_loss = var_loss * (kl_weight_sched[step] if not opt.no_latent_reg_schedule else opt.kl_weight)
                weighted_kl_loss = var_loss * opt.kl_weight

                gen_loss_total = opt.l1_weight * dist_loss + \
                                 opt.gan_weight * loss_g_gan + \
                                 opt.reg_weight * reg_loss + \
                                 weighted_proxy_loss + \
                                 weighted_kl_loss

                #####
                # Backward passes
                gen_loss_total.backward(retain_graph=False)

                # Log parameter norms
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_scalar("_" + name + "_grad_norm", param.grad.data.norm(2).item(), iter)

                optimizer.step()

                print("Iter %07d   Epoch %03d   dist_loss %0.4f   KL %0.4f   loss_gen %0.4f   loss_discrim %0.4f    reg_loss %0.4f    proxy_loss %0.4f" %
                      (iter, epoch, dist_loss*opt.l1_weight, weighted_kl_loss, loss_g_gan, loss_d_gan, reg_loss*opt.reg_weight, weighted_proxy_loss))

                model.write_updates(writer, model_outputs, ground_truth, iter)
                writer.add_scalar("generator_loss", loss_g_gan, iter)
                writer.add_scalar("discrim_loss", loss_d_gan, iter)
                writer.add_scalar("scaled_distortion_loss", dist_loss*opt.l1_weight, iter)
                writer.add_scalar("combined_generator_loss", torch.clamp(gen_loss_total, 0, 1e3), iter)
                writer.add_scalar("scaled_reg_loss", reg_loss * opt.reg_weight, iter)
                writer.add_scalar("reg_loss", reg_loss)
                writer.add_scalar("scaled_proxy_loss", weighted_proxy_loss, iter)
                writer.add_scalar("proxy_loss", proxy_loss, iter)
                writer.add_scalar("scaled_kl_loss", weighted_kl_loss, iter)
                writer.add_scalar("kl_loss", var_loss, iter)
                writer.add_scalar("kl_weight", kl_weight_sched[step] if not opt.no_latent_reg_schedule else opt.kl_weight, iter)

                if not iter:
                    # Save parameters used into the log directory.
                    with open(os.path.join(log_dir, "params.txt"), "w") as out_file:
                        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))
                    with open(os.path.join(run_dir, "params.txt"), "w") as out_file:
                        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

                    # Save a text versin of the model into the log directory.
                    with open(os.path.join(log_dir, "model.txt"), "w") as out_file:
                        out_file.write(str(model))
                    with open(os.path.join(run_dir, "model.txt"), "w") as out_file:
                        out_file.write(str(model))

                if iter % opt.steps_til_val == 0 and val_dataset is not None:
                    print("*"*100)
                    print("Running validation set...")
                    print("*"*100)

                    model.eval()
                    with torch.no_grad():
                        psnrs = []
                        ssims = []
                        dist_losses = []
                        for model_input, ground_truth in val_dataloader:
                            model_outputs = model(model_input)

                            dist_loss = model.get_distortion_loss(model_outputs, ground_truth).cpu().numpy()
                            psnr, ssim = model.get_psnr(model_outputs, ground_truth)
                            psnrs.append(psnr)
                            ssims.append(ssim)
                            dist_losses.append(dist_loss)

                            model.write_updates(writer, model_outputs, ground_truth, iter, prefix='val_')

                        writer.add_scalar("val_dist_loss", np.mean(dist_losses), iter)
                        writer.add_scalar("val_psnr", np.mean(psnrs), iter)
                        writer.add_scalar("val_ssim", np.mean(ssims), iter)
                    model.train()

                iter += 1
                step += 1

                if iter == opt.max_steps:
                    break

                if iter % opt.steps_til_ckpt == 0:
                    util.custom_save(model,
                                     os.path.join(log_dir, 'epoch_%04d_iter_%06d.pth'%(epoch, iter)),
                                     discriminator=discriminator,
                                     optimizer=optimizer)

            if iter == opt.max_steps:
                break

    final_ckpt_path = os.path.join(log_dir, 'epoch_%04d_iter_%06d.pth'%(epoch, iter))
    util.custom_save(model,
                     final_ckpt_path,
                     discriminator=discriminator,
                     optimizer=optimizer)

    return final_ckpt_path


def test(model, dataset):
    collate_fn = dataset.collate_fn
    dataset = DataLoader(dataset,
                         collate_fn=collate_fn,
                         batch_size=1,
                         shuffle=False,
                         drop_last=True)

    if opt.checkpoint is not None:
        print("Loading model from %s"%opt.checkpoint)
        util.custom_load(model, path=opt.checkpoint,
                         discriminator=None,
                         overwrite_embeddings=opt.overwrite_embeddings)
    else:
        print("Have to give checkpoint!")
        return

    model.eval()
    model.cuda()

    # directory structure: month_day/
    dir_name = os.path.join(datetime.datetime.now().strftime('%m_%d'),
                            datetime.datetime.now().strftime('%H-%M-%S_') +
                            '_'.join(opt.checkpoint.strip('/').split('/')[-2:])[:200] + '_'
                            + opt.data_root.strip('/').split('/')[-1])

    traj_dir = os.path.join(opt.logging_root, 'test_traj', dir_name)
    comparison_dir = os.path.join(opt.logging_root, 'test_traj', dir_name, 'comparison')
    cond_mkdir(comparison_dir)
    cond_mkdir(traj_dir)

    # Save parameters used into the log directory.
    with open(os.path.join(traj_dir, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    print('Beginning evaluation...')
    with torch.no_grad():
        for idx, (model_input, ground_truth) in enumerate(dataset):
            if not idx%10:
                print(idx)

            model_outputs = model(model_input)
            model.write_eval(model_outputs, ground_truth, os.path.join(traj_dir, "%06d.png"%idx))
            model.write_comparison(model_outputs, ground_truth, os.path.join(comparison_dir, "%06d.png"%idx))
        # util.concat_vid_frames(traj_dir, os.path.join(traj_dir, 'stitched.mp4'))



def main():
    if opt.train_test == 'train':
        if opt.use_images:
            dataset = RayBundleDataset(root_dir=opt.data_root,
                                       preload=not opt.no_preloading,
                                       num_objects=opt.num_objects,
                                       num_images=opt.num_images,
                                       num_samples=-1,
                                       img_sidelength=opt.img_sidelength,
                                       samples_per_object=1,
                                       mode='val')
        else:
            dataset = RayBundleDataset(root_dir=opt.data_root,
                                       preload=not opt.no_preloading,
                                       num_objects=opt.num_objects,
                                       num_images=opt.num_images,
                                       samples_per_object=1,
                                       num_samples=opt.num_samples)

        if not opt.no_validation:
            val_dataset = RayBundleDataset(root_dir=opt.val_root,
                                           preload=not opt.no_preloading,
                                           num_objects=opt.num_val_objects,
                                           num_images=opt.num_val_images,
                                           img_sidelength=opt.img_sidelength,
                                           num_samples=-1,
                                           samples_per_object=1,
                                           mode='val')
        else:
            val_dataset = None
        # model = DeepRayModel(num_objects=2433,
        # model = DeepRayModel(num_objects=4612,
        model = SRNsModel(num_objects=dataset.num_obj,
                          embedding_size=opt.embedding_size,
                          implicit_nf=opt.implicit_nf,
                          has_params=opt.has_params,
                          mode=opt.mode,
                          renderer=opt.renderer,
                          depth_supervision=opt.depth_supervision,
                          freeze_rendering=opt.freeze_rendering,
                          orthographic=opt.orthographic,
                          freeze_var=opt.freeze_var,
                          use_gt_depth=opt.gt_depth,
                          tracing_steps=opt.tracing_steps)
        final_ckpt_path = train(model, dataset, val_dataset)
        sys.stdout.write(final_ckpt_path)
    elif opt.train_test == 'test':
        dataset = RayBundleDataset(root_dir=opt.data_root,
                                   preload=not opt.no_preloading,
                                   num_objects=opt.num_objects,
                                   num_images=-1,
                                   num_samples=-1,
                                   samples_per_object=1,
                                   img_sidelength=opt.img_sidelength,
                                   mode='val')
        # model = DeepRayModel(num_objects=2433,
        model = SRNsModel(num_objects=dataset.num_obj,
                          embedding_size=opt.embedding_size,
                          implicit_nf=opt.implicit_nf,
                          has_params=opt.has_params,
                          renderer=opt.renderer,
                          mode=opt.mode,
                          use_gt_depth=opt.gt_depth,
                          tracing_steps=opt.tracing_steps)
        test(model, dataset)
    else:
        print("Unknown mode.")
        return None



if __name__ == '__main__':
    main()
