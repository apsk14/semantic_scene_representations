import argparse
import os, time, datetime

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dataio import *
from torch.utils.data import DataLoader
from srns_vincent import *
import util

NUM_CLASSES = 6

torch.backends.cudnn.benchmark = True

# params
parser = argparse.ArgumentParser()

parser.add_argument('--train_test', type=str, required=True, help='whether to train or evaluate a model')
parser.add_argument('--latent_only', type=str, required=False, help='whether to only optimize the latent codes')
parser.add_argument('--data_root', required=True, help='path to file list of h5 train data')
parser.add_argument('--val_root', required=False, help='path to file list of h5 train data')
parser.add_argument('--logging_root', type=str, default='/media/staging/deep_sfm/',
                    required=False, help='path to file list of h5 train data')

parser.add_argument('--max_epoch', type=int, default=1501, help='number of epochs to train for')
parser.add_argument('--max_steps', type=int, default=None, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=4e-4, help='learning rate, default=0.001')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Training batch size.')

parser.add_argument('--img_weight', type=float, default=200,
                    help='learning rate, default=0.001')
parser.add_argument('--class_weight', type=float, default=7,
                    help='learning rate, default=0.001')
parser.add_argument('--latent_weight', type=float, default=1,
                    help='learning rate, default=0.001')
parser.add_argument('--reg_weight', type=float, default=1e-3,
                    help='Weight for regularization term in paper.')

parser.add_argument('--tracing_steps', type=int, default=10,
                    help='Number of steps of intersection tester')

parser.add_argument('--steps_til_ckpt', type=int, default=5000,
                    help='Number of iterations until checkpoint is saved.')
parser.add_argument('--steps_til_val', type=int, default=1000,
                    help='Number of iterations until validation set is run.')
parser.add_argument('--no_validation', action='store_true', default=False,
                    help='If no validation set should be used.')

parser.add_argument('--img_sidelength', type=int, default=128, required=False,
                    help='Sidelength of training images. If original images are bigger, they\'re downsampled.')

parser.add_argument('--preload', action='store_true', default=False,
                    help='Whether to preload data to RAM.')

parser.add_argument('--max_num_instances_train', type=int, default=-1,
                    help='If \'train_root\' has more instances, only the first max_num_instances_train are used')
parser.add_argument('--max_num_observations_train', type=int, default=50, required=False,
                    help='f gn instance has more observations, only the first max_num_observations_train are used')
parser.add_argument('--max_num_instances_val', type=int, default=10, required=False,
                    help='If \'val_root\' has more instances, only the first max_num_instances_val are used')
parser.add_argument('--max_num_observations_val', type=int, default=10, required=False,
                    help='Maximum numbers of observations per validation instance')
parser.add_argument('--specific_observation_idcs', type=str, default=None,
                    help='Only pick a subset of specific observations for each instance.')

parser.add_argument('--has_params', action='store_true', default=False,
                    help='Whether each object instance already comes with its own parameter vector.')

parser.add_argument('--checkpoint', default=None,
                    help='Checkpoint to trained model.')
parser.add_argument('--overwrite_embeddings', action='store_true', default=False,
                    help='When loading from checkpoint: Whether to discard checkpoint embeddings and initialize at random.')
parser.add_argument('--start_step', type=int, default=0,
                    help='If continuing from checkpoint, which iteration to start counting at.')

parser.add_argument('--use_unet_renderer', action='store_true',
                    help='Whether to use a DeepVoxels-style unet as rendering network or a per-pixel 1x1 convnet')
parser.add_argument('--embedding_size', type=int, default=256,
                    help='Dimensionality of latent embedding.')


opt = parser.parse_args()
print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

device = torch.device('cuda')


def train(model,
          dataset,
          val_dataset,
          batch_size,
          checkpoint=None,
          dir_name=None,
          max_steps=None):
    collate_fn = dataset.collate_fn
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            collate_fn=collate_fn,
                            pin_memory=opt.preload)

    if val_dataset is not None:
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=2,
                                    shuffle=False,
                                    drop_last=True,
                                    collate_fn=collate_fn)
    model.train()
    model.cuda()

    dir_name = os.path.join(datetime.datetime.now().strftime('%m_%d'),
                            datetime.datetime.now().strftime('%H-%M-%S_'))

    log_dir = os.path.join(opt.logging_root, 'logs', dir_name)
    run_dir = os.path.join(opt.logging_root, 'runs', dir_name)
    util.cond_mkdir(log_dir)
    util.cond_mkdir(run_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    if checkpoint is not None:
        print("Loading model from %s"%opt.checkpoint)
        util.custom_load_linear(model, path=opt.checkpoint,
                         discriminator=None,
                         optimizer=None,
                         overwrite_embeddings=opt.overwrite_embeddings)

    writer = SummaryWriter(run_dir)
    iter = opt.start_step
    print(len(dataloader))
    start_epoch = iter // len(dataloader)
    step = 0

    writer.add_scalar('Learning rate', opt.lr)

    writer.add_scalar('class_weight', opt.class_weight)
    writer.add_scalar('img_weight', opt.img_weight)
    writer.add_scalar('latent_weight', opt.latent_weight)
    writer.add_scalar('reg_weight', opt.reg_weight)

    with torch.autograd.set_detect_anomaly(True):
        print('Beginning training...')
        for epoch in range(start_epoch, opt.max_epoch):
            for model_input, ground_truth in dataloader:
                model_outputs = model(model_input)

                optimizer.zero_grad()

                # losses
                wghtd_class_loss = opt.class_weight * model.get_seg_loss(model_outputs, ground_truth)
                wghtd_img_loss = opt.img_weight * model.get_image_loss(model_outputs, ground_truth)
                wghtd_latent_loss = opt.latent_weight * model.get_latent_loss()
                wghtd_reg_loss = opt.reg_weight * model.get_regularization_loss(model_outputs, ground_truth)

                gen_loss_total = (wghtd_class_loss +
                                  wghtd_reg_loss +
                                  wghtd_img_loss +
                                  wghtd_latent_loss)
                gen_loss_total.backward()
                optimizer.step()

                print("iter %07d  epoch %03d  class_loss %0.4f img_loss %0.4f latent_loss %0.4f reg_loss %0.4f" %
                (iter, epoch, wghtd_class_loss, wghtd_img_loss, wghtd_latent_loss, wghtd_reg_loss))

                with torch.no_grad():
                    model.write_updates(writer, model_input, model_outputs, ground_truth, iter)

                    writer.add_scalar("class_loss", wghtd_class_loss, iter)
                    writer.add_scalar("latent_loss", wghtd_latent_loss, iter)
                    writer.add_scalar("reg_loss", wghtd_reg_loss, iter)
                    writer.add_scalar("img_loss", wghtd_img_loss, iter)

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

                            dist_loss = model.get_image_loss(model_outputs, ground_truth).cpu().numpy()
                            psnr, ssim = model.get_psnr(model_outputs, ground_truth)
                            psnrs.append(psnr)
                            ssims.append(ssim)
                            dist_losses.append(dist_loss)

                            model.write_updates(writer, model_input, model_outputs, ground_truth, iter, prefix='val_')

                        writer.add_scalar("val_dist_loss", np.mean(dist_losses), iter)
                        writer.add_scalar("val_psnr", np.mean(psnrs), iter)
                        writer.add_scalar("val_ssim", np.mean(ssims), iter)
                    model.train()

                iter += 1
                step += 1

                if max_steps is not None:
                    if iter == max_steps:
                        break

                if iter % opt.steps_til_ckpt == 0:
                    util.custom_save(model,
                                     os.path.join(log_dir, 'epoch_%04d_iter_%06d.pth'%(epoch, iter)),
                                     discriminator=None,
                                     optimizer=optimizer)

            if max_steps is not None:
                if iter == max_steps:
                    break

    final_ckpt_path = os.path.join(log_dir, 'epoch_%04d_iter_%06d.pth'%(epoch, iter))
    util.custom_save(model,
                     final_ckpt_path,
                     discriminator=None,
                     optimizer=optimizer)

    return final_ckpt_path


def main():
    if opt.specific_observation_idcs is not None:
        specific_observation_idcs = list(map(int, opt.specific_observation_idcs.split(',')))
    else:
        specific_observation_idcs = None

    dataset = SceneClassDataset(root_dir=opt.data_root,
                                max_num_instances=opt.max_num_instances_train,
                                max_observations_per_instance=opt.max_num_observations_train,
                                img_sidelength=opt.img_sidelength,
                                specific_observation_idcs=specific_observation_idcs,
                                samples_per_instance=1)

    if not opt.no_validation:
        if opt.val_root is None:
            raise ValueError("No validation directory passed.")
        val_dataset = SceneClassDataset(root_dir=opt.val_root,
                                        max_num_instances=opt.max_num_instances_val,
                                        max_observations_per_instance=opt.max_num_observations_val,
                                        img_sidelength=opt.img_sidelength,
                                        samples_per_instance=1)
    else:
        val_dataset = None

    model = SRNsModel(num_instances=dataset.num_instances,
                      latent_dim= opt.embedding_size,
                      tracing_steps=opt.tracing_steps)
    train(model,
          dataset,
          val_dataset,
          batch_size=opt.batch_size,
          checkpoint=opt.checkpoint,
          max_steps=opt.max_steps)


if __name__ == '__main__':
    main()
