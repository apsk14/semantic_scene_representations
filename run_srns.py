import argparse
import os, time, datetime

import torch
import numpy as np
from tensorboardX import SummaryWriter

from dataio import *
from torch.utils.data import DataLoader
from srns import *
import util

torch.backends.cudnn.benchmark = True

# params
parser = argparse.ArgumentParser()

parser.add_argument('--train_test', type=str, required=True, help='path to file list of h5 train data')
parser.add_argument('--data_root', required=True, help='path to file list of h5 train data')
parser.add_argument('--val_root', required=False, help='path to file list of h5 train data')
parser.add_argument('--logging_root', type=str, default='/media/staging/deep_sfm/',
                    required=False, help='path to file list of h5 train data')

parser.add_argument('--max_epoch', type=int, default=1501, help='number of epochs to train for')
parser.add_argument('--max_steps', type=int, default=None, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate, default=0.001')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Training batch size.')

parser.add_argument('--l1_weight', type=float, default=200,
                    help='learning rate, default=0.001')
parser.add_argument('--kl_weight', type=float, default=1,
                    help='learning rate, default=0.001')
parser.add_argument('--reg_weight', type=float, default=1e-3,
                    help='Weight for regularization term in paper.')

parser.add_argument('--tracing_steps', type=int, default=10, help='Number of steps of intersection tester')

parser.add_argument('--steps_til_ckpt', type=int, default=10000,
                    help='Number of iterations until checkpoint is saved.')
parser.add_argument('--steps_til_val', type=int, default=1000,
                    help='Number of iterations until validation set is run.')
parser.add_argument('--no_validation', action='store_true', default=False,
                    help='If no validation set should be used.')

parser.add_argument('--img_sidelength', type=int, default=128, required=False,
                    help='Sidelength of training images. If original images are bigger, they\'re downsampled.')
parser.add_argument('--mode', type=str, required=True, help='#images')

parser.add_argument('--no_preloading', action='store_true', default=False,
                    help='Whether to preload data to RAM.')

parser.add_argument('--max_num_instances_train', type=int, default=-1,
                    help='If \'train_root\' has more instances, only the first max_num_instances_train are used')
parser.add_argument('--max_num_observations_train', type=int, default=50, required=False,
                    help='If an instance has more observations, only the first max_num_observations_train are used')
parser.add_argument('--max_num_instances_val', type=int, default=10, required=False,
                    help='If \'val_root\' has more instances, only the first max_num_instances_val are used')
parser.add_argument('--max_num_observations_val', type=int, default=10, required=False,
                    help='Maximum numbers of observations per validation instance')

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
parser.add_argument('--implicit_nf', type=int, default=256,
                    help='Number of hidden units in SRN')
parser.add_argument('--embedding_size', type=int, required=True,
                    help='Dimensionality of latent embedding.')


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

    dir_name = os.path.join(datetime.datetime.now().strftime('%m_%d'),
                            datetime.datetime.now().strftime('%H-%M-%S_'))

    log_dir = os.path.join(opt.logging_root, 'logs', dir_name)
    run_dir = os.path.join(opt.logging_root, 'runs', dir_name)
    util.cond_mkdir(log_dir)
    util.cond_mkdir(run_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    if opt.checkpoint is not None:
        print("Loading model from %s"%opt.checkpoint)
        util.custom_load(model, path=opt.checkpoint,
                         discriminator=None,
                         optimizer=optimizer,
                         overwrite_embeddings=opt.overwrite_embeddings)

    writer = SummaryWriter(run_dir)
    iter = opt.start_step
    start_epoch = iter // len(dataloader)
    step = 0

    writer.add_scalar('Learning rate', opt.lr)
    writer.add_scalar('l1 weight', opt.l1_weight)

    with torch.autograd.set_detect_anomaly(True):
        print('Beginning training...')
        for epoch in range(start_epoch, opt.max_epoch):
            for model_input, ground_truth in dataloader:
                model_outputs = model(model_input)

                optimizer.zero_grad()

                dist_loss = model.get_image_loss(model_outputs, ground_truth)
                reg_loss = model.get_regularization_loss(model_outputs, ground_truth)
                var_loss = model.get_latent_loss()

                weighted_kl_loss = var_loss * opt.kl_weight

                gen_loss_total = opt.l1_weight * dist_loss + \
                                 opt.reg_weight * reg_loss + \
                                 weighted_kl_loss

                gen_loss_total.backward(retain_graph=False)

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_scalar("_" + name + "_grad_norm", param.grad.data.norm(2).item(), iter)

                optimizer.step()

                print("Iter %07d   Epoch %03d   dist_loss %0.4f   KL %0.4f   reg_loss %0.4f" %
                      (iter, epoch, dist_loss*opt.l1_weight, weighted_kl_loss, reg_loss*opt.reg_weight))

                model.write_updates(writer, model_outputs, ground_truth, iter)
                writer.add_scalar("scaled_distortion_loss", dist_loss*opt.l1_weight, iter)
                writer.add_scalar("combined_generator_loss", torch.clamp(gen_loss_total, 0, 1e3), iter)
                writer.add_scalar("scaled_reg_loss", reg_loss * opt.reg_weight, iter)
                writer.add_scalar("reg_loss", reg_loss)
                writer.add_scalar("scaled_kl_loss", weighted_kl_loss, iter)
                writer.add_scalar("kl_loss", var_loss, iter)
                writer.add_scalar("kl_weight", opt.kl_weight, iter)

                if not iter:
                    # Save parameters used into the log directory.
                    with open(os.path.join(log_dir, "params.txt"), "w") as out_file:
                        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))
                    with open(os.path.join(run_dir, "params.txt"), "w") as out_file:
                        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

                    # Save a text summary of the model into the log directory.
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

                            dist_loss = model.get_image_loss(model_outputs, ground_truth).cpu().numpy()
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
                                     discriminator=None,
                                     optimizer=optimizer)

            if iter == opt.max_steps:
                break

    final_ckpt_path = os.path.join(log_dir, 'epoch_%04d_iter_%06d.pth'%(epoch, iter))
    util.custom_save(model,
                     final_ckpt_path,
                     discriminator=None,
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
    util.cond_mkdir(comparison_dir)
    util.cond_mkdir(traj_dir)

    # Save parameters used into the log directory.
    with open(os.path.join(traj_dir, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    print('Beginning evaluation...')
    with torch.no_grad():
        psnrs, ssims = list(), list()
        for idx, (model_input, ground_truth) in enumerate(dataset):
            if not idx%10:
                print(idx)

            model_outputs = model(model_input)
            model.write_eval(model_outputs, ground_truth, os.path.join(traj_dir, "%06d.png"%idx))
            model.write_comparison(model_outputs, ground_truth, os.path.join(comparison_dir, "%06d.png"%idx))

            psnr, ssim = model.get_psnr(model_outputs, ground_truth)
            psnrs.append(psnr)
            ssims.append(ssim)

    print(np.mean(psnrs))
    print(np.mean(ssims))


def main():
    if opt.train_test == 'train':
        dataset = SceneClassDataset(root_dir=opt.data_root,
                                    preload=not opt.no_preloading,
                                    max_num_instances=opt.num_train_instances,
                                    max_observations_per_instance=opt.num_images,
                                    img_sidelength=opt.img_sidelength,
                                    samples_per_instance=1)

        if not opt.no_validation:
            val_dataset = SceneClassDataset(root_dir=opt.val_root,
                                            preload=not opt.no_preloading,
                                            max_num_instances=opt.num_val_instances,
                                            max_observations_per_instance=opt.num_val_images,
                                            img_sidelength=opt.img_sidelength,
                                            samples_per_instance=1)
        else:
            val_dataset = None

        model = SRNsModel(num_objects=dataset.num_instances,
                          embedding_size=opt.embedding_size,
                          implicit_nf=opt.implicit_nf,
                          has_params=opt.has_params,
                          mode=opt.mode,
                          use_unet_renderer=opt.use_unet_renderer,
                          tracing_steps=opt.tracing_steps)
        train(model, dataset, val_dataset)
    elif opt.train_test == 'test':
        dataset = SceneClassDataset(root_dir=opt.data_root,
                                    preload=not opt.no_preloading,
                                    max_num_instances=opt.num_objects,
                                    max_observations_per_instance=-1,
                                    samples_per_instance=1,
                                    img_sidelength=opt.img_sidelength)
        model = SRNsModel(num_objects=dataset.num_instances,
                          embedding_size=opt.embedding_size,
                          implicit_nf=opt.implicit_nf,
                          has_params=opt.has_params,
                          mode=opt.mode,
                          use_unet_renderer=opt.use_unet_renderer,
                          tracing_steps=opt.tracing_steps)
        test(model, dataset)
    else:
        print("Unknown mode.")
        return None


if __name__ == '__main__':
    main()
