import argparse
import os, time, datetime

import torch
import numpy as np
from tensorboardX import SummaryWriter

from dataio import *
from torch.utils.data import DataLoader
from srn_unet import *
import util

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
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate, default=0.001')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Training batch size.')

parser.add_argument('--l1_weight', type=float, default=200,
                    help='learning rate, default=0.001')
parser.add_argument('--kl_weight', type=float, default=1,
                    help='learning rate, default=0.001')
parser.add_argument('--reg_weight', type=float, default=1e-3,
                    help='Weight for regularization term in paper.')

parser.add_argument('--tracing_steps', type=int, default=10, help='Number of steps of intersection tester')

parser.add_argument('--steps_til_ckpt', type=int, default=5000,
                    help='Number of iterations until checkpoint is saved.')
parser.add_argument('--steps_til_val', type=int, default=1000,
                    help='Number of iterations until validation set is run.')
parser.add_argument('--no_validation', action='store_true', default=False,
                    help='If no validation set should be used.')

parser.add_argument('--img_sidelength', type=int, default=128, required=False,
                    help='Sidelength of training images. If original images are bigger, they\'re downsampled.')
parser.add_argument('--fit_single_srn', action='store_true', required=False,
                    help='Only fit a single SRN for a single scene (not a class of SRNs) --> no hypernetwork')

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
parser.add_argument('--srn_path', type=str, default=None,
                    help='SRN Model for testing single shot Unet')
parser.add_argument('--unet_path', type=str, default=None,
                    help='Unet Model for testing Unet')

parser.add_argument('--use_unet_renderer', action='store_true',
                    help='Whether to use a DeepVoxels-style unet as rendering network or a per-pixel 1x1 convnet')
parser.add_argument('--embedding_size', type=int, default=256,
                    help='Dimensionality of latent embedding.')


opt = parser.parse_args()
print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

device = torch.device('cuda')
NUM_CLASSES = 6


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

    if checkpoint is not None:
        print("Loading model from %s"%opt.checkpoint)
        util.custom_load(model, path=opt.checkpoint,
                         discriminator=None,
                         optimizer=None,
                         overwrite_embeddings=opt.overwrite_embeddings)

    writer = SummaryWriter(run_dir)
    iter = opt.start_step
    print(len(dataloader))
    start_epoch = iter // len(dataloader)
    step = 0

    writer.add_scalar('Learning rate', opt.lr)
    writer.add_scalar('l1 weight', opt.l1_weight)

    with torch.autograd.set_detect_anomaly(True):
        print('Beginning training...')
        for epoch in range(start_epoch, opt.max_epoch):
            for model_input, ground_truth in dataloader:
                observation = Observation(*model_input)
                rgb_img = observation.rgb
                model_outputs = model(rgb_img.cuda())

                optimizer.zero_grad()

                #losses
                class_loss = model.get_seg_loss(model_outputs, ground_truth)
                img_IOU_loss = model.get_img_IOU_loss(model_outputs, ground_truth)

                gen_loss_total = opt.l1_weight * (class_loss)


                gen_loss_total.backward(retain_graph=False)

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_scalar("_" + name + "_grad_norm", param.grad.data.norm(2).item(), iter)

                optimizer.step()


                print("Iter %07d   Epoch %03d  class_loss %0.4f  img_IOU_loss %0.4f " %
                (iter, epoch,class_loss*opt.l1_weight, img_IOU_loss*opt.l1_weight/2))

                with torch.no_grad():
                    model.write_updates(writer, model_input, model_outputs, ground_truth, iter)
                writer.add_scalar("scaled_class_loss", class_loss * opt.l1_weight, iter)
                writer.add_scalar("scaled_IOU_loss", img_IOU_loss * opt.l1_weight/2, iter)
                writer.add_scalar("combined_generator_loss", torch.clamp(gen_loss_total, 0, 1e3), iter)

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

                        seg_losses = []
                        iou_losses = []
                        for model_input, ground_truth in val_dataloader:
                            observation = Observation(*model_input)
                            rgb_img = observation.rgb
                            model_outputs = model(rgb_img.cuda())
                            seg_loss = model.get_seg_loss(model_outputs, ground_truth).cpu().numpy()
                            iou_loss = model.get_img_IOU_loss(model_outputs, ground_truth).cpu().numpy()

                            seg_losses.append(seg_loss)
                            iou_losses.append(iou_loss)

                            model.write_updates(writer, model_input, model_outputs, ground_truth, iter, prefix='val_')

                        writer.add_scalar("val_seg_loss", np.mean(seg_losses), iter)
                        writer.add_scalar("val_iou_loss", np.mean(iou_losses), iter)


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


def test(dataset, srn_path, unet_path):
    # colors = [(random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)) for i in
    #                range(NUM_CLASSES)]
    # colors[0] = (1, 1, 1)
    # colors = np.array(colors)
    colors = np.array([[1., 1., 1.], [0.42415977, 0.95022593, 0.88655337], [0.34309762, 0.95100353, 0.3231704],
                                [0.48631192, 0.82279855, 0.80800228], [0.27445405, 0.42794667, 0.42610895],
                                [0.53534125, 0.04302588, 0.9653457]])
    collate_fn = dataset.collate_fn
    num_instances = dataset.num_instances
    dataset = DataLoader(dataset,
                         collate_fn=collate_fn,
                         batch_size=1,
                         shuffle=False,
                         drop_last=False)


    # directory structure: month_day/
    dir_name = os.path.join(datetime.datetime.now().strftime('%m_%d'),
                            datetime.datetime.now().strftime('%H-%M-%S_') +
                            '_'.join(opt.unet_path.strip('/').split('/')[-2:])[:200] + '_'
                            + opt.data_root.strip('/').split('/')[-1])

    #traj_dir = os.path.join(opt.logging_root, 'test_traj', dir_name)
    #comparison_dir = os.path.join(opt.logging_root, 'test_traj', dir_name, 'comparison')
    traj_dir = os.path.join('/home/apsk14/data/', 'test_traj_chair_unet_single', dir_name)
    #comparison_dir = os.path.join(traj_dir, 'comparison')
    # util.cond_mkdir(comparison_dir)
    util.cond_mkdir(traj_dir)

    # Save parameters used into the log directory.
    with open(os.path.join(traj_dir, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    print('Beginning evaluation...')
    save_out_first_n = 250
    with torch.no_grad():
        obj_idx = 0
        idx = 0
        IOU = 0
        part_intersect = np.zeros(NUM_CLASSES, dtype=np.float32)
        part_union = np.zeros(NUM_CLASSES, dtype=np.float32)
        confusion = np.zeros((NUM_CLASSES, 3), dtype=int) # TODO: find a way to generalize this
        ins_idx = 0
        output_flag = 1
        model_unet = UnetModel(num_instances=num_instances,
                          latent_dim=opt.embedding_size,
                          has_params=opt.has_params,
                          fit_single_srn=opt.fit_single_srn,
                          use_unet_renderer=opt.use_unet_renderer,
                          tracing_steps=opt.tracing_steps)
        util.custom_load(model_unet, path=unet_path)
        model_unet.eval()
        model_unet.cuda()
        if srn_path is not None:
            model_srn = SRNsModel(num_instances=num_instances,
                              latent_dim=opt.embedding_size,
                              has_params=opt.has_params,
                              fit_single_srn=opt.fit_single_srn,
                              use_unet_renderer=opt.use_unet_renderer,
                              tracing_steps=opt.tracing_steps)
            util.custom_load(model_srn, path=srn_path)
            model_srn.eval()
            model_srn.cuda()
        for model_input, ground_truth in dataset:
            observation = Observation(*model_input)
            obj_idcs = observation.instance_idx.long()

            if obj_idx >= save_out_first_n:
                output_flag = 0

            if srn_path is not None:
                model_output = model_srn(model_input)
                rgb_image = model_srn.get_output_img(model_output)
                #rgb_image = eval_srn(model_input, srn_path, num_instances)
            else:
                rgb_image = observation.rgb.cuda()

            trgt_imgs, trgt_segs_tensor, trgt_depths = ground_truth
            prediction = model_unet(rgb_image)
            output_segs = model_unet.get_output_seg(prediction, colors)
            trgt_segs = model_unet.get_output_seg(trgt_segs_tensor, colors)

            #prediction, output_segs, trgt_segs = eval_unet(rgb_image, trgt_segs_tensor, unet_path, num_instances, colors)
            #comparisons = model.get_comparisons(model_input,
                                                 # prediction,
                                                 # ground_truth)
            for i in range(len(rgb_image)):
                prev_obj_idx = obj_idx
                obj_idx = obj_idcs[i]

                if prev_obj_idx != obj_idx:
                    idx = 0

                if idx == 0:
                    print('INSTANCE', ins_idx)
                    print('calculating IOU')
                    print(confusion)
                    ins_idx += 1
                print('OBS', idx)
                newIOU = get_IOU_vals(prediction, trgt_segs_tensor, confusion, part_intersect, part_union)
                print(newIOU)
                IOU += newIOU

                if output_flag:
                    img_only_path = os.path.join(traj_dir, 'images', "%06d" % obj_idx)
                    gt_only_path = os.path.join(traj_dir, 'gt', "%06d" % obj_idx)
                    seg_only_path = os.path.join(traj_dir, 'segs', "%06d" % obj_idx)
                    #comp_path = os.path.join(comparison_dir, "%06d" % obj_idx)

                    util.cond_mkdir(img_only_path)
                    util.cond_mkdir(gt_only_path)
                    util.cond_mkdir(seg_only_path)
                    #util.cond_mkdir(comp_path)


                    pred = util.convert_image(rgb_image[i].squeeze())
                    gt = util.convert_image(trgt_segs[i].squeeze())
                    pred_seg = util.convert_image(output_segs[i].squeeze())
                    #comp = util.convert_image(comparisons[i].squeeze())


                    util.write_img(pred, os.path.join(img_only_path, "%06d.png" % idx))
                    util.write_img(gt, os.path.join(gt_only_path, "%06d.png" % idx))
                    util.write_img(pred_seg, os.path.join(seg_only_path, "%06d.png" % idx))
                    #util.write_img(comp, os.path.join(comp_path, "%06d.png" % idx))
                idx += 1

    mIOU = IOU/(ins_idx * 251)

    print(colors)
    print('mIOU: ', mIOU)
    part_intersect = np.delete(part_intersect, np.where(part_union == 0))
    part_union = np.delete(part_union, np.where(part_union == 0))
    part_iou = np.divide(part_intersect[0:], part_union[0:])
    mean_part_iou = np.mean(part_iou)
    print('Category mean IoU: %f, %s' % (mean_part_iou, str(part_iou)))


def eval_srn(model_input, path, num_instances):
    model = SRNsModel(num_instances=num_instances,
                      latent_dim=opt.embedding_size,
                      has_params=opt.has_params,
                      fit_single_srn=opt.fit_single_srn,
                      use_unet_renderer=opt.use_unet_renderer,
                      tracing_steps=opt.tracing_steps)
    util.custom_load(model, path=path)
    model.eval()
    model.cuda()
    model_output = model(model_input)
    rgb_img = model.get_output_img(model_output)
    return rgb_img


def eval_unet(model_input, trgt_segs, path, num_instances,colors):
    model = UnetModel(num_instances=num_instances,
                      latent_dim=opt.embedding_size,
                      has_params=opt.has_params,
                      fit_single_srn=opt.fit_single_srn,
                      use_unet_renderer=opt.use_unet_renderer,
                      tracing_steps=opt.tracing_steps)
    util.custom_load(model, path=path)
    model.eval()
    model.cuda()
    prediction = model(model_input)
    seg_preds = model.get_output_seg(prediction,colors)
    trgt_segs = model.get_output_seg(trgt_segs, colors)
    return prediction, seg_preds, trgt_segs

def get_IOU_vals(prediction, trgt_seg, confusion, part_intersect, part_union): # had arg confusion
    # confusion vector is [true pos, false pos, false neg]

    trgt_seg = torch.reshape(trgt_seg, (1,1,trgt_seg.shape[2]*trgt_seg.shape[2]))
    prediction = torch.reshape(prediction, (1, prediction.shape[1], prediction.shape[2] * prediction.shape[2]))
    trgt_seg = trgt_seg.permute(0,2,1)
    prediction = prediction.permute(0,2,1)
    pred_segs, seg_idx = torch.max(prediction, dim=2)
    pred_labels = seg_idx.cpu().numpy().squeeze()
    real_label = trgt_seg.cpu().numpy().squeeze()

    # pred_labels = np.delete(seg_idx, np.where(trgt_seg == 0), axis=0)
    # real_label = np.delete(trgt_seg, np.where(trgt_seg == 0), axis=0)

    num_classes = NUM_CLASSES
    cur_shape_iou_tot = 0.0
    cur_shape_iou_cnt = 0
    for cur_class in range(0, num_classes):

        cur_gt_mask = (real_label == cur_class)
        cur_pred_mask = (pred_labels == cur_class)

        has_gt = (np.sum(cur_gt_mask) > 0)
        has_pred = (np.sum(cur_pred_mask) > 0)

        if has_gt or has_pred:
            intersect = np.sum(cur_gt_mask & cur_pred_mask)
            union = np.sum(cur_gt_mask | cur_pred_mask)
            iou = intersect / union

            cur_shape_iou_tot += iou
            cur_shape_iou_cnt += 1

            part_intersect[cur_class - 1] += intersect
            part_union[cur_class - 1] += union

        # expected_true = pred_labels[np.where(real_label == cur_class)]
        # expected_false = pred_labels[np.where(real_label != cur_class)]
        # true_pos[cur_class-1] = expected_true[np.where(expected_true == cur_class)].shape[0]
        # false_neg[cur_class-1] = expected_true[np.where(expected_true != cur_class)].shape[0]
        # false_pos[cur_class-1] = expected_false[np.where(expected_false == cur_class)].shape[0]
    # IOU = self.calc_mIOU(np.concatenate((true_pos, false_pos, false_neg), axis=1))
    # confusion += np.concatenate((true_pos, false_pos, false_neg), axis=1)
    if cur_shape_iou_cnt == 0:
        return 1
    return cur_shape_iou_tot / cur_shape_iou_cnt



def main():
    if opt.specific_observation_idcs is not None:
        specific_observation_idcs = list(map(int, opt.specific_observation_idcs.split(',')))
    else:
        specific_observation_idcs = None

    if opt.train_test == 'train':
        dataset = SceneClassDataset(root_dir=opt.data_root,
                                    preload=not opt.no_preloading,
                                    max_num_instances=opt.max_num_instances_train,
                                    max_observations_per_instance=opt.max_num_observations_train,
                                    img_sidelength=opt.img_sidelength,
                                    specific_observation_idcs=specific_observation_idcs,
                                    samples_per_instance=1)

        if not opt.no_validation:
            if opt.val_root is None:
                raise ValueError("No validation directory passed.")
            val_dataset = SceneClassDataset(root_dir=opt.val_root,
                                            preload=not opt.no_preloading,
                                            max_num_instances=opt.max_num_instances_val,
                                            max_observations_per_instance=opt.max_num_observations_val,
                                            img_sidelength=opt.img_sidelength,
                                            samples_per_instance=1)
        else:
            val_dataset = None

        model = UnetModel(num_instances=dataset.num_instances,
                          latent_dim=opt.embedding_size,
                          has_params=opt.has_params,
                          fit_single_srn=opt.fit_single_srn,
                          use_unet_renderer=opt.use_unet_renderer,
                          tracing_steps=opt.tracing_steps)
        train(model,
              dataset,
              val_dataset,
              batch_size=opt.batch_size,
              checkpoint=opt.checkpoint,
              max_steps=opt.max_steps)
    elif opt.train_test == 'test':
        dataset = SceneClassDataset(root_dir=opt.data_root,
                                    preload=not opt.no_preloading,
                                    max_num_instances=opt.max_num_instances_train,
                                    specific_observation_idcs=specific_observation_idcs,
                                    max_observations_per_instance=-1,
                                    samples_per_instance=1,
                                    img_sidelength=opt.img_sidelength)
        # model = SRNsModel(num_instances=dataset.num_instances,
        #                   latent_dim=opt.embedding_size,
        #                   has_params=opt.has_params,
        #                   fit_single_srn=opt.fit_single_srn,
        #                   use_unet_renderer=opt.use_unet_renderer,
        #                   tracing_steps=opt.tracing_steps)
        test(dataset, opt.srn_path, opt.unet_path)
    else:
        print("Unknown mode.")
        return None


if __name__ == '__main__':
    main()
