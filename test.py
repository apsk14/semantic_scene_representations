import configargparse
import os, time, datetime

import torch
import numpy as np

import class_dataio as dataio
from torch.utils.data import DataLoader
from srns_vincent import *
import util

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

# Note: in contrast to training, no multi-resolution!
p.add_argument('--img_sidelength', type=int, default=128, required=False,
               help='Sidelength of test images.')

p.add_argument('--data_root', required=True, help='Path to directory with training data.')

p.add_argument('--stat_root', required=True, help='Path to directory with statistics data.')

p.add_argument('--obj_name', required=True,type=str, help='Name of object in question')

p.add_argument('--logging_root', type=str, default='./logs',
               required=False, help='Path to directory where checkpoints & tensorboard events will be saved.')
p.add_argument('--batch_size', type=int, default=32, help='Batch size.')
p.add_argument('--preload', action='store_true', default=False, help='Whether to preload data to RAM.')

p.add_argument('--max_num_instances', type=int, default=-1,
               help='If \'data_root\' has more instances, only the first max_num_instances are used')
p.add_argument('--specific_observation_idcs', type=str, default=None,
               help='Only pick a subset of specific observations for each instance.')
p.add_argument('--has_params', action='store_true', default=False,
               help='Whether each object instance already comes with its own parameter vector.')
p.add_argument('--num_classes', type=int, default=6,
               help='number of seg classes for the given object')

p.add_argument('--save_out_first_n', type=int, default=250, help='Only saves images of first n object instances.')
p.add_argument('--checkpoint_path', default=None, help='Path to trained model.')

# Model options
p.add_argument('--num_instances', type=int, required=False,
               help='The number of object instances that the model was trained with.')
p.add_argument('--tracing_steps', type=int, default=10, help='Number of steps of intersection tester.')
p.add_argument('--fit_single_srn', action='store_true', required=False,
               help='Only fit a single SRN for a single scene (not a class of SRNs) --> no hypernetwork')
p.add_argument('--use_unet_renderer', action='store_true',
               help='Whether to use a DeepVoxels-style unet as rendering network or a per-pixel 1x1 convnet')
p.add_argument('--embedding_size', type=int, default=256,
               help='Dimensionality of latent embedding.')

opt = p.parse_args()

device = torch.device('cuda')


def test():
    if opt.specific_observation_idcs is not None:
        specific_observation_idcs = list(map(int, opt.specific_observation_idcs.split(',')))
    else:
        specific_observation_idcs = None

    test_set = dataio.SceneClassDataset(root_dir=opt.data_root,
                                       stat_dir=opt.stat_root,
                                       obj_name=opt.obj_name,
                                       max_num_instances=opt.max_num_instances,
                                       specific_observation_idcs=specific_observation_idcs,
                                       max_observations_per_instance=-1,
                                       samples_per_instance=1,
                                       img_sidelength=opt.img_sidelength)
    dataset = DataLoader(test_set,
                         collate_fn=test_set.collate_fn,
                         batch_size=1,
                         shuffle=False,
                         drop_last=False)

    model = SRNsModel(num_instances=test_set.num_instances,
                      latent_dim=opt.embedding_size,
                      tracing_steps=opt.tracing_steps)

    assert (opt.checkpoint_path is not None), "Have to pass checkpoint!"

    print("Loading model from %s" % opt.checkpoint_path)
    util.custom_load(model, path=opt.checkpoint_path, discriminator=None,
                     overwrite_embeddings=False)

    model.eval()
    model.cuda()

    # directory structure: month_day/
    #renderings_dir = os.path.join(opt.logging_root, 'renderings')
    util.cond_mkdir(opt.logging_root)
    #util.cond_mkdir(renderings_dir)

    # Save command-line parameters to log directory.
    with open(os.path.join(opt.logging_root, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    print('Beginning evaluation...')
    with torch.no_grad():
        IOU = 0
        part_intersect = np.zeros(NUM_CLASSES, dtype=np.float32)
        part_union = np.zeros(NUM_CLASSES, dtype=np.float32)
        confusion = np.zeros((NUM_CLASSES, 3), dtype=int) # TODO: find a way to generalize this
        output_flag = 1
        instance_idx = 0
        idx = 0
        ins_idx = 0
        # psnrs, ssims = list(), list()
        for model_input, ground_truth in dataset:
            model_outputs = model(model_input)
            # psnr, ssim = model.get_psnr(model_outputs, ground_truth)
            #
            # psnrs.extend(psnr)
            # ssims.extend(ssim)

            instance_idcs = model_input['instance_idx']
            # print("Object instance %d. Running mean PSNR %0.6f SSIM %0.6f" %
            #       (instance_idcs[-1], np.mean(psnrs), np.mean(ssims)))

            if instance_idx > opt.save_out_first_n:
                output_flag = 0

            output_imgs = model.get_output_img(model_outputs).cpu().numpy()
            output_segs = model.get_output_seg(model_outputs)

            trgt_imgs = model_input['rgb'].cuda()
            trgt_segs = model_input['seg'].cuda()

            trgt_imgs = util.lin2img(trgt_imgs)
            trgt_segs_display = model.get_output_seg(trgt_segs, trgt=True)
            comparisons = model.get_comparisons(model_input,
                                                model_outputs,
                                                ground_truth)
            for i in range(len(output_imgs)):
                prev_instance_idx = instance_idx
                instance_idx = instance_idcs[i]

                if prev_instance_idx != instance_idx:
                    idx = 0
                    ins_idx += 1
                    print(ins_idx)
                    print(instance_idx)

                newIOU = model.get_IOU_vals(model_outputs, trgt_segs, confusion, part_intersect, part_union)
                print(idx)
                print('Image IOU: ', newIOU)
                IOU += newIOU

                if output_flag:
                    rgb = os.path.join(opt.logging_root,"%06d" % instance_idx, 'rgb')
                    rgb_gt = os.path.join(opt.logging_root,"%06d" % instance_idx, 'rgb_gt')
                    seg = os.path.join(opt.logging_root,"%06d" % instance_idx, 'seg')
                    seg_gt = os.path.join(opt.logging_root,"%06d" % instance_idx, 'seg_gt')
                    input_img = os.path.join(opt.logging_root,"%06d" % instance_idx, 'input')
                    comp = os.path.join(opt.logging_root,"%06d" % instance_idx, 'comparison')

                    util.cond_mkdir(rgb)
                    util.cond_mkdir(rgb_gt)
                    util.cond_mkdir(seg)
                    util.cond_mkdir(seg_gt)
                    util.cond_mkdir(input_img)
                    util.cond_mkdir(comp)

                    rgb_out = util.convert_image(output_imgs[i].squeeze())
                    seg_out = util.convert_image(output_segs[i].squeeze())
                    rgb_gt_out = util.convert_image(trgt_imgs[i].squeeze())
                    seg_gt_out = util.convert_image(trgt_segs_display[i].squeeze())
                    comp_out = util.convert_image(comparisons[i].squeeze())

                    if idx == 102:
                        input_out = util.convert_image(trgt_imgs[i].squeeze())
                        util.write_img(input_out, os.path.join(input_img, "%06d.png" % idx))
                    util.write_img(rgb_out, os.path.join(rgb, "%06d.png" % idx))
                    util.write_img(rgb_gt_out, os.path.join(rgb_gt, "%06d.png" % idx))
                    util.write_img(seg_out, os.path.join(seg, "%06d.png" % idx))
                    util.write_img(seg_gt_out, os.path.join(seg_gt, "%06d.png" % idx))
                    util.write_img(comp_out, os.path.join(comp, "%06d.png" % idx))

                idx += 1


    print((ins_idx+1) * 251)
    mIOU = IOU / ((ins_idx+1) * 251)
    part_intersect = np.delete(part_intersect, np.where(part_union == 0))
    part_union = np.delete(part_union, np.where(part_union == 0))
    part_iou = np.divide(part_intersect[0:], part_union[0:])
    mean_part_iou = np.mean(part_iou)

    #with open(os.path.join(opt.logging_root, "results.txt"), "w") as out_file:
    #    out_file.write("%0.6f, %0.6f" % (np.mean(psnrs), np.mean(ssims)))

    #print("Final mean PSNR %0.6f SSIM %0.6f" % (np.mean(psnrs), np.mean(ssims)))
    print('mIOU: ', mIOU)
    print('Category mean IoU: %f, %s' % (mean_part_iou, str(part_iou)))


def main():
    test()


if __name__ == '__main__':
    main()

