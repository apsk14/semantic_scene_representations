import configargparse
import os, time, datetime

import torch
import numpy as np
import csv

import dataio
from torch.utils.data import DataLoader
from models import *
import util

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

# Note: in contrast to training, no multi-resolution!
p.add_argument('--img_sidelength', type=int, default=128, required=False,
               help='Sidelength of test images.')

p.add_argument('--point_cloud', action='store_true', default=False,
               help='Whether to render out point clouds.')

p.add_argument('--data_root', required=True, help='Path to directory with training data.')

p.add_argument('--obj_name', required=True,type=str, help='Name of object in question')

p.add_argument('--eval_mode', required=True,type=str, help='Which model to evaluate (linear, unet, srn)')

p.add_argument('--reverse', required=False, default=False, action='store_true', help='Flag to indicate that the input was a segmentation map')

p.add_argument('--logging_root', type=str, default='./logs',
               required=False, help='Path to directory where checkpoints & tensorboard events will be saved.')
p.add_argument('--log_dir', required=False, default = '', help='Name of dir within logging root to store checkpoints and events')               
p.add_argument('--batch_size', type=int, default=32, help='Batch size.')
p.add_argument('--preload', action='store_true', default=False, help='Whether to preload data to RAM.')

p.add_argument('--max_num_instances', type=int, default=-1,
               help='If \'data_root\' has more instances, only the first max_num_instances are used')
p.add_argument('--specific_observation_idcs', type=str, default=None,
               help='Only pick a subset of specific observations for each instance.')
p.add_argument('--input_idcs', type=str, default=None,
               help='Specifies which input views were given at test time. For visualization purposes')
p.add_argument('--has_params', action='store_true', default=False,
               help='Whether each object instance already comes with its own parameter vector.')
p.add_argument('--linear', action='store_true', default=False,
               help='Whether each object instance already comes with its own parameter vector.')

p.add_argument('--save_out_first_n', type=int, default=250, help='Only saves images of first n object instances.')
p.add_argument('--checkpoint_path', default=None, help='Path to trained model.')
p.add_argument('--linear_path', default=None, help='Path to trained model.')
p.add_argument('--unet_path', default=None, help='Path to trained model.')

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

    if opt.input_idcs is not None:
        input_idcs = list(map(int, opt.input_idcs.split(',')))
    else:
        input_idcs = []

    
    logging_dir = os.path.join(opt.logging_root, opt.log_dir)

    test_set = dataio.SceneClassDataset(root_dir=opt.data_root,
                                       obj_name=opt.obj_name,
                                       max_num_instances=opt.max_num_instances,
                                       specific_observation_idcs=specific_observation_idcs,
                                       max_observations_per_instance=-1,
                                       samples_per_instance=1,
                                       img_sidelength=opt.img_sidelength,
                                       specific_ins = None)
    dataset = DataLoader(test_set,
                         collate_fn=test_set.collate_fn,
                         batch_size=opt.batch_size,
                         shuffle=False,
                         drop_last=False)

    num_classes = test_set.num_classes


    if opt.eval_mode == 'srn':
        print('Loading SRN....')
        assert (opt.checkpoint_path is not None), "Have to pass checkpoint!"
        num_training_instances = torch.load(opt.checkpoint_path)['model']['latent_codes.weight'].shape[0]
        model = SRNsModel(num_classes=test_set.num_classes,
                      num_instances=num_training_instances,
                      latent_dim=opt.embedding_size,
                      tracing_steps=opt.tracing_steps,
                      point_cloud=opt.point_cloud)

    elif opt.eval_mode == 'unet':
        assert (opt.unet_path is not None), "Have to pass checkpoint!"
        print('Loading SRN....')
        model = SRNsModel(num_classes=test_set.num_classes,
                          num_instances=test_set.num_instances,
                          latent_dim=opt.embedding_size,
                          tracing_steps=opt.tracing_steps,
                          point_cloud=opt.point_cloud)
        print('Loading UNet....')
        model_unet = UnetModel(num_classes=test_set.num_classes,)

    elif opt.eval_mode == 'linear':
        assert (opt.linear_path is not None), "Have to pass checkpoint!"
        print('Loading Linear....')
        num_training_instances = torch.load(opt.checkpoint_path)['model']['latent_codes.weight'].shape[0]
        model = SRNsModel(num_classes=test_set.num_classes,
                          num_instances=num_training_instances, 
                          latent_dim=opt.embedding_size,
                          tracing_steps=opt.tracing_steps,
                          point_cloud=opt.point_cloud)

        model_linear = LinearModel(num_classes=test_set.num_classes,)



    print("Loading model from %s" % opt.checkpoint_path)
    util.custom_load(model, path=opt.checkpoint_path, discriminator=None,
                     overwrite_embeddings=False)

    model.eval()
    model.cuda()

    if opt.eval_mode == 'linear':
        print("Loading model from %s" % opt.linear_path)
        util.custom_load(model_linear, path=opt.linear_path, discriminator=None,
                         overwrite_embeddings=False)

        model_linear.eval()
        model_linear.cuda()

    elif opt.eval_mode == 'unet':
        print("Loading model from %s" % opt.unet_path)
        util.custom_load(model_unet, path=opt.unet_path, discriminator=None,
                         overwrite_embeddings=False)

        model_unet.eval()
        model_unet.cuda()

    util.cond_mkdir(logging_dir)

    # Save command-line parameters to log directory.
    with open(os.path.join(logging_dir, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    print('Beginning evaluation...')
    with torch.no_grad():
        mious = list()
        psnrs = list()
        ssims = list()

        instance_mious = list()
        instance_psnrs = list()
        instance_ssims = list()

        part_intersect = np.zeros(num_classes, dtype=np.float32)
        part_union = np.zeros(num_classes, dtype=np.float32)
        confusion = np.zeros((num_classes, 3), dtype=int) # TODO: find a way to generalize this
        
        output_flag = 1
        instance_idx = 0
        idx = 0
        max_classes = 0
        global_dict_data = []
        main_pc_seg = []
        main_pc_rgb = []
        global_data_columns = ['instance_num', 'instance_id', 'miou', 'stdiou', 'psnr', 'ssim' ,'max_classes']

        for model_input, ground_truth in dataset:
            if opt.eval_mode == 'linear' or opt.eval_mode == 'srn':
                model_outputs = model(model_input)

            else:
                model_outputs = model(model_input)
                srn_preds = model.get_output_img(model_outputs).cuda()
                prediction = model_unet(srn_preds)

                prediction = torch.reshape(prediction, (prediction.shape[0], prediction.shape[1], prediction.shape[2] * prediction.shape[2]))
                pred_segs = prediction.permute(0, 2, 1)


            if opt.eval_mode == 'linear':
                seg_out = model_linear(model_outputs['features'])
                model_outputs.update(seg_out)


            instance_idcs = model_input['instance_idx']

            if instance_idx >= opt.save_out_first_n:
                output_flag = 0

            trgt_imgs = model_input['rgb'].cuda()
            trgt_segs = model_input['seg'].cuda()
            trgt_imgs = util.lin2img(trgt_imgs)
            trgt_segs_display = model.get_output_seg(trgt_segs, trgt=True)

            if opt.eval_mode == 'linear' or opt.eval_mode == 'srn':
                output_imgs = model.get_output_img(model_outputs).cpu().numpy()
                output_segs = model.get_output_seg(model_outputs)
                pred_segs = model_outputs['seg']
                comparisons = model.get_comparisons(model_input,
                                                    model_outputs,
                                                    ground_truth)
                if opt.point_cloud:
                    output_pc_seg = model.get_output_pc(model_outputs, seg=True)
                    output_pc_rgb = model.get_output_pc(model_outputs, seg=False)

            else:
                output_imgs = trgt_imgs
                output_segs = model_unet.get_output_seg(pred_segs)
                comparisons = model_unet.get_comparisons_unet(output_segs, trgt_segs_display)

            for i in range(len(output_imgs)):
                prev_instance_idx = instance_idx
                instance_idx = instance_idcs[i]

                if prev_instance_idx != instance_idx:
                    idx = 0
                    global_sample = {'instance_num': prev_instance_idx.cpu().numpy(), 'instance_id': model_input['instance_id'][i-1],
                                         'miou': np.mean(instance_mious) ,'stdiou': np.std(instance_mious), 'psnr': np.mean(instance_psnrs),
                                         'ssim': np.mean(instance_ssims), 'max_classes': max_classes}
                    global_dict_data.append(global_sample)
                    max_classes = 0
                    instance_mious.clear()
                    instance_psnrs.clear()
                    instance_ssims.clear()
                    if opt.point_cloud:
                        main_pc_seg = np.concatenate(main_pc_seg)
                        np.savetxt(os.path.join(instance_dir, 'point_cloud_seg.txt'), main_pc_seg)
                        main_pc_seg = []

                        main_pc_rgb = np.concatenate(main_pc_rgb)
                        np.savetxt(os.path.join(instance_dir, 'point_cloud_rgb.txt'), main_pc_rgb)
                        main_pc_rgb = []


                numclasses = np.unique(trgt_segs.cpu().numpy()).shape[0]
                if numclasses > max_classes:
                    max_classes = numclasses

                newIOU = model.get_IOU_vals(pred_segs[i].unsqueeze(0), trgt_segs[i], confusion, part_intersect, part_union)
                instance_mious.append(newIOU)
                mious.append(newIOU)
    
                psnr, ssim = model.get_psnr(model_outputs, ground_truth)
                psnr = psnr[0]
                ssim = ssim[0]
                instance_psnrs.append(psnr)
                psnrs.append(psnr)
                instance_ssims.append(ssim)
                ssims.append(ssim)

                print('instance %04d   observation %03d   miou %0.4f   psnr %0.4f   ssim %0.4f' % (int(instance_idx.cpu().numpy()), idx , newIOU, psnr, ssim))

                if output_flag:
                    instance_dir = os.path.join(logging_dir, "%06d" % instance_idx)
                    rgb = os.path.join(instance_dir, 'rgb')
                    rgb_gt = os.path.join(instance_dir, 'rgb_gt')
                    seg = os.path.join(instance_dir, 'seg')
                    seg_gt = os.path.join(instance_dir, 'seg_gt')
                    input_img = os.path.join(instance_dir, 'input')
                    comp = os.path.join(instance_dir , 'comparison')
                    
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

                    if opt.point_cloud:
                        pc_out_seg = output_pc_seg[i, :, :]
                        pc_out_rgb = output_pc_rgb[i, :, :]
                        background = pc_out_seg[:, 3:] == [255, 255, 255]
                        pc_out_rgb = pc_out_rgb[np.where(background == False)[0]]
                        pc_out_seg = pc_out_seg[np.where(background == False)[0]]

                        if idx % (test_set.num_per_instance_observations[0]//10) == 0: # 10 sets the amt of samples for the point cloud
                            main_pc_seg += [pc_out_seg]
                            main_pc_rgb += [pc_out_rgb]

                    if idx in input_idcs:
                        if opt.reverse:
                            input_out = util.convert_image(trgt_segs_display[i].squeeze())
                        else:
                            input_out = util.convert_image(trgt_imgs[i].squeeze())
                        util.write_img(input_out, os.path.join(input_img, "%06d.png" % idx))
                    util.write_img(rgb_out, os.path.join(rgb, "%06d.png" % idx))
                    util.write_img(rgb_gt_out, os.path.join(rgb_gt, "%06d.png" % idx))
                    util.write_img(seg_out, os.path.join(seg, "%06d.png" % idx))
                    util.write_img(seg_gt_out, os.path.join(seg_gt, "%06d.png" % idx))
                    util.write_img(comp_out, os.path.join(comp, "%06d.png" % idx))


                idx += 1

    if opt.point_cloud:
        main_pc_seg = np.concatenate(main_pc_seg)
        main_pc_rgb = np.concatenate(main_pc_rgb)
        print(main_pc_rgb.shape)
        np.savetxt(os.path.join(instance_dir, 'point_cloud_seg.txt'), main_pc_seg)
        np.savetxt(os.path.join(instance_dir, 'point_cloud_rgb.txt'), main_pc_rgb)
    global_sample = {'instance_num': instance_idx.cpu().numpy(), 'instance_id': model_input['instance_id'][i],
                     'miou': np.mean(instance_mious), 'stdiou': np.std(instance_mious), 'psnr':np.mean(instance_psnrs), 
                     'ssim': np.mean(instance_ssims),  'max_classes': max_classes}
    global_dict_data.append(global_sample)
    csv_file = os.path.join(logging_dir, 'final_miou.csv')
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=global_data_columns)
        writer.writeheader()
        for data in global_dict_data:
            writer.writerow(data)
    print('Total Num Images: ', len(mious))
    mIOU_mean = np.mean(mious)
    part_intersect = np.delete(part_intersect, np.where(part_union == 0))
    part_union = np.delete(part_union, np.where(part_union == 0))
    part_iou = np.divide(part_intersect[0:], part_union[0:])
    mean_part_iou = np.mean(part_iou)

    with open(os.path.join(logging_dir, "results.txt"), "w") as out_file:
       out_file.write("mIOU %0.6f, PSNR %0.6f, SSIM %0.6f" % (np.mean(mious), np.mean(psnrs), np.mean(ssims)))

    print('mIOU: ', mIOU_mean)
    print('Category mIoU: %f, %s' % (mean_part_iou, str(part_iou)))
    print('mPSNR: ', np.mean(psnrs))
    print('mSSIM: ', np.mean(ssim))



def main():
    test()


if __name__ == '__main__':
    main()

