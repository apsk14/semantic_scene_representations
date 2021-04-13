import configargparse
import os, time, datetime

import torch
import numpy as np
import csv

import class_dataio as dataio
from torch.utils.data import DataLoader
from srns_vincent import *
#from linear import *
import util

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

# Note: in contrast to training, no multi-resolution!
p.add_argument('--img_sidelength', type=int, default=128, required=False,
               help='Sidelength of test images.')

p.add_argument('--data_root', required=True, help='Path to directory with training data.')

p.add_argument('--stat_root', required=True, help='Path to directory with statistics data.')

p.add_argument('--obj_name', required=True,type=str, help='Name of object in question')

p.add_argument('--eval_mode', required=True,type=str, help='Which model to evaluate (linear, unet, srn)')

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

    test_set = dataio.SceneClassDataset(root_dir=opt.data_root,
                                       stat_dir=opt.stat_root,
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

    if opt.eval_mode == 'srn':
        print('Loading SRN....')
        model = SRNsModel(num_instances= test_set.num_instances,
                      latent_dim=opt.embedding_size,
                      tracing_steps=opt.tracing_steps)

    elif opt.eval_mode == 'unet':
        print('Loading SRN....')
        model = SRNsModel(num_instances=test_set.num_instances,
                          latent_dim=opt.embedding_size,
                          tracing_steps=opt.tracing_steps)
        print('Loading UNet....')
        model_unet = UnetModel()

    elif opt.eval_mode == 'linear':
        print('Loading Linear....')
        model = SRNsModel(num_instances=4489,   #num_instances=test_set.num_instances
                          latent_dim=opt.embedding_size,
                          tracing_steps=opt.tracing_steps)

        model_linear = LinearModel()




    assert (opt.checkpoint_path is not None), "Have to pass checkpoint!"

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

    # directory structure: month_day/
    #renderings_dir = os.path.join(opt.logging_root, 'renderings')
    util.cond_mkdir(opt.logging_root)
    #util.cond_mkdir(renderings_dir)

    # Save command-line parameters to log directory.
    # with open(os.path.join(opt.logging_root, "params.txt"), "w") as out_file:
    #     out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    print('Beginning evaluation...')
    with torch.no_grad():
        instance_idx = 0
        idx = 0
        max_classes = 0

        for model_input, ground_truth in dataset:

            model_outputs = model(model_input)
            instance_idcs = model_input['instance_idx']
            trgt_segs = model_input['seg']
            #trgt_segs_display = model.get_output_seg(trgt_segs, trgt=True)
            features = model_outputs['features'].cpu().numpy()
            output_imgs = model.get_output_img(model_outputs).cpu().numpy()

            for i in range(len(output_imgs)):
                prev_instance_idx = instance_idx
                instance_idx = instance_idcs[i]

                if prev_instance_idx != instance_idx:
                    idx = 0
                    max_classes = 0


                numclasses = np.unique(trgt_segs.cpu().numpy()).shape[0]
                if numclasses > max_classes:
                    max_classes = numclasses

                features_with_label = np.concatenate([features.squeeze(), trgt_segs.squeeze()[:, None]], axis=1)
                name = 'ins_' + str(int(instance_idx.item())) + '_pose_' + str(idx+1)
                np.save(os.path.join(opt.logging_root, name), features_with_label)
                print(name)
                # if instance_idx.item() == 50:
                #     print(np.unique(trgt_segs))
                #     seg_gt_out = util.convert_image(trgt_segs_display[i].squeeze())
                #     #util.write_img(seg_gt_out, os.path.join(opt.logging_root, name+'.png'))
                #     np.save(os.path.join(opt.logging_root, name), features_with_label)
                #     exit()



                idx += 1


def main():
    test()


if __name__ == '__main__':
    main()

