#!/usr/bin/env python3

import argparse
import os, time, datetime
import random
import itertools

import torch
from torch import nn
import torchvision
import numpy as np
import cv2
#import adabound

from torch.utils.data import DataLoader

#from srns_vincent import *
from srns_vincent import *
#from tensorboardX import SummaryWriter
from data_util import *
import util
from class_dataio import *

import trajectories

# params
parser = argparse.ArgumentParser()

parser.add_argument('--data_root', required=True, help='path to file list of h5 train data')
parser.add_argument('--stat_root', required=True, help='path to file list of h5 train data')
parser.add_argument('--obj_name', required=True, help='path to file list of h5 train data')
parser.add_argument('--logging_root', type=str, default='/media/staging/deep_sfm/',
                    required=False, help='path to file list of h5 train data')

parser.add_argument('--tracing_steps', type=int, default=10, help='Number of steps of intersection tester')
parser.add_argument('--experiment_name', type=str, default='', help='path to file list of h5 train data')

parser.add_argument('--img_sidelength', type=int, default=128, required=False, help='start epoch')

parser.add_argument('--num_images', type=int, default=-1, required=False, help='start epoch')
parser.add_argument('--num_samples', type=int, default=64**2, required=False, help='start epoch')
parser.add_argument('--embedding_size', type=int, required=True, help='start epoch')

parser.add_argument('--no_preloading', action='store_true', default=False, help='#images')
parser.add_argument('--num_objects', type=int, default=-1, help='start epoch')
parser.add_argument('--orthographic', action='store_true', default=False, help='start epoch')

parser.add_argument('--checkpoint_srn', default=None, help='model to load')
parser.add_argument('--checkpoint_linear', default=None, help='model to load')
parser.add_argument('--renderer', type=str, default='fc', help='start epoch')
parser.add_argument('--implicit_nf', type=int, default=256, help='start epoch')


opt = parser.parse_args()
print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

device = torch.device('cuda')


def interpolate(trajectory, obj_idcs):

    # train_dataset = dataio.SceneClassDataset(root_dir=opt.data_root,
    #                                          stat_dir=opt.stat_root,
    #                                          obj_name=opt.obj_name,
    #                                          img_sidelength=img_sidelengths[0],
    #                                          specific_observation_idcs=obj_idcs,
    #                                          samples_per_instance=1)

    model_srn =  SRNsModel(num_instances=1216,#train_dataset.num_instances,
                      latent_dim=opt.embedding_size,
                      tracing_steps=opt.tracing_steps)

    model_linear = LinearModel()

    # model = DeepRayModel(num_objects=opt.num_objects,
    #                      embedding_size=opt.embedding_size,
    #                      implicit_nf=opt.implicit_nf,
    #                      has_params=True,
    #                      renderer=opt.renderer,
    #                      mode='hyper',
    #                      use_gt_depth=False,
    #                      tracing_steps=opt.tracing_steps)

    if opt.checkpoint_linear is not None:
        print("Loading model from %s"%opt.checkpoint_linear)
        util.custom_load_linear(model_linear, path=opt.checkpoint_linear,
                         discriminator=None,
                         overwrite_embeddings=False)
    else:
        print("Have to give checkpoint!")
        return


    if opt.checkpoint_srn is not None:
        print("Loading model from %s"%opt.checkpoint_srn)
        util.custom_load_linear(model_srn, path=opt.checkpoint_srn,
                         discriminator=None,
                         overwrite_embeddings=False)
    else:
        print("Have to give checkpoint!")
        return

    model_linear.eval()
    model_linear.cuda()

    model_srn.eval()
    model_srn.cuda()


    # directory structure: month_day/
    dir_name = os.path.join(datetime.datetime.now().strftime('%m_%d'),
                            datetime.datetime.now().strftime('%H-%M-%S_') +
                            '_'.join(opt.checkpoint_linear.strip('/').split('/')[-2:])[:200])

    #traj_dir = os.path.join(opt.logging_root, dir_name)
    traj_dir = opt.logging_root
    cond_mkdir(traj_dir)

    # Save parameters used into the log directory.
    with open(os.path.join(traj_dir, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    interpolation_frames = len(trajectory)

    # Assemble model input
    uv = np.mgrid[0:opt.img_sidelength, 0:opt.img_sidelength].astype(np.int32)
    uv = torch.from_numpy(np.flip(uv, axis=0).copy()).long()
    uv = uv.reshape(2, -1).transpose(1, 0)

    intrinsics, _, _, world2cam_poses = util.parse_intrinsics(os.path.join(opt.data_root, 'intrinsics.txt'),
                                                              trgt_sidelength=opt.img_sidelength)
    index = 0
    print(len(obj_idcs))
    print('Beginning evaluation...')
    with torch.no_grad():
        for i in range(1, len(obj_idcs)):
            print(i)
            embedding_1 = model_srn.latent_codes(torch.tensor(obj_idcs[i-1]).cuda())
            embedding_2 = model_srn.latent_codes(torch.tensor(obj_idcs[i]).cuda())

            # interpolation_dir = os.path.join(traj_dir, "%06d"%i)
            interpolation_dir = traj_dir
            cond_mkdir(interpolation_dir)

            for j, pose in enumerate(trajectory):
                if not j%10:
                    print(j)

                # model_input = RayBundle(obj_idx=torch.Tensor([0]).squeeze(),
                #                         rgb=torch.Tensor([0]).float(),
                #                         pose=torch.from_numpy(pose).float().unsqueeze(0),
                #                         depth=torch.Tensor([0]).float(),
                #                         xy=xy.unsqueeze(0),
                #                         param=torch.Tensor([0]).float(),
                #                         intrinsics=torch.from_numpy(intrinsics).float().unsqueeze(0))

                model_input = {
                    "instance_idx": torch.Tensor(np.array([0])),
                    "rgb": torch.from_numpy(np.array([0])).float(),
                    "pose": torch.from_numpy(pose).float().unsqueeze(0),
                    "uv": uv.unsqueeze(0),
                    "seg": torch.from_numpy(np.array([0])).int(),
                    "intrinsics": torch.from_numpy(intrinsics).float().unsqueeze(0),
                    "instance_id":torch.Tensor(np.array([0]))
                }

                # model_input = {'instance_idx':0,
                #                'intrinsics': torch.from_numpy(intrinsics).float().unsqueeze((0)),
                #                }

                weight_1 = (interpolation_frames - j)/interpolation_frames
                weight_2 = j/interpolation_frames
                int_embedding = weight_1 * embedding_1 + weight_2 * embedding_2

                model_outputs_srn = model_srn(model_input, int_embedding)
                model_outputs = model_linear(model_outputs_srn['features'])
                model_outputs.update({'rgb': model_outputs_srn['rgb']})
                model_outputs.update({'depth': model_outputs_srn['depth']})

                model_srn.write_eval(model_input, model_outputs, interpolation_dir, "%06d.png"%(index))
                index = index + 1


def main():
    trajectory = trajectories.around(look_at_fn=trajectories.look_at_cars, radius=1.5, altitude=22.5) #See blender #radius=2.5
    obj_idcs = [3, 42]
    permutations = list(itertools.permutations(obj_idcs))
    random.seed(0)
    random.shuffle(permutations)
    permutations = permutations[:5]

    list_of_obj_idcs = list()

    for permutation in permutations:
        list_of_obj_idcs.extend(list(permutation))
    interpolate(trajectory,  list_of_obj_idcs)


if __name__ == '__main__':
    main()
