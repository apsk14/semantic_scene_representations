import os
import torch
import numpy as np
from glob import glob
import data_util
import util

from collections import namedtuple
Observation = namedtuple('observation', 'instance_idx rgb seg depth uv pose intrinsics param pts rgb_pts labels')

class Preloader():
    def __init__(self, paths, load_to_ram, loading_function):
        self.load_to_ram = load_to_ram
        self.paths = paths
        self.loading_function = loading_function

        if self.load_to_ram:
            print("Buffering...")
            self.all_items = []

            for path in paths:
                self.all_items.append(loading_function(path))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self.load_to_ram:
            return self.all_items[idx]
        else:
            return self.loading_function(self.paths[idx])

def pick(list, item_idcs):
    if not list:
        return list
    return [list[i] for i in item_idcs]


class SceneInstanceDataset():
    def __init__(self,
                 instance_idx,
                 instance_dir,
                 load_to_ram,
                 specific_observation_idcs=None,  # For few-shot case: Can pick specific observations only
                 img_sidelength=None,
                 num_images=-1,
                 part_name2id={},
                 part_old2new={}):
        super().__init__()

        self.instance_idx = instance_idx

        object_name = instance_dir.split('/')[-4]
        object_dir = instance_dir.split('/')[-3]
        if object_dir == 'Table.train_val':
            object_dir = 'Table.train'
        og_dir = '/media/data1/apsk14/srn_seg_data/' + object_name + '/' + object_dir


        color_dir = os.path.join(instance_dir, 'rgb')
        seg_dir = os.path.join(instance_dir, 'seg')
        pose_dir = os.path.join(instance_dir, 'pose')
        depth_dir = os.path.join(instance_dir, 'depth')
        param_dir = os.path.join(instance_dir, 'params')
        pn_dir = os.path.join(og_dir, os.path.basename(os.path.split(instance_dir)[0]), 'partnet')
        if not os.path.isdir(color_dir):
            print("Error! root dir %s is wrong" % instance_dir)
            return

        self.has_depth = os.path.isdir(depth_dir)
        self.has_params = os.path.isdir(param_dir)

        self.color_paths = sorted(data_util.glob_imgs(color_dir))
        self.seg_paths = sorted(data_util.glob_imgs(seg_dir))
        self.pose_paths = sorted(glob(os.path.join(pose_dir, '*.txt')))
        self.pts_path = os.path.join(pn_dir, 'point_sample', 'sample-points-all-pts-nor-rgba-10000.txt')
        self.labels_path = os.path.join(pn_dir, 'point_sample', 'sample-points-all-label-10000.txt')
        self.mapping_path = os.path.join(pn_dir, 'result.json')
        self.transfer_path = os.path.join(pn_dir, 'result_after_merging.json')

        if self.has_depth:
            self.depth_paths = sorted(glob(os.path.join(depth_dir, '*.png')))
        else:
            self.depth_paths = []

        if self.has_params:
            self.param_paths = sorted(glob(os.path.join(param_dir, '*.txt')))
        else:
            self.param_paths = []

        if specific_observation_idcs is not None:
            self.color_paths = pick(self.color_paths, specific_observation_idcs)
            self.seg_paths = pick(self.seg_paths, specific_observation_idcs)
            self.pose_paths = pick(self.pose_paths, specific_observation_idcs)
            self.depth_paths = pick(self.depth_paths, specific_observation_idcs)
            self.param_paths = pick(self.param_paths, specific_observation_idcs)
        elif num_images != -1:
            idcs = np.linspace(0, stop=len(self.color_paths), num=num_images, endpoint=False, dtype=int)
            self.color_paths = pick(self.color_paths, idcs)
            self.seg_paths = pick(self.seg_paths, idcs)
            self.pose_paths = pick(self.pose_paths, idcs)
            self.depth_paths = pick(self.depth_paths, idcs)
            self.param_paths = pick(self.param_paths, idcs)

        load_to_ram = False

        transfer_map = data_util.load_transfer_map(self.transfer_path, part_name2id)


        self.rgbs = Preloader(self.color_paths,
                              load_to_ram=load_to_ram,
                              loading_function=lambda path: data_util.load_rgb(path, sidelength=img_sidelength))

        self.segs = Preloader(self.seg_paths,
                              load_to_ram=load_to_ram,
                              loading_function=lambda path: data_util.transfer_labels(path, transfer_map, sidelength=img_sidelength))
        self.poses = Preloader(self.pose_paths,
                               load_to_ram=load_to_ram,
                               loading_function=data_util.load_pose)
        self.depths = Preloader(self.depth_paths,
                                load_to_ram=load_to_ram,
                                loading_function=lambda path: data_util.load_depth(path, sidelength=img_sidelength))
        self.params = Preloader(self.param_paths,
                                load_to_ram=load_to_ram,
                                loading_function=data_util.load_params)

        self.pts, self.rgb_pts = data_util.load_pts(self.pts_path)

        label_map = data_util.load_label_map(self.mapping_path)
        self.labels = data_util.load_label(self.labels_path, label_map, part_name2id, part_old2new)

        self.img_width, self.img_height = self.rgbs[0].shape[1], self.rgbs[0].shape[2]

        self.dummy = np.zeros((self.img_width*self.img_height, 1))

        intrinsics, _, _, world2cam_poses = util.parse_intrinsics(os.path.join(instance_dir, 'intrinsics.txt'),
                                                                  trgt_sidelength=self.img_width)
        self.intrinsics = torch.Tensor(intrinsics).float()

        print("*"*20)
        print(instance_dir)
        print(intrinsics)
        print(world2cam_poses)
        print(len(self.rgbs), len(self.segs), len(self.poses), len(self.depths))

    def __len__(self):
        return len(self.pose_paths)

    def __getitem__(self, idx):
        if self.has_depth:
            depth = self.depths[idx]
        else:
            depth = self.dummy

        uv = np.mgrid[0:self.img_width, 0:self.img_height].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).long()

        uv = uv.reshape(2,-1).transpose(1,0)
        rgbs = self.rgbs[idx].reshape(3, -1).transpose(1, 0)
        segs = self.segs[idx].reshape(1, -1).transpose(1, 0)
        # rgbs = self.rgbs[idx]
        # segs = self.segs[idx]
        depths = depth.reshape(-1, 1)

        return Observation(instance_idx=torch.Tensor([self.instance_idx]).squeeze(),
                           rgb=torch.from_numpy(rgbs).float(),
                           seg=torch.from_numpy(segs).int(),
                           pose=torch.from_numpy(self.poses[idx]).float(),
                           depth=torch.from_numpy(depths).float(),
                           uv=uv,
                           param=torch.from_numpy(self.params[idx]).float() if self.has_params else torch.Tensor([0]).float(),
                           intrinsics=self.intrinsics,
                           pts=torch.from_numpy(self.pts),
                           rgb_pts=torch.from_numpy(self.rgb_pts),
                           labels=torch.from_numpy(self.labels))


class SceneClassDataset():
    def __init__(self,
                 root_dir,
                 preload=True,
                 img_sidelength=None,
                 max_num_instances=-1,
                 max_observations_per_instance=-1,
                 specific_observation_idcs=None,  # For few-shot case: Can pick specific observations only
                 samples_per_instance=2):
        super().__init__()

        self.samples_per_instance = samples_per_instance


        print(root_dir)
        obj_name = root_dir.split('/')[-3]
        self.instance_dirs = sorted(glob(os.path.join(root_dir, '*/')))
        print('\n'.join(self.instance_dirs))

        assert (len(self.instance_dirs) != 0), "No objects in the data directory"

        if max_num_instances != -1:
            self.instance_dirs = self.instance_dirs[:max_num_instances]

        in_fn = '/media/data1/apsk14/srn_seg_data/' + obj_name + '/' + obj_name + '-level-1.txt'
        with open(in_fn, 'r') as fin:
            part_name2id = {d.split()[1]: (cnt + 1) for cnt, d in enumerate(fin.readlines())}
        in_fn = '/media/data1/apsk14/srn_seg_data/' + obj_name + '/' + obj_name + '.txt'
        with open(in_fn, 'r') as fin:
            part_old2new = {d.rstrip().split()[0]: d.rstrip().split()[1] for d in fin.readlines()}

        self.all_instances = [SceneInstanceDataset(instance_idx=idx,
                                                   instance_dir=dir,
                                                   load_to_ram=preload,
                                                   specific_observation_idcs=specific_observation_idcs,
                                                   img_sidelength=img_sidelength,
                                                   num_images=max_observations_per_instance,
                                                   part_name2id=part_name2id,
                                                   part_old2new=part_old2new)
                              for idx, dir in enumerate(self.instance_dirs)]

        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]
        self.num_instances = len(self.all_instances)

    def __len__(self):
        return np.sum([len(obj_ds) for obj_ds in self.all_instances])

    def get_instance_idx(self, idx):
        '''Maps an index into all tuples of all objects to the idx of the tuple relative to the other tuples of that
        object
        '''
        obj_idx = 0
        while idx >= 0:
            idx -= self.num_per_instance_observations[obj_idx]
            obj_idx += 1
        return obj_idx - 1, int(idx + self.num_per_instance_observations[obj_idx - 1])

    def collate_fn(self, batch_list):
        model_inputs, trgts = zip(*batch_list)

        # Flatten the list of lists of inputs
        flat_inputs = []
        for sublist in model_inputs:
            for item in sublist:
                flat_inputs.append(item)

        num_input_members = len(flat_inputs[0])
        all_input_members = [list() for _ in range(num_input_members)]

        for i in range(len(flat_inputs)):
            for j in range(num_input_members):
                all_input_members[j].append(flat_inputs[i][j])

        model_inputs = Observation(*tuple(torch.stack(all_input_members[j], dim=0) for j in range(num_input_members)))

        # Flatten the list of lists of targets
        num_trgt_members = len(trgts[0])
        all_trgt_members = [list() for _ in range(num_trgt_members)]

        for i in range(len(trgts)):
            for j in range(num_trgt_members):
                all_trgt_members[j].extend(trgts[i][j])

        model_trgts = tuple([torch.stack(all_trgt_members[i], dim=0) for i in range(num_trgt_members)])

        return model_inputs, model_trgts

    def __getitem__(self, idx):
        obj_idx, rel_idx = self.get_instance_idx(idx)

        observations = []
        observations.append(self.all_instances[obj_idx][rel_idx])

        for i in range(self.samples_per_instance - 1):
            observations.append(self.all_instances[obj_idx][np.random.randint(len(self.all_instances[obj_idx]))])

        return observations, ([ray_bundle.rgb for ray_bundle in observations], [ray_bundle.seg for ray_bundle in observations],
                             [ray_bundle.depth for ray_bundle in observations])

