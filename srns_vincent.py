import torch
import os
import torch.nn as nn
import numpy as np
import random
import cv2
import imageio

import torchvision
from custom_layers import *
import util

from dataio import Observation
import data_util
import skimage.measure, skimage.transform
from torch.nn import functional as F

from pytorch_prototyping import pytorch_prototyping
import hyperlayers

import matplotlib.cm as cm

#NUM_CLASSES = 18 # LAMPS
NUM_CLASSES = 6 # Chairs

class UnetModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.unet = pytorch_prototyping.Unet(
            in_channels=4,
            out_channels=NUM_CLASSES,
            nf0=64,
            num_down=7,
            max_channels=512,
            use_dropout=True,
            upsampling_mode='transpose',
            dropout_prob=0.1,
            norm=nn.BatchNorm2d,
            outermost_linear=True)

        # colors for displaying segmented images
        self.colors = np.concatenate([np.array([[1., 1., 1.]]),
                                      cm.rainbow(np.linspace(0, 1, NUM_CLASSES - 1))[:, :3]],
                                     axis=0)
        # loss fn
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

        # List of logs
        self.logs = list()

        print("*" * 100)
        print(self)
        print("*" * 100)
        print("Number of parameters:")
        util.print_network(self)
        print("*" * 100)

    def forward(self, input, z=None):
        self.logs = list()
        novel_views_seg = self.unet(input.cuda())
        return novel_views_seg

class SRNsModel(nn.Module):
    def __init__(self,
                 num_instances,
                 latent_dim,
                 tracing_steps):
        super().__init__()

        self.latent_dim = latent_dim

        self.colors = np.concatenate([np.array([[1.,1.,1.]]),
                                      cm.rainbow(np.linspace(0, 1, NUM_CLASSES-1))[:,:3]],
                                     axis=0)

        self.num_hidden_units_phi = 256
        self.phi_layers = 4  # includes the in and out layers
        self.rendering_layers = 5  # includes the in and out layers
        self.sphere_trace_steps = tracing_steps

        self.latent_codes = nn.Embedding(num_instances, latent_dim, sparse=True).cuda()
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

        self.hyper_phi = hyperlayers.HyperFC(hyper_in_ch=self.latent_dim,
                                             hyper_num_hidden_layers=1,
                                             hyper_hidden_ch=self.latent_dim,
                                             hidden_ch=self.num_hidden_units_phi,
                                             num_hidden_layers=self.phi_layers - 2,
                                             in_ch=3,
                                             out_ch=self.num_hidden_units_phi)

        self.ray_marcher = Raymarcher(num_feature_channels=self.num_hidden_units_phi,
                                      raymarch_steps=self.sphere_trace_steps)

        self.pixel_generator = pytorch_prototyping.FCBlock(hidden_ch=self.num_hidden_units_phi,
                                                           num_hidden_layers=self.rendering_layers - 1,
                                                           in_features=self.num_hidden_units_phi,
                                                           out_features=4,
                                                           outermost_linear=True)

        self.class_generator = pytorch_prototyping.Unet(in_channels=self.num_hidden_units_phi,
                                                        out_channels=NUM_CLASSES,
                                                        outermost_linear=True,
                                                        use_dropout=False,
                                                        dropout_prob=0.1,
                                                        nf0=64,
                                                        norm=nn.BatchNorm2d,
                                                        max_channels=128,
                                                        num_down=4)
        self.counter = 0

        # Losses
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

        # List of logs
        self.logs = list()

        print("*" * 100)
        print(self)
        print("*" * 100)
        print("Number of parameters:")
        util.print_network(self)
        print("*" * 100)

    def get_regularization_loss(self, prediction, ground_truth):
        '''Computes regularization loss on final depth map (L_{depth} in eq. 6 in paper)

        :param prediction (tuple): Output of forward pass.
        :param ground_truth: Ground-truth (unused).
        :return: Regularization loss on final depth map.
        '''
        depth = prediction['depth']

        neg_penalty = (torch.min(depth, torch.zeros_like(depth)) ** 2)
        return torch.mean(neg_penalty) * 10000

    def get_image_loss(self, prediction, ground_truth):
        '''Computes loss on predicted image (L_{img} in eq. 6 in paper)

        :param prediction (tuple): Output of forward pass.
        :param ground_truth: Ground-truth (unused).
        :return: image reconstruction loss.
        '''
        pred_imgs = prediction['rgb']
        trgt_imgs = ground_truth['rgb']

        trgt_imgs = trgt_imgs.cuda()

        loss = self.l2_loss(pred_imgs, trgt_imgs)
        return loss

    def get_seg_loss(self, prediction, ground_truth):
        '''Computes loss on predicted image (L_{img} in eq. 6 in paper)

        :param prediction (tuple): Output of forward pass.
        :param ground_truth: Ground-truth (unused).
        :return: image reconstruction loss.
        '''
        pred_segs = prediction['seg']
        pred_segs = pred_segs.permute(0, 2, 1)

        trgt_segs = ground_truth['seg']
        trgt_segs = trgt_segs.permute(0, 2, 1).squeeze().long().cuda()

        loss = self.cross_entropy_loss(pred_segs, trgt_segs)
        return loss

    def get_latent_loss(self):
        '''Computes loss on latent code vectors (L_{latent} in eq. 6 in paper)
        :return: Latent loss.
        '''
        self.latent_reg_loss = torch.mean(self.z ** 2)

        return self.latent_reg_loss

    def get_psnr(self, prediction, ground_truth):
        '''Compute PSNR of model image predictions.

        :param prediction: Return value of forward pass.
        :param ground_truth: Ground truth.
        :return: (psnr, ssim): tuple of floats
        '''
        pred_imgs = prediction['rgb']
        trgt_imgs = ground_truth['rgb'].cuda()

        batch_size = pred_imgs.shape[0]

        if not isinstance(pred_imgs, np.ndarray):
            pred_imgs = util.lin2img(pred_imgs).detach().cpu().numpy()

        if not isinstance(trgt_imgs, np.ndarray):
            trgt_imgs = util.lin2img(trgt_imgs).detach().cpu().numpy()

        psnrs, ssims = list(), list()
        for i in range(batch_size):
            p = pred_imgs[i].squeeze().transpose(1, 2, 0)
            trgt = trgt_imgs[i].squeeze().transpose(1, 2, 0)

            p = (p / 2.) + 0.5
            p = np.clip(p, a_min=0., a_max=1.)

            trgt = (trgt / 2.) + 0.5

            ssim = skimage.measure.compare_ssim(p, trgt, multichannel=True, data_range=1)
            psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)

            psnrs.append(psnr)
            ssims.append(ssim)

        return psnrs, ssims

    def get_comparisons(self, model_input, pred, ground_truth=None):
        pred_rgb, pred_seg, pred_depth = pred['rgb'], pred['seg'], pred['depth']
        batch_size = pred_rgb.shape[0]

        # Parse model input.
        intrinsics = model_input["intrinsics"].cuda()
        uv = model_input["uv"].cuda().float()

        x_cam = uv[:, :, 0].view(batch_size, -1)
        y_cam = uv[:, :, 1].view(batch_size, -1)
        z_cam = pred_depth.view(batch_size, -1)

        normals = geometry.compute_normal_map(x_img=x_cam, y_img=y_cam, z=z_cam, intrinsics=intrinsics)
        normals = F.pad(normals, pad=(1, 1, 1, 1), mode='constant', value=1.)

        pred_rgb = util.lin2img(pred_rgb)[:,0:3,:,:]

        pred_seg = self.get_output_seg(pred)
        pred_seg = torch.from_numpy(pred_seg)

        if ground_truth is not None:
            trgt_imgs, trgt_segs = model_input['rgb'],model_input['seg']
            trgt_imgs = util.lin2img(trgt_imgs)[:,0:3,:,:]
            trgt_segs = util.lin2img(trgt_segs)
            trgt_segs = (self.colors[trgt_segs.cpu().numpy()])[:,0,:,:].transpose(0, 3, 1, 2)
            trgt_segs = torch.from_numpy(trgt_segs)


            return torch.cat((normals.cpu(), pred_rgb.cpu(), trgt_imgs.cpu(),
                              pred_seg.cpu().float(), trgt_segs.cpu().float()), dim=3).numpy()
        else:
            return torch.cat((normals.cpu(), pred_rgb.cpu()), dim=3).numpy()


    def get_comparisons_unet(self, model_input, pred_rgb, pred_seg,trgt_imgs, trgt_segs, model_output, ground_truth=None):
        batch_size = pred_rgb.shape[0]
        pred_depth = model_output['depth']
        # Parse model input.
        intrinsics = model_input["intrinsics"].cuda()
        uv = model_input["uv"].cuda().float()

        x_cam = uv[:, :, 0].view(batch_size, -1)
        y_cam = uv[:, :, 1].view(batch_size, -1)
        z_cam = pred_depth.view(batch_size, -1)

        normals = geometry.compute_normal_map(x_img=x_cam, y_img=y_cam, z=z_cam, intrinsics=intrinsics)
        normals = F.pad(normals, pad=(1, 1, 1, 1), mode='constant', value=1.)
        #
        # pred_rgb = util.lin2img(pred_rgb)[:,0:3,:,:]
        #
        # pred_seg = self.get_output_seg(pred_seg, tr)
        pred_seg = torch.from_numpy(pred_seg)
        pred_rgb = torch.from_numpy(pred_rgb)[:,0:3,:,:]

        # if ground_truth is not None:
        #     trgt_imgs, trgt_segs = model_input['rgb'],model_input['seg']
        #     trgt_imgs = util.lin2img(trgt_imgs)[:,0:3,:,:]
        #     trgt_segs = util.lin2img(trgt_segs)
        #     trgt_segs = (self.colors[trgt_segs.cpu().numpy()])[:,0,:,:].transpose(0, 3, 1, 2)
        trgt_imgs = trgt_imgs[:,0:3,:,:]
        trgt_segs = torch.from_numpy(trgt_segs)


        return torch.cat((normals.cpu(), pred_rgb.cpu(), trgt_imgs.cpu(),
                              pred_seg.cpu().float(), trgt_segs.cpu().float()), dim=3).numpy()
        #else:
         #   return torch.cat((normals.cpu(), pred_rgb.cpu()), dim=3).numpy()

    def get_output_img(self, prediction):
        pred = prediction['rgb']
        return util.lin2img(pred)

    def get_output_seg(self, prediction, trgt=False):
        if not trgt:
            pred = prediction['seg']
            pred_segs, pred = torch.max(pred, dim=2)
            pred = pred[:, :, None]
        else:
            pred = prediction
        output_seg = util.lin2img(pred)
        output_seg = (self.colors[output_seg.cpu().numpy().astype(int)])[:,0,:,:].transpose(0, 3, 1, 2)
        return output_seg.astype(np.float32)

    def get_output_seg_unet(self, prediction, trgt=False):
        if not trgt:
            pred_segs, pred = torch.max(prediction, dim=2)
            pred = pred[:, :, None]
        else:
            pred = prediction
        output_seg = util.lin2img(pred)
        output_seg = (self.colors[output_seg.cpu().numpy().astype(int)])[:,0,:,:].transpose(0, 3, 1, 2)
        return output_seg.astype(np.float32)

    def get_IOU_vals(self, prediction , trgt_seg, confusion, part_intersect, part_union): # had arg confusion
        # confusion vector is [true pos, false pos, false neg]
        pred_segs = prediction['seg']
        pred_segs, seg_idx = torch.max(pred_segs, dim=2)
        pred_labels = seg_idx.cpu().numpy().squeeze()
        real_label = trgt_seg.cpu().numpy().squeeze()

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

        if cur_shape_iou_cnt == 0:
            return 1
        return cur_shape_iou_tot / cur_shape_iou_cnt

    def get_IOU_vals_unet(self, prediction , trgt_seg, confusion, part_intersect, part_union): # had arg confusion
        # confusion vector is [true pos, false pos, false neg]
        pred_segs, prediction = torch.max(prediction, dim=2)
        pred_labels = prediction.cpu().numpy().squeeze()
        real_label = trgt_seg.cpu().numpy().squeeze()

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

        if cur_shape_iou_cnt == 0:
            return 1
        return cur_shape_iou_tot / cur_shape_iou_cnt

    def calc_mIOU(self, confusion):
        check_empty = np.sum(confusion, axis=1)
        empty = np.where(check_empty == 0)
        confusion = np.delete(confusion, empty, axis=0)
        num_classes = confusion.shape[0]

        true_pos = confusion[:, 0]
        false_pos = confusion[:, 1]
        false_neg = confusion[:, 2]
        IOU = np.divide(true_pos, (false_pos + false_neg + true_pos),  where=(false_pos + false_neg + true_pos) != 0)
        mIOU = np.sum(IOU, axis=0)/num_classes
        return mIOU

    def write_updates(self, writer, input, predictions, iter, prefix=''):
        '''Writes tensorboard summaries using tensorboardx api.

        :param writer: tensorboardx writer object.
        :param predictions: Output of forward pass.
        :param ground_truth: Ground truth.
        :param iter: Iteration number.
        :param prefix: Every summary will be prefixed with this string.
        '''
        pred_rgb, pred_seg, pred_depth = predictions['rgb'], predictions['seg'], predictions['depth']

        trgt_imgs = input['rgb'].cuda()
        trgt_segs = input['seg'].cuda()
        colors = self.colors

        pred_seg, seg_idx = torch.max(pred_seg, dim=2)
        seg_idx = seg_idx[:,:,None]

        batch_size, num_samples, _ = trgt_imgs.shape

        # Module's own log
        for type, name, content, every_n in self.logs:
            name = prefix + name

            if not iter % every_n:
                if type == 'image':
                    writer.add_image(name, content.detach().cpu().numpy(), iter)
                    writer.add_scalar(name + '_min', content.min(), iter)
                    writer.add_scalar(name + '_max', content.max(), iter)
                elif type == 'figure':
                    writer.add_figure(name, content, iter, close=True)
                elif type == 'histogram':
                    writer.add_histogram(name, content.detach().cpu().numpy(), iter)
                elif type == 'scalar':
                    writer.add_scalar(name, content.detach().cpu().numpy(), iter)
                elif type == 'embedding':
                    writer.add_embedding(mat=content, global_step=iter)

        if not iter % 50:
            # RGB image outputs
            output_vs_gt = torch.cat((pred_rgb, trgt_imgs), dim=0)
            output_vs_gt = util.lin2img(output_vs_gt)
            print('MACKSi', torch.max(output_vs_gt))
            print('DISPRGB', output_vs_gt.shape)
            writer.add_image(prefix + "Output_vs_gt",
                             torchvision.utils.make_grid(output_vs_gt[:,:3,:,:],
                                                         scale_each=False,
                                                         normalize=True).cpu().detach().numpy(),
                             iter)

            # Segmentation image outputs
            output_vs_gt_seg = torch.cat((seg_idx.int(), trgt_segs.int()), dim=0)
            output_vs_gt_seg = util.lin2img(output_vs_gt_seg).int()
            output_vs_gt_seg = torch.from_numpy(colors[output_vs_gt_seg.cpu().numpy()].squeeze()).permute(0,3,1,2)
            print('DISPSEG', output_vs_gt_seg.shape)
            writer.add_image(prefix + "Output_vs_gt_seg",
                             torchvision.utils.make_grid(output_vs_gt_seg[:,:3,:,:],
                                                         scale_each=False,
                                                         normalize=False).cpu().detach().numpy(),
                             iter)

            depth_maps_plot = util.lin2img(pred_depth)
            writer.add_image(prefix + "pred_depth",
                             torchvision.utils.make_grid(depth_maps_plot.repeat(1, 3, 1, 1),
                                                         scale_each=True,
                                                         normalize=True).cpu().detach().numpy(),
                             iter)

        writer.add_scalar(prefix + "out_min", pred_rgb.min(), iter)
        writer.add_scalar(prefix + "out_max", pred_rgb.max(), iter)

        writer.add_scalar(prefix + "trgt_min", trgt_imgs.min(), iter)
        writer.add_scalar(prefix + "trgt_max", trgt_imgs.max(), iter)

        if iter:
            writer.add_scalar(prefix + "latent_reg_loss", self.latent_reg_loss, iter)

    def forward(self, input, z=None):
        self.logs = list()

        # Parse model input.
        instance_idcs = input["instance_idx"].long().cuda()
        pose = input["pose"].cuda()
        intrinsics = input["intrinsics"].cuda()
        uv = input["uv"].cuda().float()

        batch_size, num_samples = uv.shape[:2]

        self.z = self.latent_codes(instance_idcs)

        phi = self.hyper_phi(self.z)

        if not self.counter and self.training:
            print(phi)

        points_xyz, depth_maps, log = self.ray_marcher(cam2world=pose,
                                                       intrinsics=intrinsics,
                                                       uv=uv,
                                                       phi=phi)
        self.logs.extend(log)

        v = phi(points_xyz)
        novel_views = self.pixel_generator(v)

        sidelen = int(np.sqrt(num_samples))
        class_gen_input = v.permute(0,2,1).view(batch_size, self.num_hidden_units_phi, sidelen, sidelen)
        novel_views_seg = self.class_generator(class_gen_input)
        novel_views_seg = novel_views_seg.view(batch_size, NUM_CLASSES, -1).permute(0,2,1)

        # Calculate normal map
        with torch.no_grad():
            batch_size = uv.shape[0]
            x_cam = uv[:, :, 0].view(batch_size, -1)
            y_cam = uv[:, :, 1].view(batch_size, -1)
            z_cam = depth_maps.view(batch_size, -1)

            normals = geometry.compute_normal_map(x_img=x_cam, y_img=y_cam, z=z_cam, intrinsics=intrinsics)
            self.logs.append(('image', 'normals',
                              torchvision.utils.make_grid(normals, scale_each=True, normalize=True), 100))

        self.logs.append(('embedding', '', self.latent_codes.weight, 500))
        self.logs.append(('scalar', 'embed_min', self.z.min(), 1))
        self.logs.append(('scalar', 'embed_max', self.z.max(), 1))

        if self.training:
            self.counter += 1

        return {'rgb':novel_views, 'seg':novel_views_seg, 'depth':depth_maps}

