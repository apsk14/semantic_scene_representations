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

NUM_CLASSES = 6

class SRNsModel(nn.Module):
    def __init__(self,
                 num_instances,
                 latent_dim,
                 tracing_steps,
                 has_params=False,
                 fit_single_srn=False,
                 use_unet_renderer=False):
        super().__init__()

        self.latent_dim = latent_dim
        self.has_params = has_params
        # self.colors = [(random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)) for i in range(NUM_CLASSES)]
        # self.colors[0] = (1,1,1)
        # self.colors = np.array(self.colors)
        self.colors = np.array([[1., 1., 1.], [0.42415977, 0.95022593, 0.88655337], [0.34309762, 0.95100353, 0.3231704],
                                [0.48631192, 0.82279855, 0.80800228], [0.27445405, 0.42794667, 0.42610895],
                                [0.53534125, 0.04302588, 0.9653457]])

        self.num_hidden_units_phi = 256
        self.phi_layers = 4  # includes the in and out layers
        self.rendering_layers = 5  # includes the in and out layers
        self.sphere_trace_steps = tracing_steps

        self.fit_single_srn = fit_single_srn

        if self.fit_single_srn:  # Fit a single scene with a single SRN (no hypernetworks)
            self.phi = pytorch_prototyping.FCBlock(hidden_ch=self.num_hidden_units_phi,
                                                   num_hidden_layers=self.phi_layers - 2,
                                                   in_features=3,
                                                   out_features=self.num_hidden_units_phi)
        else:
            # Auto-decoder: each scene instance gets its own code vector
            self.latent_codes = nn.Embedding(num_instances, latent_dim).cuda()
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

        if use_unet_renderer:
            self.pixel_generator = DeepvoxelsRenderer(nf0=32, in_channels=self.num_hidden_units_phi,
                                                      input_resolution=128, img_sidelength=128)
        else:
            self.pixel_generator = pytorch_prototyping.FCBlock(hidden_ch=self.num_hidden_units_phi,
                                                               num_hidden_layers=self.rendering_layers - 1,
                                                               in_features=self.num_hidden_units_phi,
                                                               out_features=3,
                                                               outermost_linear=True)

        self.class_generator = pytorch_prototyping.FCBlock(hidden_ch=self.num_hidden_units_phi,
                                                           num_hidden_layers=self.rendering_layers - 1,
                                                           in_features=self.num_hidden_units_phi,
                                                           out_features=NUM_CLASSES,
                                                           outermost_linear=True)
        self.counter = 0

        # Losses
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')

        # List of logs
        self.logs = list()

        # print("*" * 100)
        # print(self)
        # print("*" * 100)
        # print("Number of parameters:")
        # util.print_network(self)
        # print("*" * 100)

    def get_output_img(self, prediction):
        pred_imgs, _, _, _, _ = prediction
        return util.lin2img(pred_imgs)

    def forward(self, input, z=None):
        self.logs = list()

        # Parse model input.
        observation = Observation(*input)
        instance_idcs = observation.instance_idx.long().cuda()
        pose = observation.pose.cuda()
        intrinsics = observation.intrinsics.cuda()
        uv = observation.uv.cuda().float()
        pts = observation.pts.cuda()



        if self.fit_single_srn:
            phi = self.phi
        else:
            if self.has_params:
                if z is None:
                    self.z = observation.param.cuda()
                else:
                    self.z = z
            else:
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
        p = phi(pts)

        novel_views = self.pixel_generator(v)
        novel_views_seg = self.class_generator(v)
        point_cloud = self.class_generator(p)



        # Calculate normal map
        with torch.no_grad():
            batch_size = uv.shape[0]
            x_cam = uv[:, :, 0].view(batch_size, -1)
            y_cam = uv[:, :, 1].view(batch_size, -1)
            z_cam = depth_maps.view(batch_size, -1)

            normals = geometry.compute_normal_map(x_img=x_cam, y_img=y_cam, z=z_cam, intrinsics=intrinsics)
            self.logs.append(('image', 'normals',
                              torchvision.utils.make_grid(normals, scale_each=True, normalize=True), 100))

        if not self.fit_single_srn:
            self.logs.append(('embedding', '', self.latent_codes.weight, 500))
            self.logs.append(('scalar', 'embed_min', self.z.min(), 1))
            self.logs.append(('scalar', 'embed_max', self.z.max(), 1))

        if self.training:
            self.counter += 1

        return novel_views, novel_views_seg, depth_maps, point_cloud, points_xyz



class UnetModel(nn.Module):
    def __init__(self,
                 num_instances,
                 latent_dim,
                 tracing_steps,
                 has_params=False,
                 fit_single_srn=False,
                 use_unet_renderer=False):
        super().__init__()


        self.unet = pytorch_prototyping.Unet(
                 in_channels=3,
                 out_channels=NUM_CLASSES,
                 nf0=64,
                 num_down=7,
                 max_channels=512,
                 use_dropout=True,
                 upsampling_mode='transpose',
                 dropout_prob=0.1,
                 norm=nn.BatchNorm2d,
                 outermost_linear=True)


        # self.colors = [(random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)) for i in range(NUM_CLASSES)]
        # self.colors[0] = (1,1,1)
        # self.colors = np.array(self.colors)
        self.colors = np.array([[1., 1., 1.], [0.42415977, 0.95022593, 0.88655337], [0.34309762, 0.95100353, 0.3231704],
                                [0.48631192, 0.82279855, 0.80800228], [0.27445405, 0.42794667, 0.42610895],
                                [0.53534125, 0.04302588, 0.9653457]])


        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')

        # List of logs
        self.logs = list()

        # print("*" * 100)
        # print(self)
        # print("*" * 100)
        # print("Number of parameters:")
        # util.print_network(self)
        # print("*" * 100)



    def get_seg_loss(self, prediction, ground_truth):
        '''Computes loss on predicted image (L_{img} in eq. 6 in paper)

        :param prediction (tuple): Output of forward pass.
        :param ground_truth: Ground-truth (unused).
        :return: image reconstruction loss.
        '''
        pred_segs = prediction
        #pred_segs = pred_segs.permute(0,2,3,1)
        _, trgt_segs, _ = ground_truth
        trgt_segs = trgt_segs.squeeze().long()
        #trgt_segs = trgt_segs.permute(0, 2, 1).squeeze().long()
        trgt_segs = trgt_segs.cuda()


        loss = self.cross_entropy_loss(pred_segs, trgt_segs)
        return loss


    def get_img_IOU_loss(self, prediction, ground_truth):

        pred_segs = prediction
        _, trgt_segs, _ = ground_truth
        trgt_segs = trgt_segs.squeeze().long()
        trgt_segs = trgt_segs.cuda()

        preds = F.softmax(pred_segs, dim=1)
        n = preds.shape[0]
        numClasses = preds.shape[1]

        labels_onehot = self.toOneHot(trgt_segs, numClasses)

        per_pix_intersection = preds*labels_onehot
        intersection = per_pix_intersection.view(n, numClasses, -1).sum(2)
        union = (preds + labels_onehot - per_pix_intersection)
        union = union.view(n, numClasses,-1).sum(2)

        loss = intersection/union

        return -loss.mean()


    def get_latent_loss(self):
        '''Computes loss on latent code vectors (L_{latent} in eq. 6 in paper)
        :return: Latent loss.
        '''
        if self.fit_single_srn:
            self.latent_reg_loss = 0
        else:
            self.latent_reg_loss = torch.mean(self.z ** 2)

        return self.latent_reg_loss

    def get_psnr(self, prediction, ground_truth):
        '''Compute PSNR of model image predictions.

        :param prediction: Return value of forward pass.
        :param ground_truth: Ground truth.
        :return: (psnr, ssim): tuple of floats
        '''
        pred_imgs, _, _, _, _ = prediction
        trgt_imgs, _, _ = ground_truth

        trgt_imgs = trgt_imgs.cuda()
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

    def get_comparisons(self, model_input, prediction, ground_truth=None):
        predictions, seg_pred, depth_maps, point_cloud = prediction

        batch_size = predictions.shape[0]

        observation = Observation(*model_input)

        # Parse model input.
        intrinsics = observation.intrinsics.cuda()
        xy = observation.uv.cuda().float()

        x_cam = xy[:, :, 0].view(batch_size, -1)
        y_cam = xy[:, :, 1].view(batch_size, -1)
        z_cam = depth_maps.view(batch_size, -1)

        normals = geometry.compute_normal_map(x_img=x_cam, y_img=y_cam, z=z_cam, intrinsics=intrinsics)
        normals = F.pad(normals, pad=(1, 1, 1, 1), mode='constant', value=1.)

        predictions = util.lin2img(predictions)

        if ground_truth is not None:
            trgt_imgs, trgt_segs, trgt_depths = ground_truth
            trgt_imgs = util.lin2img(trgt_imgs)
            trgt_segs = util.lin2img(trgt_segs)

            return torch.cat((normals.cpu(), predictions.cpu(), trgt_imgs.cpu()), dim=3).numpy()
        else:
            return torch.cat((normals.cpu(), predictions.cpu()), dim=3).numpy()


    def toOneHot(self, input, numClasses):
        n, h, w = input.shape
        one_hot = torch.zeros(n, numClasses, h, w).scatter_(1, input.view(n, 1, h, w).cpu(), 1).cuda()
        return one_hot


    def get_output_img(self, prediction):
        pred_imgs, _, _, _, _ = prediction
        return util.lin2img(pred_imgs)


    def get_output_seg(self, prediction, colors):
        nc = prediction.shape[1]
        prediction = torch.reshape(prediction, (1, prediction.shape[1], prediction.shape[2] * prediction.shape[2]))
        prediction = prediction.permute(0, 2, 1)
        if nc > 1:
            pred_seg, prediction = torch.max(prediction, dim=2)
            prediction = prediction[:, :, None]
        output_seg = util.lin2img(prediction)
        output_seg = (colors[output_seg.cpu().numpy()])[:, 0, :, :].transpose(0, 3, 1, 2)
        return output_seg

    def get_output_pc(self, prediction, input, out_fn, real_fn):
        observation = Observation(*input)
        pts = observation.pts.cpu().numpy().squeeze()
        labels = observation.labels.cpu().numpy().squeeze()
        _, _, _, pc_classes, _ = prediction
        _, pc_idx = torch.max(pc_classes, dim=2)
        label_colors = self.colors[pc_idx.cpu().numpy()].squeeze()
        label_colors_real = self.colors[labels].squeeze()
        src_pc = np.matrix(np.concatenate((pts, label_colors), axis=1))
        real_pc = np.matrix(np.concatenate((pts, label_colors_real), axis=1))
        with open(out_fn, 'wb') as f:
            for line in src_pc:
                np.savetxt(f, line)

        with open(real_fn, 'wb') as f:
            for line in real_pc:
                np.savetxt(f, line)



    def get_IOU_vals(self, prediction, input, confusion, part_intersect, part_union): # had arg confusion
        # confusion vector is [true pos, false pos, false neg]
        observation = Observation(*input)
        real_label = observation.labels.cpu().numpy().squeeze()
        _, pred_labels = torch.max(prediction, dim=2)
        pred_labels = pred_labels.cpu().numpy().squeeze()

        # pred_labels = np.delete(pred_labels, np.where(real_label == 0), axis=0)
        # real_label = np.delete(real_label, np.where(real_label == 0), axis=0)

        # num_classes = np.max(real_label) + 1
        num_classes = NUM_CLASSES
        true_pos = np.zeros((num_classes, 1), dtype=int)
        false_pos = np.zeros((num_classes, 1), dtype=int)
        false_neg = np.zeros((num_classes, 1), dtype=int)
        cur_shape_iou_tot = 0.0
        cur_shape_iou_cnt = 0
        for cur_class in range(1, num_classes + 1):

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
        return cur_shape_iou_tot / cur_shape_iou_cnt



    def calc_mIOU(self, confusion):
        check_empty = np.sum(confusion, axis=1)
        empty = np.where(check_empty == 0)
        confusion = np.delete(confusion, empty, axis=0)
        num_classes = confusion.shape[0]
        #print(confusion)
        true_pos = confusion[:, 0]
        false_pos = confusion[:, 1]
        false_neg = confusion[:, 2]
        IOU = np.divide(true_pos, (false_pos + false_neg + true_pos),  where=(false_pos + false_neg + true_pos) != 0)
        mIOU = np.sum(IOU, axis=0)/num_classes
        return mIOU

    def get_pc(self, pts_real, label_real, pts_pred, label_pred, iter):
        print('outputting point cloud')
        out_fn = '/home/apsk14/data/srn_point_clouds_2/%s.txt' % str(iter)
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)
        real_fn = '/home/apsk14/data/srn_point_clouds_2/real_%s.txt' % str(iter)
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)


        _, pts_labels = torch.max(label_pred, dim=2)
        label_colors = self.colors[pts_labels.cpu().numpy()]
        real_colors = self.colors[label_real.cpu().numpy()]
        labels_chosen = label_colors[0, :, :]
        real_chosen = real_colors[0, :, :]

        pts_real = pts_real.cpu().numpy()
        pts_pred = pts_pred.cpu().numpy()
        pts_real_chosen = pts_real[0,:,:]
        pts_pred_chosen = pts_pred[0,:,:]

        src_pc = np.matrix(np.concatenate((pts_pred_chosen, np.ones_like(pts_pred_chosen)), axis=1))
        real_pc = np.matrix(np.concatenate((pts_real_chosen, real_chosen), axis=1))
        with open(out_fn, 'wb') as f:
            for line in src_pc:
                np.savetxt(f, line)

        with open(real_fn, 'wb') as f:
            for line in real_pc:
                np.savetxt(f, line)

    def write_updates(self, writer, input, predictions, ground_truth, iter, prefix=''):
        '''Writes tensorboard summaries using tensorboardx api.

        :param writer: tensorboardx writer object.
        :param predictions: Output of forward pass.
        :param ground_truth: Ground truth.
        :param iter: Iteration number.
        :param prefix: Every summary will be prefixed with this string.
        '''
        seg_preds = predictions
        _, trgt_segs, _ = ground_truth

        trgt_segs = trgt_segs.cuda()
        colors = self.colors

        seg_preds, seg_idx = torch.max(seg_preds, dim=1)
        seg_idx = seg_idx[:,None,:,:]
        #seg_idx = seg_idx[:,:,None]

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
            # output_vs_gt = torch.cat((predictions, trgt_imgs), dim=0)
            # output_vs_gt = util.lin2img(output_vs_gt)
            # print('MACKSi', torch.max(output_vs_gt))
            # print('DISPRGB', output_vs_gt.shape)
            # writer.add_image(prefix + "Output_vs_gt",
            #                  torchvision.utils.make_grid(output_vs_gt,
            #                                              scale_each=False,
            #                                              normalize=True).cpu().detach().numpy(),
            #                  iter)

            # Segmentation image outputs
            output_vs_gt_seg = torch.cat((seg_idx.int(), trgt_segs.int()), dim=0)
            #output_vs_gt_seg = util.lin2img(output_vs_gt_seg).int()
            output_vs_gt_seg = torch.from_numpy(colors[output_vs_gt_seg.cpu().numpy()].squeeze()).permute(0,3,1,2)
            writer.add_image(prefix + "Output_vs_gt_seg",
                             torchvision.utils.make_grid(output_vs_gt_seg,
                                                         scale_each=False,
                                                         normalize=False).cpu().detach().numpy(),
                             iter)


            # rgb_loss = ((predictions.float().cuda() - trgt_imgs.float().cuda()) ** 2).mean(dim=2, keepdim=True)
            # rgb_loss = util.lin2img(rgb_loss)

            # fig = util.show_images([rgb_loss[i].detach().cpu().numpy().squeeze()
            #                         for i in range(batch_size)])
            # writer.add_figure(prefix + 'rgb_error_fig',
            #                   fig,
            #                   iter,
            #                   close=True)
            #
            # # depth_maps[trgt_depths==1] = 0.
            # depth_maps_plot = util.lin2img(depth_maps)
            # writer.add_image(prefix + "pred_depth",
            #                  torchvision.utils.make_grid(depth_maps_plot.repeat(1, 3, 1, 1),
            #                                              scale_each=True,
            #                                              normalize=True).cpu().detach().numpy(),
            #                  iter)
            #
            # depth_loss = (depth_maps.float().cuda() - trgt_depths.float().cuda()) ** 2
            # depth_loss[trgt_depths == 1] = 0.
            # depth_loss = util.lin2img(depth_loss)
            #
            # if np.any(depth_loss[0].detach().cpu().numpy() != 0.):
            #     fig = util.show_images([depth_loss[i].detach().cpu().numpy().squeeze()
            #                             for i in range(batch_size)])
            #     writer.add_figure(prefix + 'depth_error_fig',
            #                       fig,
            #                       iter,
            #                       close=True)

            #self.get_pc(pts_real, label_real, pts_pred, label_pred, iter)

        # depth_loss = (depth_maps.float().cuda() - trgt_depths.float().cuda()) ** 2
        # depth_loss[trgt_depths == 1] = 0.
        # if np.any(depth_loss[0].detach().cpu().numpy() != 0.):
        #     writer.add_scalar(prefix + "depth_error", depth_loss.mean(), iter)
        #
        # writer.add_scalar(prefix + "out_min", predictions.min(), iter)
        # writer.add_scalar(prefix + "out_max", predictions.max(), iter)
        #
        # writer.add_scalar(prefix + "trgt_min", trgt_imgs.min(), iter)
        # writer.add_scalar(prefix + "trgt_max", trgt_imgs.max(), iter)
        #
        # if iter:
        #     writer.add_scalar(prefix + "latent_reg_loss", self.latent_reg_loss, iter)

    def forward(self, input, z=None):
        self.logs = list()

        novel_views_seg = self.unet(input)


        return novel_views_seg
