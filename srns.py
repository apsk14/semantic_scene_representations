import torch
import torch.nn as nn
import numpy as np
import cv2

import torchvision
from custom_layers import *
import util

from dataio import Observation
import skimage.measure, skimage.transform

from pytorch_prototyping import pytorch_prototyping
import hyperlayers


class SRNsModel(nn.Module):
    def __init__(self,
                 num_objects,
                 embedding_size,
                 implicit_nf,
                 tracing_steps,
                 has_params=False,
                 fit_single_srn=False,
                 use_unet_renderer=False,
                 depth_supervision=False):
        super().__init__()

        self.embedding_size = embedding_size
        self.has_params = has_params
        self.depth_supervision = depth_supervision

        self.implicit_nf = implicit_nf

        self.phi_layers = 4 # includes the in and out layers
        self.rendering_layers = 5 # includes the in and out layers
        self.sphere_trace_steps = tracing_steps

        self.fit_single_srn = fit_single_srn

        if self.fit_single_srn: # Fit a single scene with a single SRN (no hypernetworks)
            self.phi = pytorch_prototyping.FCBlock(hidden_ch=implicit_nf,
                                                   num_hidden_layers=self.phi_layers-2,
                                                   in_features=3,
                                                   out_features=self.implicit_nf)
        else:
            # Auto-decoder: each scene instance gets its own code vector
            self.latent_codes = nn.Embedding(num_objects, embedding_size).cuda()
            nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

            self.hyper_phi = hyperlayers.HyperFC(hyper_in_ch=self.embedding_size,
                                                 hyper_num_hidden_layers=1,
                                                 hyper_hidden_ch=self.embedding_size,
                                                 hidden_ch=implicit_nf,
                                                 num_hidden_layers=self.phi_layers-2,
                                                 in_ch=3,
                                                 out_ch=self.implicit_nf)

        self.ray_marcher = RaycasterNet(n_grid_feats=self.implicit_nf,
                                        raycast_steps=self.sphere_trace_steps)

        if use_unet_renderer:
            self.pixel_generator = DeepvoxelsRenderer(nf0=32, in_channels=implicit_nf,
                                                      input_resolution=128, img_sidelength=128)
        else:
            self.pixel_generator = pytorch_prototyping.FCBlock(hidden_ch=self.implicit_nf,
                                                               num_hidden_layers=self.rendering_layers-1,
                                                               in_features=self.implicit_nf,
                                                               out_features=3,
                                                               outermost_linear=True)

        self.counter = 0

        # Losses
        self.l2_loss = nn.MSELoss(reduction='mean')

        # List of logs
        self.logs = list()

        print("*"*100)
        print(self)
        print("*"*100)
        print("Number of parameters:")
        util.print_network(self)
        print("*"*100)

    def get_regularization_loss(self, prediction, ground_truth):
        '''Computes regularization loss on final depth map (L_{depth} in eq. 6 in paper)

        :param prediction (tuple): Output of forward pass.
        :param ground_truth: Ground-truth (unused).
        :return: Regularization loss on final depth map.
        '''
        _, depth = prediction

        neg_penalty = (torch.min(depth, torch.zeros_like(depth)) ** 2)
        return torch.mean(neg_penalty)*10000

    def get_image_loss(self, prediction, ground_truth):
        '''Computes loss on predicted image (L_{img} in eq. 6 in paper)

        :param prediction (tuple): Output of forward pass.
        :param ground_truth: Ground-truth (unused).
        :return: image reconstruction loss.
        '''
        pred_imgs, _ = prediction
        trgt_imgs, _ = ground_truth

        trgt_imgs = trgt_imgs.cuda()

        loss = self.l2_loss(pred_imgs, trgt_imgs)
        return loss

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
        pred_imgs, _ = prediction
        trgt_imgs, _ = ground_truth

        trgt_imgs = trgt_imgs.cuda()
        batch_size = pred_imgs.shape[0]

        pred_imgs = util.lin2img(pred_imgs)
        trgt_imgs = util.lin2img(trgt_imgs)

        psnrs, ssims = list(), list()
        for i in range(batch_size):
            p = pred_imgs[i,:,5:-5,5:-5].squeeze().permute(1,2,0).detach().cpu()
            trgt = trgt_imgs[i,:,5:-5,5:-5].squeeze().permute(1,2,0).detach().cpu()

            p /= 2.
            p += 0.5
            p = torch.clamp(p, 0., 1.)

            trgt /= 2.
            trgt += 0.5

            p = p.numpy()
            trgt = trgt.numpy()

            ssim = skimage.measure.compare_ssim(p, trgt, multichannel=True, data_range=1)
            psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)

            psnrs.append(psnr)
            ssims.append(ssim)

        return np.mean(psnrs), np.mean(ssims)

    def write_comparison(self, prediction, ground_truth, path, plot_ground_truth=False):
        pred_imgs, pred_depth_maps = prediction
        trgt_imgs, _ = ground_truth

        trgt_imgs = trgt_imgs.cuda()

        pred_imgs = util.lin2img(pred_imgs)
        trgt_imgs = util.lin2img(trgt_imgs)
        depth_maps = util.lin2img(pred_depth_maps)

        gt = util.convert_image(trgt_imgs)
        pred = util.convert_image(pred_imgs)

        depth_img = depth_maps.squeeze()[:,:,None].cpu().numpy()
        depth_img = (depth_img - np.amin(depth_img)) / (np.amax(depth_img) - np.amin(depth_img))
        depth_img *= 2**8 - 1
        depth_img = depth_img.round()
        depth_img = cv2.applyColorMap(depth_img.astype(np.uint8), cv2.COLORMAP_JET)
        depth_img = (depth_img.astype(np.float32) / (2**8-1)) * (2**16 -1)

        if plot_ground_truth:
            output = np.concatenate((depth_img, pred, gt), axis=1)
        else:
            output = np.concatenate((depth_img, pred), axis=1)
        util.write_img(output, path)

    def write_eval(self, prediction, ground_truth, path):
        predictions, depth_maps = prediction
        predictions = util.lin2img(predictions)
        pred = util.convert_image(predictions)
        util.write_img(pred, path)

    def write_updates(self, writer, predictions, ground_truth, iter, prefix=''):
        predictions, depth_maps = predictions
        trgt_imgs, trgt_depths = ground_truth

        trgt_imgs = trgt_imgs.cuda()
        trgt_depths = trgt_depths.cuda()

        batch_size, num_samples, _ = predictions.shape

        # Module's own log
        for type, name, content, every_n in self.logs:
            name = prefix + name

            if not iter % every_n:
                if type=='image':
                    writer.add_image(name, content.detach().cpu().numpy(), iter)
                    writer.add_scalar(name + '_min', content.min(), iter)
                    writer.add_scalar(name + '_max', content.max(), iter)
                elif type=='figure':
                    writer.add_figure(name, content, iter, close=True)
                elif type=='histogram':
                    writer.add_histogram(name, content.detach().cpu().numpy(), iter)
                elif type=='scalar':
                    writer.add_scalar(name, content.detach().cpu().numpy(), iter)
                elif type=='embedding':
                    writer.add_embedding(mat=content, global_step=iter)

        if not iter % 100:
            output_vs_gt = torch.cat((predictions, trgt_imgs), dim=0)
            output_vs_gt = util.lin2img(output_vs_gt)
            writer.add_image(prefix + "Output_vs_gt",
                             torchvision.utils.make_grid(output_vs_gt,
                                                         scale_each=False,
                                                         normalize=True).cpu().detach().numpy(),
                             iter)

            rgb_loss = ((predictions.float().cuda() - trgt_imgs.float().cuda())**2).mean(dim=2, keepdim=True)
            rgb_loss = util.lin2img(rgb_loss)

            fig = util.show_images([rgb_loss[i].detach().cpu().numpy().squeeze()
                                    for i in range(batch_size)])
            writer.add_figure(prefix + 'rgb_error_fig',
                              fig,
                              iter,
                              close=True)

            # depth_maps[trgt_depths==1] = 0.
            depth_maps_plot = util.lin2img(depth_maps)
            writer.add_image(prefix + "pred_depth",
                             torchvision.utils.make_grid(depth_maps_plot.repeat(1,3,1,1),
                                                         scale_each=True,
                                                         normalize=True).cpu().detach().numpy(),
                             iter)


            depth_loss = (depth_maps.float().cuda() - trgt_depths.float().cuda())**2
            depth_loss[trgt_depths==1] = 0.
            depth_loss = util.lin2img(depth_loss)

            if np.any(depth_loss[0].detach().cpu().numpy()!=0.):
                fig = util.show_images([depth_loss[i].detach().cpu().numpy().squeeze()
                                        for i in range(batch_size)])
                writer.add_figure(prefix + 'depth_error_fig',
                                  fig,
                                  iter,
                                  close=True)


        depth_loss = (depth_maps.float().cuda() - trgt_depths.float().cuda())**2
        depth_loss[trgt_depths==1] = 0.
        if np.any(depth_loss[0].detach().cpu().numpy()!=0.):
            writer.add_scalar(prefix + "depth_error", depth_loss.mean(), iter)

        writer.add_scalar(prefix + "out_min", predictions.min(), iter)
        writer.add_scalar(prefix + "out_max", predictions.max(), iter)

        writer.add_scalar(prefix + "trgt_min", trgt_imgs.min(), iter)
        writer.add_scalar(prefix + "trgt_max", trgt_imgs.max(), iter)

        if iter:
            writer.add_scalar(prefix + "latent_reg_loss", self.latent_reg_loss, iter)


    def forward(self, input, z=None):
        self.logs = list()

        # Parse model input.
        observation = Observation(*input)
        instance_idcs = observation.instance_idx.long().cuda()
        pose = observation.pose.cuda()
        intrinsics = observation.intrinsics.cuda()
        xy = observation.xy.cuda().float()

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
                                                       xy=xy,
                                                       feature_net=phi)
        self.logs.extend(log)

        v = phi(points_xyz)
        novel_views = self.pixel_generator(v)

        if self.mode == 'hyper':
            self.logs.append(('embedding', '', self.latent_codes.weight, 500))
            self.logs.append(('scalar', 'embed_min', self.z.min(), 1))
            self.logs.append(('scalar', 'embed_max', self.z.max(), 1))

        if self.training:
            self.counter += 1

        return novel_views, depth_maps


