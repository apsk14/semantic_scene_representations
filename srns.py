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
                 mode='hyper',
                 renderer='fc',
                 depth_supervision=False):
        super().__init__()

        self.embedding_size = embedding_size
        self.has_params = has_params
        self.depth_supervision = depth_supervision

        self.mode = mode
        self.implicit_nf = implicit_nf

        self.phi_layers = 4 # includes the in and out layers
        self.rendering_layers = 5 # includes the in and out layers
        self.sphere_trace_steps = tracing_steps

        if mode=='hyper':
            # Auto-decoder: each scene instance gets its own code vector
            self.obj_embedding = nn.Embedding(num_objects, embedding_size).cuda()
            nn.init.normal_(self.obj_embedding.weight, mean=0, std=0.01)

            self.hyper_phi = hyperlayers.HyperFC(hyper_in_ch=self.embedding_size,
                                                 hyper_num_hidden_layers=1,
                                                 hyper_hidden_ch=self.embedding_size,
                                                 hidden_ch=implicit_nf,
                                                 num_hidden_layers=self.phi_layers-2,
                                                 in_ch=3,
                                                 out_ch=self.implicit_nf)
        elif mode == 'single': # Fit a single scene with a single SRN (no hypernetworks)
            self.phi = pytorch_prototyping.FCBlock(hidden_ch=implicit_nf,
                                                   num_hidden_layers=self.phi_layers-2,
                                                   in_features=3,
                                                   out_features=self.implicit_nf)
        else:
            raise ValueError("Unknown SRN mode")

        self.intersection_net = RaycasterNet(n_grid_feats=self.implicit_nf,
                                             raycast_steps=self.sphere_trace_steps)

        if renderer == 'fc':
            self.rendering_net = pytorch_prototyping.FCBlock(hidden_ch=self.implicit_nf,
                                                             num_hidden_layers=self.rendering_layers-1,
                                                             in_features=self.implicit_nf,
                                                             out_features=3,
                                                             outermost_linear=True)
        elif renderer == 'conv':
            self.rendering_net = DeepvoxelsRenderer(nf0=32, in_channels=implicit_nf,
                                                    input_resolution=128, img_sidelength=128)
        else:
            raise ValueError("Unknown renderer")

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
        _, depth = prediction

        neg_penalty = (torch.min(depth, torch.zeros_like(depth)) ** 2)
        return torch.mean(neg_penalty)*10000

    def get_distortion_loss(self, prediction, ground_truth):
        predictions, depths = prediction
        trgt_imgs, trgt_depths = ground_truth

        trgt_imgs = trgt_imgs.cuda()

        loss = self.l2_loss(predictions, trgt_imgs)
        return loss

    def get_variational_loss(self):
        if self.mode == 'hyper':
            self.latent_reg_loss = torch.mean(self.embedding**2)
        else:
            self.latent_reg_loss = 0

        return self.latent_reg_loss

    def get_psnr(self, prediction, ground_truth):
        predictions, depth_maps = prediction
        trgt_imgs, trgt_depths = ground_truth

        trgt_imgs = trgt_imgs.cuda()
        batch_size = predictions.shape[0]

        predictions = util.lin2img(predictions)
        trgt_imgs = util.lin2img(trgt_imgs)

        psnrs, ssims = list(), list()
        for i in range(batch_size):
            p = predictions[i,:,5:-5,5:-5].squeeze().permute(1,2,0).detach().cpu()
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
        predictions, depth_maps = prediction
        trgt_imgs, trgt_depths = ground_truth

        trgt_imgs = trgt_imgs.cuda()

        predictions = util.lin2img(predictions)
        trgt_imgs = util.lin2img(trgt_imgs)
        depth_maps = util.lin2img(depth_maps)

        gt = util.convert_image(trgt_imgs)
        pred = util.convert_image(predictions)

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


    def forward(self, input, embedding=None):
        self.logs = list()

        ray_bundle = Observation(*input)

        # Parse model input.
        obj_idcs = ray_bundle.obj_idx.long().cuda()
        trgt_pose = ray_bundle.pose.cuda()
        intrinsics = ray_bundle.intrinsics.cuda()
        xy = ray_bundle.xy.cuda().float()

        if self.mode == 'hyper':
            if self.has_params:
                if embedding is None:
                    self.embedding = ray_bundle.param.cuda()
                else:
                    self.embedding = embedding
            else:
                self.embedding = self.obj_embedding(obj_idcs)

            phi = self.hyper_phi(self.embedding)
        elif self.mode == 'single':
            phi = self.phi

        if not self.counter and self.training:
            print(phi)

        points_xyz, depth_maps, log = self.intersection_net(cam2world=trgt_pose,
                                                            intrinsics=intrinsics,
                                                            xy=xy,
                                                            feature_net=phi)
        self.logs.extend(log)

        feats = phi(points_xyz) # feats: (batch, num_samples, num_grid_feats)
        novel_views = self.rendering_net(feats)

        if self.mode == 'hyper':
            self.logs.append(('embedding', '', self.obj_embedding.weight, 500))
            self.logs.append(('scalar', 'embed_min', self.embedding.min(), 1))
            self.logs.append(('scalar', 'embed_max', self.embedding.max(), 1))

        if self.training:
            self.counter += 1

        return novel_views, depth_maps


