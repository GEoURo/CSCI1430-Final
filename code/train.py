import os
import imageio
import time

# import torch.backends.mps

from tqdm import tqdm, trange
from datetime import datetime
from utils.argparser import config_parser
from utils.nerf_utils import *
from utils.load_blender import load_blender_data
from model import NeRF
from volumetric_rendering import volumeRender


def runModel(pts, view_dir, model, chunk=1024 * 64):
    pts_flatten = pts.view(pts.shape[0] * pts.shape[1], pts.shape[2])  # (N, 64, 3) -> (N * 64, 3)
    view_dir = view_dir.view(view_dir.shape[0], 1, view_dir.shape[1])  # (N, 3) -> (N, 1, 3)
    view_dir = view_dir.repeat(1, pts.shape[1], 1)  # (N, 1, 3) -> (N, 64, 3)
    view_dir = view_dir.view(view_dir.shape[0] * view_dir.shape[1], view_dir.shape[2])  # (N, 64, 3) -> (N * 64, 3)

    outputs_flat = torch.cat([model(pts_flatten[i:i + chunk], view_dir[i:i + chunk])
                              for i in range(0, pts_flatten.shape[0], chunk)],
                             dim=0)
    outputs = torch.reshape(outputs_flat, list(pts.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def render(H, W, focal, model_coarse, model_fine=None,
           ray_chunk=1024 * 32, point_chunk=1024 * 64,
           rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           **kwargs):
    """
    Render rays, this function is the entry of the whole NeRF rendering process
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      model_coarse: The coarse model with low points sampling rate, required.
      model_fine: The fine model with high points sampling rate, optional.
      ray_chunk: int. Maximum number of rays to process simultaneously.
      point_chunk: int. Maximum number of points to process simultaneously.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for each sample in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image, usually when testing
        rays_o, rays_d = generate_rays(H, W, focal, c2w)
    else:
        # use provided ray batch, usually when training
        rays_o, rays_d = rays

    # provide ray directions as input
    viewdirs = rays_d

    # normalize the direction vectors
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]

    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # reshape the ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    # prepare the whole ray batch vector by adding the near, far and view directions
    near = near * torch.ones_like(rays_d[..., :1])
    far = far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far, viewdirs], -1)

    # Render and reshape
    all_ret = {}
    for i in range(0, rays.shape[0], ray_chunk):
        ret = render_rays(rays[i:i + ray_chunk], model_coarse,
                          model_fine=model_fine, point_chunk=point_chunk, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_rays(ray_batch,
                model_coarse,
                model_fine=None,
                point_chunk=1024 * 64,
                N_samples=64,
                N_importance=0,
                perturb=0.,
                raw_noise_std=0.,
                lindisp=False,
                white_bkgd=False,
                ):
    """
    Process a batch of rays and return the results
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, near, far, normalized view direction.
      model_coarse: Model with coarse sampling rate.
      model_fine: Model with fine sampling rate.
      point_chunk: int. Number of points to process at one time.
      N_samples: int. Number of different times to sample along each ray for the coarse model.
      N_importance: int. Number of additional times to sample along each ray for the fine model.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      raw_noise_std: The std of the noise to be added to the final output.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      white_bkgd: bool. If True, assume a white background. This is set to True for blender dataset.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    # unpack the information of rays
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # obtain the sample points along the ray and adjust it based on near and far
    t_vals = torch.linspace(0., 1., steps=N_samples, device=ray_batch.device)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * t_vals
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=ray_batch.device)

        # apply linear interpolation
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    # the render pass of the coarse model
    raw = runModel(pts, viewdirs, model_coarse, point_chunk)
    rgb_map, disp_map, acc_map, weights, depth_map = volumeRender(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    ret = {}

    if N_importance > 0:
        # the render pass for the fine model
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

        # N_samples + N_importance points total along each ray
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        raw = runModel(pts, viewdirs, model_fine, point_chunk)

        rgb_map, disp_map, acc_map, weights, depth_map = volumeRender(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    ret['rgb_map'] = rgb_map
    ret['disp_map'] = disp_map
    ret['acc_map'] = acc_map

    return ret


def renderPoses(poses, hwf, render_args, model_coarse, model_fine,
                ray_chunk=1024 * 32, point_chunk=1024 * 64,
                saveDir=None, render_factor=0):
    H, W, focal = hwf

    if render_factor != 0:
        # Render down-sampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, focal, model_coarse, model_fine=model_fine,
                                   ray_chunk=ray_chunk, point_chunk=point_chunk,
                                   c2w=c2w[:3, :4], **render_args)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        if saveDir is not None:
            rgb8 = toRGBA_UInt8(rgbs[-1])
            filename = os.path.join(saveDir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_models(args, device):
    model_coarse = NeRF().to(device)
    parameters = list(model_coarse.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF().to(device)
        parameters += list(model_fine.parameters())

    # Create optimizer
    optimizer = torch.optim.Adam(params=parameters, lr=args.lrate, betas=(0.9, 0.999))

    train_args = {
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'N_samples': args.N_samples,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        train_args['ndc'] = False
        train_args['lindisp'] = args.lindisp

    # make a copy of the dictionary for testing purpose
    test_args = {k: train_args[k] for k in train_args}
    test_args['perturb'] = False
    test_args['raw_noise_std'] = 0.

    return model_coarse, model_fine, train_args, test_args, parameters, optimizer


def train():
    parser = config_parser()
    args = parser.parse_args()

    np.random.seed(0)

    # select different torch devices
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load data set
    if args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        # blender dataset uses a specific near and far configuration, why??
        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast to correct types for later training
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # if test only, collect the poses for rendering
    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    args.expname += datetime.now().strftime('_%H_%M_%d')

    basedir = args.basedir
    expname = args.expname

    # code reused from original code release for logging
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    model_coarse, model_fine, train_args, test_args, grad_vars, optimizer = create_models(args, device)

    appendix_dict = {
        'near': near,
        'far': far,
    }
    train_args.update(appendix_dict)
    test_args.update(appendix_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    N_rand = args.N_rand
    N_iters = args.N_iters + 1

    poses = torch.Tensor(poses).to(device)

    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    for step in trange(0, N_iters):
        # randomly select one image for the current iteration
        selected_train_i = np.random.choice(i_train)
        selected_train_image = images[selected_train_i]
        selected_train_image = torch.Tensor(selected_train_image).to(device)

        # acquire the camera view matrix
        pose = poses[selected_train_i, :3, :4]

        # get the rays based on the camera view matrix
        rays_o, rays_d = generate_rays(H, W, focal, pose)  # (H, W, 3), (H, W, 3)

        # prepare a mesh grid so that we can randomly choose a subset of rays for training
        pixels = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H, device=device),
                                            torch.linspace(0, W - 1, W, device=device)),
                             dim=-1)  # (H, W, 2)

        pixels = torch.reshape(pixels, [-1, 2])  # (H * W, 2)
        selected_indices = np.random.choice(pixels.shape[0], size=[N_rand], replace=False)  # (N_rand,)
        selected_pixels = pixels[selected_indices].long()  # (N_rand, 2)

        # filter out the randomly selected rays
        rays_o = rays_o[selected_pixels[:, 0], selected_pixels[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[selected_pixels[:, 0], selected_pixels[:, 1]]  # (N_rand, 3)
        ray_batch = torch.stack([rays_o, rays_d], 0)
        selected_test_pixels = selected_train_image[selected_pixels[:, 0], selected_pixels[:, 1]]  # (N_rand, 3)

        # render the rays
        rgb, disp, acc, extras = render(H, W, focal, model_coarse, model_fine=model_fine,
                                        ray_chunk=args.chunk, point_chunk=args.netchunk,
                                        rays=ray_batch, **train_args)

        # compute loss and back propagate
        optimizer.zero_grad()
        img_loss = MSE_Loss(rgb, selected_test_pixels)
        loss = img_loss
        psnr = psnr_from_loss(img_loss)

        if 'rgb0' in extras:
            img_loss0 = MSE_Loss(extras['rgb0'], selected_test_pixels)
            loss = loss + img_loss0
            psnr0 = psnr_from_loss(img_loss0)

        loss.backward()
        optimizer.step()

        # learning rate decay
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        # for logging, reused and referenced most of them from official code release
        if step % args.i_weights == 0 and step > 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(step))
            torch.save({
                'model_coarse_state_dict': model_coarse.state_dict(),
                'model_fine_state_dict': model_fine.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if step % args.i_video == 0 and step > 0:
            with torch.no_grad():
                rgbs, disps = renderPoses(render_poses, hwf, test_args,
                                          model_coarse, model_fine,
                                          ray_chunk=args.chunk, point_chunk=args.netchunk)
            print('Done, saving', rgbs.shape, disps.shape)
            movie_base_name = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, step))
            imageio.mimwrite(movie_base_name + 'rgb.mp4', toRGBA_UInt8(rgbs), fps=30, quality=8)

        if step % args.i_testset == 0 and step > 0:
            test_dir = os.path.join(basedir, expname, 'test_{:06d}'.format(step))
            os.makedirs(test_dir, exist_ok=True)
            with torch.no_grad():
                # only render the first 5 to save some time
                renderPoses(poses[i_test[:5]], hwf, test_args,
                            model_coarse, model_fine,
                            ray_chunk=args.chunk, point_chunk=args.netchunk,
                            saveDir=test_dir, render_factor=args.render_factor)
            print('Saved test set')

        if step % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {step} Loss: {loss.item()}  PSNR: {psnr.item()}")


if __name__ == '__main__':
    train()
