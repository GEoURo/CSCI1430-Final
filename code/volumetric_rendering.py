import torch
import torch.nn as nn


def computeAlpha(rgb, dists):
    return 1. - torch.exp(-nn.functional.relu(rgb) * dists)


def volumeRender(raw, z_vals, rays_d, raw_noise_std: float = 0., white_bkgd: bool = False):
    """
    Accumulate the model's raw output into final pixel colors.
    The math here is to identically follow the official code release.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Direct output from the MLP model.
        z_vals: [num_rays, num_samples along ray]. Depth of each point along the ray.
        rays_d: [num_rays, 3]. Direction of each ray.
        raw_noise_std: float. The standard variation of the noise
        white_bkgd: boolean. A flag specifying whether to render white background
    Returns:
        colors: [num_rays, 3]. Estimated RGB color of a ray.
        disparity: [num_rays]. Disparity map. Inverse of depth map.
        weights_accumulated: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth: [num_rays]. Estimated distance to object.
    """
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    one_e_10 = torch.tensor([1e10], device=raw.device)

    dists = torch.cat([dists, one_e_10.expand(dists[..., :1].shape)], dim=-1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # apply sigmoid to get the RGB color
    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    alpha = computeAlpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

    # this is to produce the same effect of tf.math.cumprod in the official code release
    temp = torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1. - alpha + 1e-10], dim=-1)
    weights = alpha * torch.cumprod(temp, -1)[:, :-1]
    colors = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth = torch.sum(weights * z_vals, -1)
    disparity = 1. / torch.max(1e-10 * torch.ones_like(depth), depth / torch.sum(weights, -1))
    weights_accumulated = torch.sum(weights, -1)

    if white_bkgd:
        colors = colors + (1. - weights_accumulated[..., None])

    return colors, disparity, weights_accumulated, weights, depth
