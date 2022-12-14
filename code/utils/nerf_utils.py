import torch
import numpy as np


def MSE_Loss(x, y):
    return torch.mean((x - y) ** 2)


def psnr_from_loss(x):
    ten = torch.Tensor([10.]).to(x)
    return -10. * torch.log(x) / torch.log(ten)


def toRGBA_UInt8(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


# Ray helpers
def generate_rays(h, w, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, w - 1, w, device=c2w.device),
                          torch.linspace(0, h - 1, h, device=c2w.device))
    # pytorch's mesh grid has indexing='ij', so we have to transpose
    i = i.t()
    j = j.t()

    # The camera is looking into the -1 direction of z-axis in camera space, x and y depends on pixel indices
    dirs = torch.stack([(i - w * 0.5) / focal, -(j - h * 0.5) / focal, -torch.ones_like(i)], -1)

    # transform the direction vector into world space
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)

    # the last column of the camera view matrix is indeed the camera's world space position,
    # which is the origin of all rays
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


# This function is derived from https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py
# to perfectly follow its math
def ndc_rays(h, w, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (w / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (h / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (w / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (h / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
# This function is derived from https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py
# to perfectly follow its math
def sample_pdf(bins, weights, n_samples, det=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=n_samples).to(weights)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
