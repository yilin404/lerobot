import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from functorch import combine_state_for_ensemble

# Lifted directly from https://github.com/nicklashansen/tdmpc2
DREG_BINS = None


def soft_ce(pred, target, cfg):
    """Computes the cross entropy loss between predictions and soft targets."""
    pred = F.log_softmax(pred, dim=-1)
    target = two_hot(target, cfg)
    return -(target * pred).sum(-1, keepdim=True)


@torch.jit.script
def log_std(x, low, dif):
    return low + 0.5 * dif * (torch.tanh(x) + 1)


@torch.jit.script
def _gaussian_residual(eps, log_std):
    return -0.5 * eps.pow(2) - log_std


@torch.jit.script
def _gaussian_logprob(residual):
    return residual - 0.5 * torch.log(2 * torch.pi)


def gaussian_logprob(eps, log_std, size=None):
    """Compute Gaussian log probability."""
    residual = _gaussian_residual(eps, log_std).sum(-1, keepdim=True)
    if size is None:
        size = eps.size(-1)
    return _gaussian_logprob(residual) * size


@torch.jit.script
def _squash(pi):
    return torch.log(F.relu(1 - pi.pow(2)) + 1e-6)


def squash(mu, pi, log_pi):
    """Apply squashing function."""
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    log_pi -= _squash(pi).sum(-1, keepdim=True)
    return mu, pi, log_pi


@torch.jit.script
def symexp(x):
    """
    Symmetric exponential function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


@torch.jit.script
def symlog(x):
    """
    Symmetric logarithmic function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))


@torch.jit.script
def log_std_fn(x, low, dif):
    return low + 0.5 * dif * (torch.tanh(x) + 1)


def two_hot(x, cfg):
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symlog(x)
    x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax).squeeze(1)
    bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size).long()
    bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx.float()).unsqueeze(-1)
    soft_two_hot = torch.zeros(x.size(0), cfg.num_bins, device=x.device)

    # print("x shape:", x.shape)
    # print("bin_idx shape:", bin_idx.shape)
    # print("bin_offset shape:", bin_offset.shape)
    # print("soft_two_hot shape:", soft_two_hot.shape)

    # from IPython import embed; embed()

    soft_two_hot.scatter_(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot.scatter_(1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset)
    return soft_two_hot


def two_hot_inv(x, cfg):
    """Converts a batch of soft two-hot encoded vectors to scalars."""
    global DREG_BINS
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symexp(x)
    if DREG_BINS is None:
        DREG_BINS = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device)
    x = F.softmax(x, dim=-1)
    x = torch.sum(x * DREG_BINS, dim=-1, keepdim=True)
    return symexp(x)


class Ensemble(nn.Module):
    """
    Vectorized ensemble of modules.
    """

    def __init__(self, modules, **kwargs):
        super().__init__()
        modules = nn.ModuleList(modules)
        fn, params, _ = combine_state_for_ensemble(modules)
        self.vmap = torch.vmap(fn, in_dims=(0, 0, None), randomness="different", **kwargs)
        self.params = nn.ParameterList([nn.Parameter(p) for p in params])
        self._repr = str(modules)

    def forward(self, *args, **kwargs):
        return self.vmap(list(self.params), (), *args, **kwargs)

    def __repr__(self):
        return "Vectorized " + self._repr


class ShiftAug(nn.Module):
    """
    Random shift image augmentation.
    Adapted from https://github.com/facebookresearch/drqv2
    """

    def __init__(self, pad=3):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, _, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class PixelPreprocess(nn.Module):
    """
    Normalizes pixel observations to [-0.5, 0.5].
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div_(255.0).sub_(0.5)


class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, cfg):
        super().__init__()
        # TODO: move to config
        self.dim = 8  # cfg.simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.
    """

    def __init__(self, *args, dropout=0.0, act=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = nn.Mish(inplace=True) if act is None else act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return (
            f"NormedLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}{repr_dropout}, "
            f"act={self.act.__class__.__name__})"
        )


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.0):
    """
    Basic building block of TD-MPC2.
    MLP with LayerNorm, Mish activations, and optionally dropout.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)))
    mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*mlp)


def conv(in_shape, num_channels, act=None):
    """
    Basic convolutional encoder for TD-MPC2 with raw image observations.
    4 layers of convolution with ReLU activations, followed by a linear layer.
    """
    assert in_shape[-1] == 64  # assumes rgb observations to be 64x64
    layers = [
        ShiftAug(),
        PixelPreprocess(),
        nn.Conv2d(in_shape[0], num_channels, 7, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(num_channels, num_channels, 5, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(num_channels, num_channels, 3, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(num_channels, num_channels, 3, stride=1),
        nn.Flatten(),
    ]
    if act:
        layers.append(act)
    return nn.Sequential(*layers)


class RunningScale:
    """Running trimmed scale estimator."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._value = torch.ones(1, dtype=torch.float32, device=torch.device("cuda"))
        self._percentiles = torch.tensor([5, 95], dtype=torch.float32, device=torch.device("cuda"))

    def state_dict(self):
        return {"value": self._value, "percentiles": self._percentiles}

    def load_state_dict(self, state_dict):
        self._value.data.copy_(state_dict["value"])
        self._percentiles.data.copy_(state_dict["percentiles"])

    @property
    def value(self):
        return self._value.cpu().item()

    def _percentile(self, x):
        x_dtype, x_shape = x.dtype, x.shape
        x = x.view(x.shape[0], -1)
        in_sorted, _ = torch.sort(x, dim=0)
        positions = self._percentiles * (x.shape[0] - 1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > x.shape[0] - 1] = x.shape[0] - 1
        weight_ceiled = positions - floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        return (d0 + d1).view(-1, *x_shape[1:]).type(x_dtype)

    def update(self, x):
        percentiles = self._percentile(x.detach())
        value = torch.clamp(percentiles[1] - percentiles[0], min=1.0)
        self._value.data.lerp_(value, self.cfg.tau)

    def __call__(self, x, update=False):
        if update:
            self.update(x)
        return x * (1 / self.value)

    def __repr__(self):
        return f"RunningScale(S: {self.value})"
