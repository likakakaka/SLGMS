import torch


# pytorch cross scan =============
class Cross_Recombination_Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        left_half, right_half = torch.chunk(x, 2, dim=3)
        right_half = torch.flip(right_half, dims=[3])
        top_half, bottom_half = torch.chunk(x.transpose(dim0=2, dim1=3), 2, dim=2)
        bottom_half = torch.flip(bottom_half, dims=[2])
        if left_half.shape[-1] == right_half.shape[-1]:
            output_image1 = torch.stack(
                [torch.stack([left_half[:, :, :, j], right_half[:, :, :, j]], dim=3) for j in
                 range(left_half.size(-1))],
                dim=3).reshape(x.shape)

            output_image2 = torch.stack(
                [torch.stack([top_half[:, :, j, :], bottom_half[:, :, j, :]], dim=2) for j in range(top_half.size(2))],
                dim=2).reshape(x.shape)
        else:
            output_image1 = torch.stack(
                [torch.stack([left_half[:, :, :, j], right_half[:, :, :, j]], dim=3) for j in
                 range(right_half.size(-1))],
                dim=3).flatten(3, 4)
            output_image1 = torch.cat([output_image1, left_half[:, :, :, -1].unsqueeze(-1)], dim=3)
            output_image2 = torch.stack(
                [torch.stack([top_half[:, :, j, :], bottom_half[:, :, j, :]], dim=2) for j in
                 range(bottom_half.size(2))],
                dim=2).flatten(2, 3)
            output_image2 = torch.cat([output_image2, top_half[:, :, -1, :].unsqueeze(2)], dim=2)

        xs = x.new_empty((B, 4, C, H * W))
        # xs[:, 0] = x.flatten(2, 3)
        # xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 0] = output_image1.flatten(2, 3)
        xs[:, 1] = output_image2.flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        # ys0 = ys[:, 0].view(B, -1, H, W)
        # ys1 = ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous()
        if H % 2 == 0:
            ys2 = ys[:, 0].view(B, -1, H, W // 2, 2)
            left_half = ys2[..., 0]
            right_half = ys2[..., 1]
            right_half_unflipped = torch.flip(right_half, dims=[3])
            # 将左半部分和翻转回原位的右半部分拼接回原始图像
            ys2 = torch.cat([left_half, right_half_unflipped], dim=3).view(B, -1, H, W)
            ys3 = ys[:, 1].view(B, -1, H // 2, 2, W)
            top_half11 = ys3[..., 0, :]
            bottom_half11 = ys3[..., 1, :]
            bottom_half_unflipped = torch.flip(bottom_half11, dims=[2])
            # 将左半部分和翻转回原位的右半部分拼接回原始图像
            ys3 = torch.cat([top_half11, bottom_half_unflipped], dim=2).view(B, -1, W, H).transpose(dim0=2, dim1=3)
        else:
            ys2org = ys[:, 0].view(B, -1, H, W)
            ys2_1 = ys2org[..., -1].view(B, -1, H, 1)
            ys2 = ys2org[..., :-1].view(B, -1, H, W // 2, 2)
            left_half = ys2[..., 0]
            right_half = ys2[..., 1]
            right_half_unflipped = torch.flip(right_half, dims=[3])
            # 将左半部分和翻转回原位的右半部分拼接回原始图像
            ys2 = torch.cat([left_half, ys2_1, right_half_unflipped], dim=3).view(B, -1, H, W)
            ys3org = ys[:, 1].view(B, -1, H, W)
            ys3_1 = ys3org[..., -1, :].view(B, -1, 1, W)
            ys3 = ys3org[..., :-1, :].view(B, -1, H // 2, 2, W)
            top_half11 = ys3[..., 0, :]
            bottom_half11 = ys3[..., 1, :]
            bottom_half_unflipped = torch.flip(bottom_half11, dims=[2])
            # 将左半部分和翻转回原位的右半部分拼接回原始图像
            ys3 = torch.cat([top_half11, ys3_1, bottom_half_unflipped], dim=2).view(B, -1, W, H).transpose(dim0=2,
                                                                                                           dim1=3)

        y = ys2 + ys3
        return y


# pytorch cross scan =============
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class Cross_Recombination_Merge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        # out: (b, k, d, l)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, H * W)
        # ys0 = ys[:, 0].view(B, -1, H, W)
        # ys1 = ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous()
        if W % 2 == 0:
            ys2 = ys[:, 0].view(B, -1, H, W // 2, 2)
            left_half = ys2[..., 0]
            right_half = ys2[..., 1]
            right_half_unflipped = torch.flip(right_half, dims=[3])
            # 将左半部分和翻转回原位的右半部分拼接回原始图像
            ys2 = torch.cat([left_half, right_half_unflipped], dim=3).view(B, -1, H, W)
            ys3 = ys[:, 1].view(B, -1, H // 2, 2, W)
            top_half11 = ys3[..., 0, :]
            bottom_half11 = ys3[..., 1, :]
            bottom_half_unflipped = torch.flip(bottom_half11, dims=[2])
            # 将左半部分和翻转回原位的右半部分拼接回原始图像
            ys3 = torch.cat([top_half11, bottom_half_unflipped], dim=2).view(B, -1, W, H).transpose(dim0=2, dim1=3)
        else:
            ys2org = ys[:, 0].view(B, -1, H, W)
            ys2_1 = ys2org[..., -1].view(B, -1, H, 1)
            ys2 = ys2org[..., :-1].view(B, -1, H, W // 2, 2)
            left_half = ys2[..., 0]
            right_half = ys2[..., 1]
            right_half_unflipped = torch.flip(right_half, dims=[3])
            # 将左半部分和翻转回原位的右半部分拼接回原始图像
            ys2 = torch.cat([left_half, ys2_1, right_half_unflipped], dim=3).view(B, -1, H, W)
            ys3org = ys[:, 1].view(B, -1, H, W)
            ys3_1 = ys3org[..., -1, :].view(B, -1, 1, W)
            ys3 = ys3org[..., :-1, :].view(B, -1, H // 2, 2, W)
            top_half11 = ys3[..., 0, :]
            bottom_half11 = ys3[..., 1, :]
            bottom_half_unflipped = torch.flip(bottom_half11, dims=[2])
            # 将左半部分和翻转回原位的右半部分拼接回原始图像
            ys3 = torch.cat([top_half11, ys3_1, bottom_half_unflipped], dim=2).view(B, -1, W, H).transpose(dim0=2,
                                                                                                           dim1=3)

        y = (ys2 + ys3).view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        # xs[:, 0] = x
        # xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)

        left_half, right_half = torch.chunk(x.view(B, C, H, W), 2, dim=3)
        right_half = torch.flip(right_half, dims=[3])
        top_half, bottom_half = torch.chunk(x.view(B, C, H, W).transpose(dim0=2, dim1=3), 2, dim=2)
        bottom_half = torch.flip(bottom_half, dims=[2])
        if left_half.shape[-1] == right_half.shape[-1]:
            output_image1 = torch.stack(
                [torch.stack([left_half[:, :, :, j], right_half[:, :, :, j]], dim=3) for j in
                 range(left_half.size(-1))],
                dim=3).reshape(x.shape)
            output_image2 = torch.stack(
                [torch.stack([top_half[:, :, j, :], bottom_half[:, :, j, :]], dim=2) for j in range(top_half.size(2))],
                dim=2).reshape(x.shape)
        else:
            output_image1 = torch.stack(
                [torch.stack([left_half[:, :, :, j], right_half[:, :, :, j]], dim=3) for j in
                 range(right_half.size(-1))],
                dim=3).flatten(3, 4)
            output_image1 = torch.cat([output_image1, left_half[:, :, :, -1].unsqueeze(-1)], dim=3).reshape(x.shape)
            output_image2 = torch.stack(
                [torch.stack([top_half[:, :, j, :], bottom_half[:, :, j, :]], dim=2) for j in
                 range(bottom_half.size(2))],
                dim=2).flatten(2, 3)
            output_image2 = torch.cat([output_image2, top_half[:, :, -1, :].unsqueeze(2)], dim=2).reshape(x.shape)

        xs[:, 0] = output_image1
        xs[:, 1] = output_image2
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs


# these are for ablations =============
class CrossScan_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
        x = torch.cat([x, x.flip(dims=[-1])], dim=1)
        return x

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        return ys.sum(1).view(B, -1, H, W)


class CrossMerge_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        return ys.contiguous().sum(1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
        x = torch.cat([x, x.flip(dims=[-1])], dim=1)
        return x.view(B, 4, C, H, W)


class CrossScan_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
        return x

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        return ys.view(B, 4, -1, H, W).sum(1)


class CrossMerge_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, C, H, W = ys.shape
        ctx.shape = (B, C, H, W)
        return ys.view(B, 4, -1, H * W).sum(1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        B, C, H, W = ctx.shape
        return x.view(B, 1, C, H, W).repeat(1, 4, 1, 1, 1)


# import selective scan ==============================
try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...
    print(f"WARNING: can not import selective_scan_cuda_oflex.", flush=True)
    print(e, flush=True)

try:
    import selective_scan_cuda_core
except Exception as e:
    ...
    print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    print(e, flush=True)

try:
    import selective_scan_cuda
except Exception as e:
    ...
    print(f"WARNING: can not import selective_scan_cuda.", flush=True)
    print(e, flush=True)


def check_nan_inf(tag: str, x: torch.Tensor, enable=True):
    if enable:
        if torch.isinf(x).any() or torch.isnan(x).any():
            print(tag, torch.isinf(x).any(), torch.isnan(x).any(), flush=True)
            import pdb;
            pdb.set_trace()


# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


# this is only for selective_scan_ref...
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


# cross selective scan ===============================
# comment all checks if inside cross_selective_scan
class SelectiveScanMamba(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanCore(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)

        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


def selective_scan_flop_jit(inputs, outputs, flops_fn=flops_selective_scan_fn):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops