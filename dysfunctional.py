import vapoursynth as vs
core = vs.core

def STGraino(clip, planes: [0,1,2],
            pre = [1,2,2], 
            str: float = [1,4,4], 
            show_diff = False,
            **kwargs) -> vs.VideoNode:
    """
    KNLMeansCL documentation can be found here: https://github.com/Khanattila/KNLMeansCL/wiki/Filter-description
    STPresso: https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/master/havsfunc.py#L3535
    STPresso args can be parsed directly into this wrapper, i.e;
    grn = dsf.STGraino(b16, planes=[0,1,2], pre=[1,2,2], str = [1.5,4,4], limit = 4, bias = 65, tbias = 50, tthr = 32, show_diff=True)
    -
    TODO: parse knlm args, per-plane show_diff
    """
    from vsutil import split, join, depth as Depth
    from havsfunc import STPresso
    from rgvs import RemoveGrain

    rgb = RGBtoYUV(clip)
    b16 = Depth(rgb, 16)
    p = split(b16)

    if 0 in planes:
        if pre[0] == 1:
            p[0] = p[0].knlm.KNLMeansCL(h=str[0])
        elif pre[0] == 2:
            p[0] = RemoveGrain(p[0], mode=str[0])
    if 1 in planes:
        if pre[0] == 1:
            p[1] = p[1].knlm.KNLMeansCL(h=str[1])
        elif pre[0] == 2:
            p[1] = RemoveGrain(p[1], mode=str[1])
    if 2 in planes:
        if pre[0] == 1:
            p[2] = p[2].knlm.KNLMeansCL(h=str[2])
        elif pre[0] == 2:
            p[2] = RemoveGrain(p[2], mode=str[2])

    dns = join(p)
    dif = core.std.MakeDiff(b16, dns, planes=planes)
    rgn = STPresso(dif, **kwargs, planes=planes)
    mrg = core.std.MergeDiff(dns, rgn, planes=planes)

    if show_diff is True:
        return dif
    else: 
        return Depth(mrg, clip.format.bits_per_sample)


def RGBtoYUV(clip):
    if clip.format.color_family == vs.RGB:
        return core.resize.Point(clip, format=vs.YUV444P16, matrix_s='709', dither_type='error_diffusion')
    else:
        return clip


def FDOG(clip: vs.VideoNode, retinex=True, x=2, y=2, bits=16, opencl=False) -> vs.VideoNode:
    from vsutil import depth, get_depth, get_y
    def __FDOG(clip: vs.VideoNode) -> vs.VideoNode:
        if retinex is True: bits=16
        
        lma = core.std.ShufflePlanes(depth(clip, bits), 0, vs.GRAY)
        gx = core.std.Convolution(lma, [1, 1, 0, -1, -1, 2, 2, 0, -2, -2, 3, 3, 0, -3, -3, 2, 2, 0, -2, -2, 1, 1, 0, -1, -1], divisor=x, saturate=False)
        gy = core.std.Convolution(lma, [-1, -2, -3, -2, -1, -1, -2, -3, -2, -1, 0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 1, 2, 3, 2, 1], divisor=y, saturate=False)
        expr = core.std.Expr([gx, gy], 'x dup * y dup * + sqrt')

        if bits > 16: return depth(expr, 16)
        else:
            return expr

    def __retinex_fdog(clip: vs.VideoNode, sigma=1.5, sigma_rtx=[50, 200, 350], opencl=opencl) -> vs.VideoNode:
        tcanny = core.tcanny.TCannyCL if opencl else core.tcanny.TCanny
        luma = get_y(clip)
        fdog = __FDOG(luma)
        max_value = 1 if clip.format.sample_type == vs.FLOAT else (1 << get_depth(clip)) - 1
        ret = core.retinex.MSRCP(luma, sigma=sigma_rtx, upper_thr=0.005)
        tcanny = tcanny(ret, mode=1, sigma=sigma).std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])
        return core.std.Expr([fdog, tcanny], f'x y + {max_value} min')

    if retinex is True: return __retinex_fdog(clip)
    else:
        return __FDOG(clip)