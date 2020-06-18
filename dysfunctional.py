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