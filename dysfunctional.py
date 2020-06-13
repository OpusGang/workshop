import vapoursynth as vs
core = vs.core

def STGraino(clip, pre=None, str=1, show_diff=False, **kwargs) -> vs.VideoNode:
    """
    KNLMeansCL documentation can be found here: https://github.com/Khanattila/KNLMeansCL/wiki/Filter-description
    STPresso: https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/master/havsfunc.py#L3535
    STPresso args can be parsed directly into this wrapper, i.e;
    tst = dsf.STGraino(src, pre = 'knl', str = 2, limit = 6, bias = 70, tbias = 55, tthr = 32)
    -
    STPresso is being fed a GRAY16 clip, don't expect anything on [1,2].
    This is getting very dumb very fast
    TODO: Handle Y/Cb/Cr
    """
    from vsutil import depth as Depth
    from havsfunc import STPresso

    rgb = RGBtoYUV(clip)
    b16 = Depth(rgb, 16)
    lma = core.std.ShufflePlanes(b16, 0, vs.GRAY)

    if pre == 'knl':
        denoise = core.knlm.KNLMeansCL(lma, h=str, d=2, device_type='gpu', device_id=0, channels='Y')
    elif pre == 'cdg':
        from cooldegrain import CoolDegrain
        denoise = CoolDegrain(lma, tr=1, thsad=str)
    elif pre is 'rgsf' or 'rgvs':
        from rgvs import RemoveGrain
        try:
            denoise = RemoveGrain(lma, mode=str)
        except AttributeError:
            denoise = core.rgvs.RemoveGrain(lma, mode=str)
    else:
        raise TypeError("STGraino: Invalid pre-filter!")

    dns = denoise
    dif = core.std.MakeDiff(lma, dns)
    rgn = STPresso(dif, **kwargs)
    mrg = core.std.MergeDiff(dns, rgn)
    shf = core.std.ShufflePlanes([mrg, b16], [0,1,2], vs.YUV)

    if show_diff is True:
        return dif
    else: 
        return Depth(shf, clip.format.bits_per_sample)


def RGBtoYUV(clip):
    if clip.format.color_family == vs.RGB:
        return core.resize.Point(clip, format=vs.YUV444P16, matrix_s='709', dither_type='error_diffusion')
    else:
        return clip
