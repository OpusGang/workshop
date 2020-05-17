import vapoursynth as vs
core = vs.core

def STGrain(clip, knl=1, **kwargs) -> vs.VideoNode:
    """
    This can be either imported into the script or defined manually,
    if you would like to make changes (such as using a different denoiser) I suggest the latter
    I may add the ability to define a custom denoiser later.
    KNLMeansCL documentation can be found here: https://github.com/Khanattila/KNLMeansCL/wiki/Filter-description
    STPresso: https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/master/havsfunc.py#L3535
    STPresso args can be parsed directly into this wrapper, i.e;
    tst = dsf.STGrain(src, knl = 2, limit = 6, bias = 70, tbias = 55, tthr = 32)
    -
    STPresso is being fed a GRAY16 clip, don't expect anything on [1,2].
    TODO: Handle Y/Cb/Cr
    """
    from fvsfunc import Depth
    from havsfunc import STPresso

    if clip.format.color_family == vs.RGB:
        clip = core.resize.Point(clip, format=vs.YUV420P16, matrix_s='709')

    b16 = Depth(clip, 16)
    lma = core.std.ShufflePlanes(b16, 0, vs.GRAY)
    dns = core.knlm.KNLMeansCL(lma, h=knl, d=2, device_type='gpu', device_id=0, channels='Y')
    dif = core.std.MakeDiff(lma, dns)
    rgn = STPresso(dif, **kwargs)
    mrg = core.std.MergeDiff(dns, rgn)
    shf = core.std.ShufflePlanes([mrg, b16], [0,1,2], vs.YUV)
    return Depth(shf, clip.format.bits_per_sample)