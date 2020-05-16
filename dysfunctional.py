import vapoursynth as vs
from vapoursynth import core

def STGrain(clip, **kwargs) -> vs.VideoNode:
    """
    STPresso is being fed a GRAY16 clip, don't expect anything on [1,2].
    TODO: Handle Y/Cb/Cr
    """
    from fvsfunc import Depth
    from havsfunc import STPresso

    if clip.format.color_family == vs.RGB:
        clip = core.resize.Point(clip, format=vs.YUV420P16, matrix_s='709')

    b16 = Depth(clip, 16)
    lma = core.std.ShufflePlanes(b16, 0, vs.GRAY)
    dns = core.knlm.KNLMeansCL(lma, h = 2)
    dif = core.std.MakeDiff(lma, dns)
    rgn = STPresso(dif, **kwargs)
    mrg = core.std.MergeDiff(dns, rgn)
    shf = core.std.ShufflePlanes([mrg, b16], [0,1,2], vs.YUV)
    return Depth(shf, clip.format.bits_per_sample)