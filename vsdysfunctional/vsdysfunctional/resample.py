import vapoursynth as vs
from typing import Optional, Dict, Any
from vsutil import get_y, depth, split, join
core = vs.core


def ssimBetter(clip: vs.VideoNode, preset: int = 1080,
               width: int = None, height: int = None,
               smooth: float | vs.VideoNode = 2/3,
               chroma: bool = False,
               repair: tuple = (0, 0),
               postfilter: vs.VideoNode = core.knlm.KNLMeansCL,
               ssim_args: Dict[str, Any] = {},
               prefilter_args: Dict[str, Any] = {}) -> vs.VideoNode:
    """
    smooth: detail enhancement
    repair: reduce (dark, bright) halos. Maybe try (0.5, 0).
    """
    from awsmfunc.base import zresize
    from muvsfunc import SSIM_downsample

    noiseDown = zresize(clip, preset=preset, width=width,
                        height=height, kernel='bicubic')

    if chroma:
        clip = clip, noiseDown
    else:
        clip = [get_y(x) for x in (clip, noiseDown)]

    detailDown = SSIM_downsample(clip[0], w=noiseDown.width,
                                 h=noiseDown.height,
                                 smooth=smooth, **ssim_args)
    detailDown = depth(detailDown, noiseDown.format.bits_per_sample)

    postFilter = [postfilter(ref, **prefilter_args)
                  for ref in (clip[1], detailDown)]

    storeDiff = core.std.MakeDiff(clip[1], postFilter[0])
    mergeDiff = core.std.MergeDiff(storeDiff, postFilter[1])
    
    return mergeDiff
    #if all(x == 0 for x in repair) is False:
    #    clamp = [max(min(rep, 1.0), 0.0) for rep in repair]
#
    #    deHalo = FineDehalo(mergeDiff)
    #    # MaskedLimitFilter here??
    #    # maybe instead of FineDehalo, just use the mask and merge w weights?
    #    mergeDiff = core.std.Expr([mergeDiff, deHalo],
    #                              expr=[f'x y < x x y - {clamp[0]} \
    #                              * - x x y - {clamp[1]} * - ?'])
#
    #return core.std.ShufflePlanes([mergeDiff, noiseDown], [0, 1, 2], vs.YUV)


def ssimdown(clip: vs.VideoNode, preset: Optional[int] = None,
             repair: Optional[list[float]] = None, width: Optional[int] = None,
             height: Optional[int] = None, left: int = 0, right: int = 0,
             bottom: int = 0, top: int = 0, ar: str = 16 / 9,
             shader_path: Optional[str] = None,
             shader_str: Optional[str] = None,
             repair_fun: Optional[Dict[str, Any]] = None) -> vs.VideoNode:
    """
    ssimdownscaler wrapper to resize chroma with spline36 and optional (hopefully working) side cropping
    only works with 420 atm since 444 would probably add krigbilateral to the mix
    """
    import math

    fun: Dict[str, Any] = dict(filter_param_a=0, filter_param_b=0)
    if repair_fun is not None:
        fun |= repair_fun

    if not shader_str:
        if shader_path:
            with open(shader_path) as f:
                shader = f.read()
        else:
            import urllib.request

            shader = urllib.request.urlopen("https://gist.githubusercontent.com/igv/36508af3ffc84410fe39761d6969be10/raw/ac09db2c0664150863e85d5a4f9f0106b6443a12/SSimDownscaler.glsl")
            shader = shader.read()
    else:
        shader = shader_str

    if preset == width == height is not None:
        preset = 1080

    if preset:
        if clip.width / clip.height > ar:
            return ssimdown(clip, width=ar * preset, left=left, right=right, top=top, bottom=bottom,
                            shader_str=shader, repair=repair, repair_fun=repair_fun)
        else:
            return ssimdown(clip, height=preset, left=left, right=right, top=top, bottom=bottom,
                            shader_str=shader, repair=repair, repair_fun=repair_fun)

    if (width is None) and (height is None):
        width = clip.width
        height = clip.height
        rh = rw = 1
    elif width is None:
        rh = rw = height / (clip.height - top - bottom)
    elif height is None:
        rh = rw = width / (clip.width - left - right)
    else:
        rh = height / clip.height
        rw = width / clip.width

    orig_w = clip.width
    orig_h = clip.height

    w = round(((orig_w - left - right) * rw) / 2) * 2
    h = round(((orig_h - top - bottom) * rh) / 2) * 2

    if clip.format.subsampling_w != 1 or clip.format.subsampling_h != 1:
        raise TypeError('the input clip must be 4:2:0')
    elif orig_w == w and orig_h == h:
        # noop
        return clip

    ind = clip.format.bits_per_sample

    clip = depth(clip, 16)

    shift = .25 - .25 * orig_w / w

    y, u, v = split(clip)

    if left or right or top or bottom:
        oc = [u.width, u.height]
        y = y.std.Crop(left, right, top, bottom)
        c = [math.ceil(left / 2), math.ceil(right / 2), math.ceil(top / 2), math.ceil(bottom / 2)]
        u = u.std.Crop(c[0], c[1], c[2], c[3])
        v = v.std.Crop(c[0], c[1], c[2], c[3])
        c = 4 * [0]
        if left % 2 == 1:
            c[0] = .5
        if right % 2 == 1:
            c[1] = .5
        if top % 2 == 1:
            c[2] = .5
        if bottom % 2 == 1:
            c[3] = .5
        clip = y.resize.Point(format=vs.YUV444P16)
    else:
        c = 4 * [0]

    # noizuy: pretty sure these don't need to be set: linearize=0, sigmoidize=0)
    # NSQY: As of writing (vs-|lib)placebo is not performing Y'CbCr -> linear RGB conversion,
    # See: https://github.com/Lypheo/vs-placebo/commit/ca73796ae214f6974cee01fb50c0a56a42806c80
    # To use these paramiters, the input clip must be RGB or GRAY. Default transfer (trc) is probably not what we want.

    # // Outdated info, Linearize/Sigmoidize works on Y'CbCr for placebo.Shader but not placebo.Resample
    # // leaving here as a reference, probably an oversight

    # igv: Tuned for use with dscale=mitchell and linear-downscaling=no.
    y = clip.placebo.Shader(shader_s=shader, width=w, height=h, filter="mitchell")

    if repair is not None:
        import rgvs

        darkstr = repair[0]
        brightstr = repair[1]

        bicubic = get_y(clip).resize.Bicubic(width=w, height=h, format=y.format, **fun)
        rep = rgvs.Repair(y, bicubic, mode=20)
        # NSQY: clamp to 1.0...
        limit = core.std.Expr([y, rep],
                              expr=[f'x y < x x y - {max(min(darkstr, 1.0), 0.0)} \
                                  * - x x y - {max(min(brightstr, 1.0), 0.0)} * - ?', ''])
        y = limit

    # noizuy: I hope the shifts are correctly set
    # NSQY: This casues a huge amount of ringing on CbCr
    # don't use this on media with finely detailed chroma...
    # repair could be done after shifting
    u = u.resize.Spline36(w / 2, h / 2, src_left=shift + c[0], src_width=u.width - c[0] - c[1], src_top=c[2],
                          src_height=u.height - c[2] - c[3])
    v = v.resize.Spline36(w / 2, h / 2, src_left=shift + c[0], src_width=v.width - c[0] - c[1], src_top=c[2],
                          src_height=v.height - c[2] - c[3])

    return depth(join([y, u, v]), ind) 
