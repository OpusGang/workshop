from functools import partial
from typing import Any, Callable, Dict, Optional

import vapoursynth as vs
from awsmfunc.base import zresize
from muvsfunc import SSIM_downsample
from vsutil import depth, get_y, join, split

core = vs.core


def ssimBetter(clip: vs.VideoNode, preset: int = 1080,
               width: int = None, height: int = None,
               smooth: float | vs.VideoNode = 1/3,
               chroma: bool = False,
               repair: tuple = (0, 0),
               postfilter: Callable[[vs.VideoNode], vs.VideoNode] = None,
               ssim_args: Dict[str, Any] = {},
               ) -> vs.VideoNode:

    bits = clip.format.bits_per_sample
    
    if bits < 16:
        clip = depth(clip, 16)
    
    if postfilter is None:
        postfilter = partial(core.dfttest.DFTTest, tbsize=1, sigma=20)
    
    noiseDown = zresize(clip, preset=preset, width=width, height=height,
                        kernel='lanczos', filter_param_b=2)

    if chroma:
        clip = clip, noiseDown
    else:
        clip = [get_y(x) for x in (clip, noiseDown)]

    detailDown = ssim_downsample(clip[0], width=noiseDown.width,
                                 height=noiseDown.height,
                                 smooth=smooth, **ssim_args)
    detailDown = depth(detailDown, noiseDown.format.bits_per_sample)

    postFilter = [postfilter(ref)
                  for ref in (clip[1], detailDown)]

    storeDiff = core.std.MakeDiff(clip[1], postFilter[0])
    mergeDiff = core.std.MergeDiff(storeDiff, postFilter[1])

    if all(x == 0 for x in repair) is False:
        clamp = [max(min(rep, 1.0), 0.0) for rep in repair]
        # basic limiter taken from dehalo_alpha
        mergeDiff = core.std.Expr([mergeDiff, noiseDown],
                                  expr=[f'x y < x x y - {clamp[0]} \
                                  * - x x y - {clamp[1]} * - ?'])

    if chroma:
        return depth(mergeDiff, bits)
    
    shuffle = core.std.ShufflePlanes(
        [mergeDiff, noiseDown], [0, 1, 2], vs.YUV
        )
    
    return depth(shuffle, bits)


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

    fun: Dict[str, Any] = dict(filter_param_a=2)
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
    y = clip.placebo.Shader(shader_s=shader, width=w, height=h, filter="mitchell", linearize=False)

    if repair is not None:
        import rgvs

        def _clamp(val: float, min_val: float, max_val: float) -> float:
            return min_val if val < min_val else max_val if val > max_val else val

        darkstr = _clamp(repair[0], min_val=0, max_val=1.0)
        brightstr = _clamp(repair[1], min_val=0, max_val=1.0)

        bicubic = get_y(clip).resize.Lanczos(width=w, height=h, format=y.format, **fun)
        rep = rgvs.Repair(y, bicubic, mode=20)
        # NSQY: clamp to 1.0
        limit = core.std.Expr(
            [y, rep], expr=[f'x y < x x y - {darkstr} * - x x y - {brightstr} * - ?', '']
            )
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


def zfun(clip: vs.VideoNode,
         preset: int | None = None,
         width: int | None = None,
         height: int | None = None,
         left: int = 0,
         right: int = 0,
         top: int = 0,
         bottom: int = 0,
         ar: float = 16 / 9,
         **kwargs) -> vs.VideoNode:

    orig_w = clip.width
    orig_h = clip.height

    orig_cropped_w = orig_w - left - right
    orig_cropped_h = orig_h - top - bottom

    if preset:
        if orig_cropped_w / orig_cropped_h > ar:
            print(int(ar * preset))
        else:
            return zresize(clip, height=preset, left=left, right=right, top=top, bottom=bottom, **kwargs)

    if (width is None) and (height is None):
        width = orig_w
        height = orig_h
        rh = rw = 1
    elif width is None:
        rh = rw = height / orig_cropped_h
    elif height is None:
        rh = rw = width / orig_cropped_w
    else:
        rh = height / orig_h
        rw = width / orig_w

    w = round((orig_cropped_w * rw) / 2) * 2
    h = round((orig_cropped_h * rh) / 2) * 2

    return dict(width=w, height=h, src_left=left, src_top=top, src_width=orig_cropped_w, src_height=orig_cropped_h)

