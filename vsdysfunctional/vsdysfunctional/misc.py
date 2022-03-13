from functools import partial
import vapoursynth as vs
core = vs.core

from typing import Callable, Optional, Dict, Any
from vsutil import get_y, depth, Range, scale_value, split, join


def csharp(flt: vs.VideoNode, src: vs.VideoNode,
           mode: int = 20) -> vs.VideoNode:
    """
    Stolen from Zastin
    works ok
    """
    import rgvs as RGToolsVS

    np = flt.format.num_planes
    blur = RGToolsVS.RemoveGrain(flt, mode=mode)
    return core.std.Expr([flt, src, blur],
                         ['x dup + z - x y min max x y max min', '', ''][:np])


def fastFreqMerge(lo: vs.VideoNode, hi: vs.VideoNode,
                  thresh: int = 12) -> vs.VideoNode:

    def _gauss(clip: vs.VideoNode) -> vs.VideoNode:
        gauss = core.fmtc.resample(clip, w=clip.width * 2, h=clip.height * 2,
                                   kernel='gauss', a1=100)
        return core.fmtc.resample(gauss, w=clip.width, h=clip.height,
                                  kernel='gauss', a1=thresh)

    hi_freq = core.std.MakeDiff(hi, _gauss(hi))
    return core.std.MergeDiff(_gauss(lo), hi_freq)


def retinex(clip: vs.VideoNode,
            mask: vs.VideoNode,
            fast: bool = True,
            msrcp_dict: None | dict = None,
            tcanny_dict: None | dict = None) -> vs.VideoNode:
    """
    edgeKirsch = retinex(srcPre, vsmask.edge.Kirsch().get_mask(clip),
                         fast=True, msrcp_dict=dict(op=5))
    """

    msrcp_args: Dict[str, Any] = dict(sigma=[50, 200, 350],
                                      upper_thr=0.005, fulls=False)
    if msrcp_dict is not None:
        msrcp_args |= msrcp_dict

    tcanny_args: Dict[str, Any] = dict(sigma=1, mode=1, op=2)
    if tcanny_dict is not None:
        tcanny_args |= tcanny_dict

    if clip.format.num_planes > 1:
        clip, mask = [get_y(x) for x in (clip, mask)]

    def resample(clip: vs.VideoNode,
                 function: vs.VideoNode,
                 dither_type: str = 'none',
                 input_depth: int = 16) -> vs.VideoNode:

        down = depth(clip, input_depth, dither_type=dither_type)
        filtered = function(down)
        return depth(filtered, clip.format.bits_per_sample, dither_type=dither_type)

    if fast:
        sqrt = resample(clip, lambda e: core.std.Expr(e, ["x 5 * x * sqrt"]), input_depth=16)
        ret = core.std.MaskedMerge(clip, sqrt, clip.std.PlaneStats().adg.Mask())
    else:
        ret = core.retinex.MSRCP(clip, **msrcp_args) if clip.format.bits_per_sample <= 16 else \
            resample(clip, partial(core.retinex.MSRCP, **msrcp_args), dither_type='none')

    max_value = scale_value(1, 32, clip.format.bits_per_sample, scale_offsets=True, range=Range.FULL)

    tcanny = core.tcanny.TCanny(ret, **tcanny_args).std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])
    expr = core.std.Expr([mask, tcanny], f'x y + {max_value} min')
    return depth(expr, clip.format.bits_per_sample, dither_type='none')


def Cambi(clip: vs.VideoNode,
          filter: vs.VideoNode,
          thr: float = 5.0,
          cambi_args: Optional[Dict[str, Any]] = None,
          debug: bool = False) -> vs.VideoNode:

    cambi_dict: Dict[str, Any] = dict(topk=0.001)
    if cambi_args is not None:
        cambi_dict |= cambi_args

    def _fun(n, f, clip, filter):
        return filter if f.props['CAMBI'] > thr else clip

    ref = depth(clip, 8, dither_type='ordered') \
        if clip.format.bits_per_sample > 8 else clip
    # Maybe we can use our own prefilter to get more consistent scores
    # I'm pretty concerned about the potential for 'flickering'
    # from rgvs import blur
    # rgvs.blur(radius=2): Output 1001 frames in 2.25 seconds (444.40 fps)
    # Convolution [9]*9: Output 1001 frames in 2.11 seconds (475.48 fps)
    # Convolution [9]*25: Output 1001 frames in 3.59 seconds (279.10 fps)

    cambi = core.akarin.Cambi(ref, **cambi_dict)
    process = core.std.FrameEval(clip, partial(_fun, clip=clip, filter=filter), cambi)
    # pass props to clip
    props = core.std.CopyFrameProps(clip, prop_src=cambi)

    if debug is True:
        props = core.std.CopyFrameProps(process, prop_src=cambi)
        return core.text.FrameProps(props, props="CAMBI")

    return props


def autoDeband(clip: vs.VideoNode,
               thr: int | float = 12.0,
               deband_range: tuple | list = (24, 48, 4),
               deband_scale: int | float = 1.2,
               grainer: None | bool | vs.VideoNode = None,
               debander: None | vs.VideoNode = None,
               cambi_args: None | dict = None,
               downsample: None | int = None,
               debug: bool = False) -> vs.VideoNode:
    """
    Automated banding detection and filtration via the use of CAMBI
    A range of potential debanding functions are spawned, of which
    an approximated value is chosen based off the score returned by CAMBI.
    Please see: https://github.com/AkarinVS/vapoursynth-plugin/wiki/CAMBI

    For instance, with default values a CAMBI score of 26.5 would
    return a debanding function with a (strength) value of 24.

    This approach was regretfully necessary due to performance
    issues with std.FrameEval. Or perhaps my code was just bad.

    Default values are more or less targeting live action content.

    Function is extensible, allowing for custom functions for
    debanding and grain applied in place of defaults.
    For anime, consider either disabling the graining function, or
    or using adptvgrnmod with static=True

    Requirements:
        Plugins:
            https://github.com/AkarinVS/vapoursynth-plugin

        Modules:
            numpy
            https://gitlab.com/Ututu/adptvgrnmod
            https://github.com/HomeOfVapourSynthEvolution/havsfunc
            https://github.com/Irrational-Encoding-Wizardry/vs-debandshit

    :param clip:            Clip to be processed.
    :param thr:             CAMBI threshold for processing.
                            Defaults to 12.0.
    :param deband_range:    List or tuple containing precisely 3 integers;
                            supplied values represent (in order):
                            Lower and upper filter threshold, number of spawned filters.
                            Defaults to (24, 48, 4) = [24, 32, 40, 48].
    :param deband_scale:    Multiplication of CAMBI score passed to function.
                            Higher values will result in a stronger median strength.
                            Defaults to 1.2.
    :param debander:        Call a custom debanding function.
                            Function should take a clip and a threshold.
                            Threshold is dynamically generated as per usual
                            Use your own mask. debander=partial(customDebandFunc)
                            Defaults to None.
    :param grainer:         Custom grain function to be applied after debanding.
                            Value passed is is CAMBI * deband_scale.
                            Custom grain function should support i16 input.
                            grainer=partial(core.grain.Add)
                            False to disable grain. Defaults to None
    :param cambi_args:      Dictionary of values passed to akarin.Cambi
                            Defaults to None
    :param downsample:      Decrease CAMBI CPU usage by downsampling input.
                            Upscaling is supported but not recommended.
                            Results may vary. Defaults to None.
    :param debug:           Show relevant frame properties.
                            Defaults to False.
    """
    import numpy as np
    from awsmfunc.base import zr

    # this is ugly asf FIXME
    defaultGrain = [True, 85, 100, 115] \
        if grainer in (None, True) else False
    defaultDeband = True if debander is None else False

    cambi_dict = dict(topk=0.1, tvi_threshold=0.012)
    if cambi_args is not None:
        cambi_dict |= cambi_args

    def _noiseFactory(clip: vs.VideoNode, threshold: float = None) -> vs.VideoNode:
        from havsfunc import GrainFactory3
        from adptvgrnMod import adptvgrnMod

        return adptvgrnMod(clip, lo=18, grainer=lambda g:
            # can I set these with a loop? [n for n in defaultGrain[1:]]
                GrainFactory3(g,
                          g1str=threshold/defaultGrain[1],
                          g2str=threshold/defaultGrain[2],
                          g3str=threshold/defaultGrain[3],
                          seed=422)
            )

    def _debandFactory(clip: vs.VideoNode, threshold: float | int = None) -> vs.VideoNode:
        from debandshit import f3kpf

        ref = depth(clip, 16) if clip.format.bits_per_sample != 16 else clip
        deband = f3kpf(ref, threshold=threshold,
                       f3kdb_args=dict(use_neo=True, sample_mode=4),
                       limflt_args=dict(thr=0.3)) \
            if defaultDeband is True else debander(ref, threshold)

        # this is pretty gross
        if grainer is False:
            pass
        elif grainer in (True, None):
            deband = _noiseFactory(deband, threshold=threshold)
        else:
            deband = grainer(deband, threshold)

        return depth(deband, clip.format.bits_per_sample)

    def _findNearest(array: tuple | list, value: int | float) -> int | float:
        array = np.asarray(array)
        return (np.abs(array - value)).argmin()

    def _fun(n, f, clip, debands) -> None:
        score = f.props['CAMBI'] * deband_scale
        approx = _findNearest(array=array, value=score)
        # How tf can I return the original score as (approx, score)
        return debands[approx] if f.props['CAMBI'] >= thr else clip

    refYClamp = get_y(clip)
    refYClamp = refYClamp.std.Limiter(min=16 << (clip.format.bits_per_sample - 8),
                                      max=235 << (clip.format.bits_per_sample - 8))

    # If we're really struggling for CPU time we can use a quick lanczos pass
    # is lanczos the best filter here? focus: predictability / consistency
    # if we know how the filter will behave, we can adjust appropriately.
    lanczos = dict(kernel='lanczos', filter_param_a=0)
    ref = zr(refYClamp, preset=downsample, **lanczos) \
        if downsample else refYClamp

    ref = depth(ref, 8, dither_type='ordered') \
        if clip.format.bits_per_sample > 8 else get_y(ref)

    cambi = core.akarin.Cambi(ref, **cambi_dict)
    props = core.std.CopyFrameProps(clip, prop_src=cambi)

    array = np.linspace(*deband_range, dtype=int).tolist()
    debands = [_debandFactory(clip, x) for x in array]
    process = clip.std.FrameEval(partial(_fun, clip=props,
                                         debands=debands), props)

    if debug:
        chooseProps = [
            "CAMBI",
            "resolution",
            "deband_str"
            ]

        if defaultGrain:
            grainProps = [
                "g1str",
                "g2str",
                "g3str"
                ]
            [chooseProps.append(p) for p in grainProps]

        def _debugProps(n, f, clip) -> None:
            val = f.props['CAMBI'] * deband_scale

            score = np.asarray(array)[_findNearest(array=array, value=val)] \
                if f.props['CAMBI'] >= thr else 0

            vals = [
                f.props['CAMBI'],
                ref.height,
                score
                ]

            if defaultGrain:
                grainVals = [
                    score / defaultGrain[1],
                    score / defaultGrain[2],
                    score / defaultGrain[3]
                    ]
                [vals.append(v) for v in grainVals]

            for prop, val in zip(chooseProps, vals):
                clip = core.std.SetFrameProp(clip, prop=prop, floatval=val)

            return clip

        process = core.std.FrameEval(props, partial(_debugProps, clip=process), props)
        return core.text.FrameProps(process, props=chooseProps)

    return process


def bbcfcalc(clip, top=0, bottom=0, left=0, right=0, radius=None, thr=32768, blur=999):

    clip = depth(clip, 16)
    radius = max([top, bottom, left, right]) * 2
    cf = clip.cf.ContinuityFixer(top=top, bottom=bottom, left=left, right=right, radius=radius)
    refv = []
    fltv = []

    refh = []
    flth = []

    if top:
        refh.append(clip.std.Crop(bottom=clip.height - top))
        flth.append(cf.std.Crop(bottom=clip.height - top))

    if bottom:
        refh.append(clip.std.Crop(top=clip.height - bottom - 1))
        flth.append(cf.std.Crop(top=clip.height - bottom - 1))

    if left:
        refv.append(clip.std.Crop(right=clip.width - left))
        fltv.append(cf.std.Crop(right=clip.width - left))

    if right:
        refv.append(clip.std.Crop(left=clip.width - right - 1))
        fltv.append(cf.std.Crop(left=clip.width - right - 1))

    bv = max(clip.width / blur, 8)
    bh = max(clip.height / blur, 8)

    if left or right:
        refv = core.std.StackHorizontal(refv)
        fltv = core.std.StackHorizontal(fltv)
        for x in [refv, fltv]:
            x = x.resize.Point(refv.width, bv).resize.Point(refv.width, refv.height)
        outv = core.std.Expr([refv, fltv], [f"x y - {thr} > x {thr} + x y - -{thr} < x {thr} - y ? ?"])

    if top or bottom:
        refh = core.std.StackVertical(refh)
        flth = core.std.StackVertical(flth)
        for x in [refh, flth]:
            x = x.resize.Point(bh, refh.height).resize.Point(refh.width, refh.height)
        outh = core.std.Expr([refh, flth], [f"x y - {thr} > x {thr} + x y - -{thr} < x {thr} - y ? ?"])

    if top and bottom:
        clip = core.std.StackVertical([outh.std.Crop(bottom=bottom + 1),
                                       clip.std.Crop(top=top, bottom=bottom + 1),
                                       outh.std.Crop(top=top)])
    elif top:
        clip = core.std.StackVertical([outh, clip.std.Crop(top=top)])
    elif bottom:
        clip = core.std.StackVertical([clip.std.Crop(bottom=bottom + 1), outh])
    if left and right:
        clip = core.std.StackHorizontal([outv.std.Crop(right=right + 1),
                                         clip.std.Crop(left=left, right=right + 1),
                                         outv.std.Crop(left=left)])
    elif left:
        clip = core.std.StackHorizontal([outv, clip.std.Crop(left=left)])
    elif right:
        clip = core.std.StackHorizontal([clip.std.Crop(right=right + 1), outv])

    return clip


def bbcf(clip, top=0, bottom=0, left=0, right=0, radius=None, thr=128, blur=999, scale_thr=True, planes=None):
    import math

    if scale_thr:
        thr = scale_value(thr, 8, 16)
    if planes is None:
        planes = [x for x in range(clip.format.num_planes)]

    sw, sh = clip.format.subsampling_w, clip.format.subsampling_h

    if sh == 1:
        if not isinstance(top, list):
            top = [top] + 2 * [math.ceil(top / 2)]
        elif len(top) == 2:
            top.append(top[1])
        if not isinstance(bottom, list):
            bottom = [bottom] + 2 * [math.ceil(bottom / 2)]
        elif len(bottom) == 2:
            bottom.append(bottom[1])
    else:
        if not isinstance(top, list):
            top = 3 * [top]
        elif len(top) == 2:
            top.append(top[1])
        if not isinstance(bottom, list):
            bottom = 3 * [bottom]
        elif len(bottom) == 2:
            bottom.append(bottom[1])
    if sw == 1:
        if not isinstance(left, list):
            left = [left] + 2 * [math.ceil(left / 2)]
        elif len(left) == 2:
            left.append(left[1])
        if not isinstance(right, list):
            right = [right] + 2 * [math.ceil(right / 2)]
        elif len(right) == 2:
            right.append(right[1])
    else:
        if not isinstance(left, list):
            left = 3 * [left]
        elif len(left) == 2:
            left.append(left[1])
        if not isinstance(right, list):
            right = 3 * [right]
        elif len(right) == 2:
            right.append(right[1])

    if not isinstance(radius, list):
        radius = 3 * [radius]
    elif len(radius) == 2:
        radius.append(radius[1])
    if not isinstance(thr, list):
        thr = 3 * [thr]
    elif len(thr) == 2:
        thr.append(thr[1])
    if not isinstance(blur, list):
        blur = 3 * [blur]
    elif len(blur) == 2:
        blur.append(blur[1])
    if not isinstance(blur, list):
        blur = 3 * [x]
    elif len(blur) == 2:
        blur.append(x[1])
    if not isinstance(planes, list):
        planes = [planes]

    if not clip.format.color_family == vs.GRAY:
        c = split(clip)
    else:
        c = [clip]
    i = 0
    for x in c:
        if i in planes:
            c[i] = bbcfcalc(c[i], top[i], bottom[i], left[i], right[i], radius[i], thr[i], blur[i])
        i += 1

    if not clip.format.color_family == vs.GRAY:
        return join(c)
    else:
        return c[0]


def f32kdb(clip, range=15, y=32, cb=0, cr=0, sample_mode=4, dither="none"):
    """
    just an easier to read version of vardefunc's dumb3kdb with minor changes:
    * grain is always 0
    * changed defaults
    * clips are merged at 32-bit
    * you can also just pass a 32-bit clip (Idc if this is slower or something)
    """
    from vsutil import depth
    # 16 for sample_mode = 2
    # 32 for rest
    step = 16 if sample_mode == 2 else 32

    odepth = max(clip.format.bits_per_sample, 16)
    clip = depth(clip, 16, dither_type="none")

    if y % step == cb % step == cr % step == 0:
        return depth(clip.neo_f3kdb.Deband(range, y, cb, cr, 0, 0, sample_mode), odepth)
    else:
        loy, locb, locr = [max((th - 1) // step * step + 1, 0) for th in [y, cb, cr]]
        hiy, hicb, hicr = [min(lo + step, 511) for lo in [loy, locb, locr]]

        lo_clip = depth(clip.neo_f3kdb.Deband(range, loy, locb, locr, 0, 0, sample_mode), 32)
        hi_clip = depth(clip.neo_f3kdb.Deband(range, hiy, hicb, hicr, 0, 0, sample_mode), 32)

        if clip.format.color_family == vs.GRAY:
            weight = (y - loy) / step
        else:
            weight = [(y - loy) / step, (cb - locb) / step, (cr - locr) / step]

        return depth(core.std.Merge(lo_clip, hi_clip, weight), odepth, dither_type=dither)
