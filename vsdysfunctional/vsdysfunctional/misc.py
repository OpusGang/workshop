from functools import partial
import vapoursynth as vs
from typing import Optional, Dict, Any, Sequence, Tuple
from vsutil import disallow_variable_format, disallow_variable_resolution, get_y, depth, scale_value, split, join
from lvsfunc.util import get_prop

core = vs.core


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


@disallow_variable_format
@disallow_variable_resolution
def autoDeblock(clip: vs.VideoNode, edgevalue: int = 24,
                function: vs.VideoNode = core.dfttest.DFTTest,
                strs: Sequence[float] = [30, 50, 75],
                thrs: Sequence[Tuple[float, float, float]] = [(1.5, 2.0, 2.0), (3.0, 4.5, 4.5), (5.5, 7.0, 7.0)],
                write_props: bool = False,
                **function_args: Any) -> vs.VideoNode:
    """
    A rewrite of fvsfunc.AutoDeblock that uses {anything}.
    This function checks for differences between a frame and an edgemask with some processing done on it,
    and for differences between the current frame and the next frame.
    For frames where both thresholds are exceeded, it will perform deblocking at a specified strength.
    This will ideally be frames that show big temporal *and* spatial inconsistencies.
    Thresholds and calculations are added to the frameprops to use as reference when setting the thresholds.
    Thanks Vardë, louis, setsugen_no_ao!
    Dependencies:
    * vs-dpir
    :param clip:            Input clip
    :param edgevalue:       Remove edges from the edgemask that exceed this threshold (higher means more edges removed)
    :param strs:            A list of DPIR strength values (higher means stronger deblocking).
                            You can pass any arbitrary number of values here.
                            Sane deblocking strenghts lie between 1–20 for most regular deblocking.
                            Going higher than 50 is not recommended outside of very extreme cases.
                            The amount of values in strs and thrs need to be equal.
    :param thrs:            A list of thresholds, written as [(EdgeValRef, NextFrameDiff, PrevFrameDiff)].
                            You can pass any arbitrary number of values here.
                            The amount of values in strs and thrs need to be equal.
    :param write_props:     Will write verbose props
    :return:                Deblocked clip
    """
    assert clip.format

    def _eval_db(n: int, f: Sequence[vs.VideoFrame],
                 clip: vs.VideoNode, db_clips: Sequence[vs.VideoNode],
                 nthrs: Sequence[Tuple[float, float, float]]) -> vs.VideoNode:

        evref_diff, y_next_diff, y_prev_diff = [
            get_prop(f[i], prop, float)
            for i, prop in zip(range(3), ['EdgeValRefDiff', 'YNextDiff', 'YPrevDiff'])
        ]
        f_type = get_prop(f[0], '_PictType', bytes).decode('utf-8')

        if f_type == 'I':
            y_next_diff = (y_next_diff + evref_diff) / 2

        out = clip
        nthr_used = (-1., ) * 3
        for dblk, nthr in zip(db_clips, nthrs):
            if all(p > t for p, t in zip([evref_diff, y_next_diff, y_prev_diff], nthr)):
                out = dblk
                nthr_used = nthr

        if write_props:
            for prop_name, prop_val in zip(
                ['Adb_EdgeValRefDiff', 'Adb_YNextDiff', 'Adb_YPrevDiff',
                 'Adb_EdgeValRefDiffThreshold', 'Adb_YNextDiffThreshold', 'Adb_YPrevDiffThreshold'],
                [evref_diff, y_next_diff, y_prev_diff] + list(nthr_used)
            ):
                out = out.std.SetFrameProp(prop_name, floatval=max(prop_val * 255, -1))

        return out

    if len(strs) != len(thrs):
        raise ValueError('autodb_dpir: You must pass an equal amount of values to '
                         f'strenght {len(strs)} and thrs {len(thrs)}!')

    nthrs = [tuple(x / 255 for x in thr) for thr in thrs]

    rgb = clip

    maxvalue = (1 << rgb.format.bits_per_sample) - 1
    evref = core.std.Prewitt(rgb)
    evref = core.std.Expr(evref, f"x {edgevalue} >= {maxvalue} x ?")
    evref_rm = evref.std.Median().std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])

    diffevref = core.std.PlaneStats(evref, evref_rm, prop='EdgeValRef')
    diffnext = core.std.PlaneStats(rgb, rgb.std.DeleteFrames([0]), prop='YNext')
    diffprev = core.std.PlaneStats(rgb, rgb[0] + rgb, prop='YPrev')

    db_clips = [
        function(rgb, **function_args).std.SetFrameProp('Adb_DeblockStrength', intval=int(st)) for st in strs
    ]

    debl = core.std.FrameEval(
        rgb, partial(_eval_db, clip=rgb, db_clips=db_clips, nthrs=nthrs),
        prop_src=[diffevref, diffnext, diffprev]
    )

    return debl


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
