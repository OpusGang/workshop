from functools import partial
import vapoursynth as vs
core = vs.core

from typing import Callable, Optional, Dict, Any
from vsutil import get_y, depth, Range, scale_value, split, join

def CoolDegrain(clip: vs.VideoNode, tr: int = 2, thSAD: int = 72, thSADC: Optional[int] = None,
                planes: list[int] = [0, 1, 2], blksize: Optional[int] = None,
                overlap: Optional[int] = None, pel: Optional[int] = None, recalc: bool = False,
                pf: Optional[Callable[[vs.VideoNode], vs.VideoNode]] = None) -> vs.VideoNode:
    from zzfunc.util import vs_to_mv
    import rgvs

    """
    Fairly sure this originates from Beatrice-Raws
    Ostensibly, this is a simplified version of SMDegrain;
    Can result in lesser quality output despite the additional percision.
    One downside to the current implementation is that when applied to Y'CbCr content
    calculations are done for all planes regardless of what we are actually denoising.
    Due to this, it may be best to split the clip into its respective planes.

    Raises:
        TypeError: CoolDegrain: This is not a clip
        Warning: CoolDegrain: (32f) thSADC does not work at this depth
        ValueError: CoolDegrain: (16i) tr must be between 1 and 3
        ValueError: CoolDegrain: (32f) tr must be between 1 and 24

    Returns:
        vs.VideoNode: Denoised clip
    """

    if not isinstance(clip, vs.VideoNode):
        raise TypeError('CoolDegrain: (8~16i, 32f) This is not a clip')

    if clip.format.bits_per_sample <= 16 and tr not in range(1, 4):
        raise ValueError('CoolDegrain: (8~16i) tr must be between 1 and 3')

    if clip.format.bits_per_sample == 32:
        if thSADC is not None:
            raise ValueError('CoolDegrain: (32f) thSADC does not work at this depth')
        if tr not in range(1, 25):
            raise ValueError('CoolDegrain: (32f) tr must be between 1 and 24')

    plane = vs_to_mv(planes)

    pfclip = pf if pf is not None else clip

    if thSADC is None:
        thSADC = thSAD

    if blksize is None:
        if clip.width < 1280 or clip.height < 720:
            blksize = 8
        elif clip.width >= 3840 or clip.height >= 2160:
            blksize = 32
        else:
            blksize = 16

    if overlap is None:
        overlap = blksize // 2

    if pel is None:
        if clip.width < 1280 or clip.height < 720:
            pel = 2
        else:
            pel = 1

    def _CoolDegrain16(clip: vs.VideoNode, tr: int = 2, thSAD: int = 72, thSADC: Optional[int] = None,
                planes: list[int] = [0, 1, 2], blksize: Optional[int] = None,
                overlap: Optional[int] = None, pel: Optional[int] = None, recalc: bool = False,
                pf: Optional[Callable[[vs.VideoNode], vs.VideoNode]] = None) -> vs.VideoNode:

        super = core.mv.Super(pfclip, pel=pel, sharp=2, rfilter=4)

        # at least tr=1, so no checks here
        mvbw1 = core.mv.Analyse(super, isb=True, delta=1, overlap=overlap, blksize=blksize)
        mvfw1 = core.mv.Analyse(super, isb=False, delta=1, overlap=overlap, blksize=blksize)
        if tr >= 2:
            mvbw2 = core.mv.Analyse(super, isb=True, delta=2, overlap=overlap, blksize=blksize)
            mvfw2 = core.mv.Analyse(super, isb=False, delta=2, overlap=overlap, blksize=blksize)

        if tr >= 3:
            mvbw3 = core.mv.Analyse(super, isb=True, delta=3, overlap=overlap, blksize=blksize)
            mvfw3 = core.mv.Analyse(super, isb=False, delta=3, overlap=overlap, blksize=blksize)

        if recalc is True:
            hoverlap = overlap // 2
            hblksize = blksize // 2
            hthsad = thSAD // 2

            prefilt = rgvs.RemoveGrain(clip, mode=4, planes=plane)
            super_r = core.mv.Super(prefilt, pel=pel, sharp=2, rfilter=4)

            mvbw1 = core.mv.Recalculate(super_r, mvbw1, overlap=hoverlap, blksize=hblksize, thsad=hthsad)
            mvfw1 = core.mv.Recalculate(super_r, mvfw1, overlap=hoverlap, blksize=hblksize, thsad=hthsad)

            if tr >= 2:
                mvbw2 = core.mv.Recalculate(super_r, mvbw2, overlap=hoverlap, blksize=hblksize, thsad=hthsad)
                mvfw2 = core.mv.Recalculate(super_r, mvfw2, overlap=hoverlap, blksize=hblksize, thsad=hthsad)

            if tr >= 3:
                mvbw3 = core.mv.Recalculate(super_r, mvbw3, overlap=hoverlap, blksize=hblksize, thsad=hthsad)
                mvfw3 = core.mv.Recalculate(super_r, mvfw3, overlap=hoverlap, blksize=hblksize, thsad=hthsad)

        if tr == 1:
            filtered = core.mv.Degrain1(clip=clip, super=super, mvbw=mvbw1, mvfw=mvfw1, thsad=thSAD, thsadc=thSADC, plane=plane)
        elif tr == 2:
            filtered = core.mv.Degrain2(clip=clip, super=super, mvbw=mvbw1, mvfw=mvfw1, mvbw2=mvbw2, mvfw2=mvfw2, thsad=thSAD, thsadc=thSADC, plane=plane)
        elif tr == 3:
            filtered = core.mv.Degrain3(clip=clip, super=super, mvbw=mvbw1, mvfw=mvfw1, mvbw2=mvbw2, mvfw2=mvfw2, mvbw3=mvbw3, mvfw3=mvfw3, thsad=thSAD, thsadc=thSADC, plane=plane)
        return filtered


    def _CoolDegrain32(clip: vs.VideoNode, tr: int = 2, thSAD: int = 72, thSADC: Optional[int] = None,
                planes: list[int] = [0, 1, 2], blksize: Optional[int] = None,
                overlap: Optional[int] = None, pel: Optional[int] = None, recalc: bool = False,
                pf: Optional[Callable[[vs.VideoNode], vs.VideoNode]] = None) -> vs.VideoNode:

        super = core.mvsf.Super(clip, pel=pel, sharp=2, rfilter=4)
        analyse = core.mvsf.Analyze(super, radius=tr, isb=True, overlap=overlap, blksize=blksize)

        # This seems more than useless, even with better prefilters (KNL, bilateral, ...).
        if recalc is True:
            hoverlap = overlap // 2
            hblksize = blksize // 2
            hthsad = thSAD // 2

            prefilt = rgvs.removegrain(clip, mode=4, planes=planes)
            super_r = core.mvsf.Super(prefilt, pel=pel, sharp=2, rfilter=4)
            analyse = core.mvsf.Recalculate(super_r, analyse, overlap=overlap, blksize=blksize)

        # Unforunately, we cannot make use of thSADC at this depth.
        # I don't generally recommend mvtools for chroma processing anyway.
        filtered = core.mvsf.Degrain(clip, super, analyse, thsad=thSAD, plane=plane, limit=1)
        return filtered


    if clip.format.bits_per_sample <= 16:
        return _CoolDegrain16(clip=clip, tr=tr, thSAD=thSAD, thSADC=thSADC, planes=planes,
                              blksize=blksize, overlap=overlap, pel=pel, recalc=recalc, pf=pf)
    else:
        return _CoolDegrain32(clip=clip, tr=tr, thSAD=thSAD, thSADC=thSADC, planes=planes,
                        blksize=blksize, overlap=overlap, pel=pel, recalc=recalc, pf=pf)


def unknownDideeDNR1(clip: vs.VideoNode,
                     ref: Optional[Callable[[vs.VideoNode], vs.VideoNode]] = None,
                     thSAD: int = 125,
                     repair: int = 1) -> vs.VideoNode:
    """
    https://forum.doom9.org/showthread.php?p=1076491#post1076491
    This seems to be more or less a precursor to SMDegrain
    Replaced FFT3D with KNLMeansCL for a speed increase (assumping you have a GPU)

    Returns:
        vs.VideoNode: Denoised clip
    """
    import rgvs

    neutral = scale_value(128, 8, clip.format.bits_per_sample)

    # Here, we simply use FFT3DFilter. There're lots of other possibilities. Basically, you shouldn't use 
    # a clip with "a tiny bit of filtering". The search clip has to be CALM. Ideally, it should be "dead calm".
    # core.fft3dfilter.FFT3DFilter(clip, sigma=16, sigma2=10, sigma3=6, sigma4=4, bt=5, bw=16, bh=16, ow=8, oh=8)

    def knl(clip: vs.VideoNode, h: int = 2, a: int = 2, s: int = 1, d: int = 2) -> vs.VideoNode:
        knl = core.knlm.KNLMeansCL(clip, h=h, a=a, s=s, d=d, channels='Y')
        return core.knlm.KNLMeansCL(knl, h=h, a=a, s=s, d=d, channels='UV')

    refClip = ref or knl(clip)
    diffClip1 = core.std.MakeDiff(clip, refClip)

    # motion vector search (with very basic parameters. Add your own parameters as needed.)
    suClip = core.mv.Super(refClip, pel=2, sharp=2)
    b3vec1 = core.mv.Analyse(suClip, isb=True,  delta=3, pelsearch=2, overlap=4)
    b2vec1 = core.mv.Analyse(suClip, isb=True,  delta=2, pelsearch=2, overlap=4)
    b1vec1 = core.mv.Analyse(suClip, isb=True,  delta=1, pelsearch=2, overlap=4)
    f1vec1 = core.mv.Analyse(suClip, isb=False, delta=1, pelsearch=2, overlap=4)
    f2vec1 = core.mv.Analyse(suClip, isb=False, delta=2, pelsearch=2, overlap=4)
    f3vec1 = core.mv.Analyse(suClip, isb=False, delta=3, pelsearch=2, overlap=4)

    # 1st MV-denoising stage. Usually here's some temporal-median filtering going on.
    # For simplicity, we just use MVDegrain.
    removeNoise = core.mv.Degrain3(clip, super=suClip, mvbw=b1vec1, mvfw=f1vec1,
                                   mvbw2=b2vec1, mvfw2=f2vec1,
                                   mvbw3=b3vec1, mvfw3=f3vec1, thsad=thSAD)
    mergeDiff1 = core.std.MergeDiff(clip, removeNoise)

    # limit NR1 to not do more than what "spat" would do
    limitNoise = core.std.Expr([diffClip1, mergeDiff1],
                               expr=[f"x {neutral} - abs y {neutral} - abs < x y ?"])
    diffClip2 = core.std.MakeDiff(clip, limitNoise)

    # 2nd MV-denoising stage. We use MVDegrain.
    removeNoise2 = core.mv.Degrain3(diffClip2, super=suClip, mvbw=b1vec1, mvfw=f1vec1,
                                    mvbw2=b2vec1, mvfw2=f2vec1,
                                    mvbw3=b3vec1, mvfw3=f3vec1, thsad=thSAD)

    # contra-sharpening: sharpen the denoised clip, but don't add more to any pixel than what was removed previously.
    # (Here: a simple area-based version with relaxed restriction. The full version is more complicated.)

    # damp down remaining spots of the denoised clip
    postBlur = rgvs.minblur(removeNoise2, radius=1, planes=[0,1,2])
    # the difference achieved by the denoising
    postDiff = core.std.MakeDiff(clip, removeNoise2)
    # the difference of a simple kernel blur
    postDiff2 = core.std.MakeDiff(postBlur, core.std.Convolution(postBlur,
                                                                 matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]))
    # limit the difference to the max of what the denoising removed locally.
    postRepair = rgvs.repair(postDiff2, postDiff, mode=repair, planes=[0,1,2])
    # abs(diff) after limiting may not be bigger than before.
    postExpr = core.std.Expr([postRepair, postDiff2],
                             expr=[f"x {neutral} - abs y {neutral} - abs < x y ?"])

    return core.std.MergeDiff(removeNoise2, postExpr)


def retinex(clip: vs.VideoNode, mask: Callable[[vs.VideoNode], vs.VideoNode],
            msrcp_dict: Optional[Dict[str, Any]] = None, tcanny_dict: Optional[Dict[str, Any]] = None) -> vs.VideoNode:

    msrcp_args: Dict[str, Any] = dict(sigma=[50, 200, 350], upper_thr=0.005, fulls=False)
    if msrcp_dict is not None:
        msrcp_args |= msrcp_dict

    tcanny_args: Dict[str, Any] = dict(sigma=1, mode=1)
    if tcanny_dict is not None:
        tcanny_args |= tcanny_dict

    if clip.format.num_planes > 1:
        clip = get_y(clip)
        mask = get_y(mask)

    if clip.format.bits_per_sample == 32: 

        def resample(clip: vs.VideoNode, function: Callable[[vs.VideoNode], vs.VideoNode],
                     dither_type: str = 'none') -> vs.VideoNode:
            # Copy paste stolen code from lvsfunc but forced rounding

            down = depth(clip, 16, dither_type=dither_type)
            filtered = function(down)
            return depth(filtered, clip.format.bits_per_sample, dither_type=dither_type)

        ret = resample(clip, lambda x: core.retinex.MSRCP(x, **msrcp_args))
        max_value = 1

    else:
        ret = core.retinex.MSRCP(clip, **msrcp_args)
        max_value = scale_value(1, 32, clip.format.bits_per_sample, scale_offsets=True, range=Range.FULL)

    tcanny = core.tcanny.TCanny(ret, **tcanny_args).std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])
    return depth(core.std.Expr([mask, tcanny], f'x y + {max_value} min'), clip.format.bits_per_sample, dither_type='none')


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
                thr: int | float | tuple | list = (10.0, None),
                f3kdb_range: tuple | list  = (12, 48),
                f3kdb_step: int | float = 6,
                f3kdb_scale: int | float = 1.2,
                grainer: None | bool | vs.VideoNode = None,
                cambi_args: None | dict = None,
                debug: bool = False) -> vs.VideoNode:
    """Automated banding detection and filtration via the use of CAMBI
    Example range of values can be returned in your terminal:
        python -c 'print(*range(12, 48, 6))' = 12 18 24 30 36 42

    Args:
        clip (vs.VideoNode): Input clip
        thr (float | tuple): Threshold for detection. Defaults to (10.0, None).
        f3kdb_range (tuple | list): Lower and upper bound of f3kdb strength.
            Defaults to (12, 48).
        f3kdb_step (int | float): Granularity of f3kdb values. Defaults to 6.
        f3kdb_scale (int | float): Multiplication of CAMBI score.
            higher values will result in a stronger median f3dkb scrength.
                Defaults to 1.2.
        grainer (None | bool | vs.VideoNode): Grain application after debanding.
            Strength is CAMBI * fk3db_scale / 100. False to disable.
                Custom grain function should support i16 input Defaults to None.
        cambi_args (None | dict): Pass arguments to akarin.Cambi
            https://github.com/AkarinVS/vapoursynth-plugin/wiki/CAMBI.
                Defaults to None.
        debug (bool): Show relevant frame properties. Defaults to False.

    Returns:
        vs.VideoNode: [description]
    """
    from debandshit.debanders import f3kpf, dumb3kdb
    from lvsfunc.mask import detail_mask
    from lvsfunc.util import get_prop
    from rgvs import Blur
    import numpy as np

    if not isinstance(thr, tuple | list):
        thr = (thr, None)

    cambi_dict: Dict[str, Any] = dict(topk=0.001)
    if cambi_args is not None:
        cambi_dict |= cambi_args

    def _noiseFactory(clip: vs.VideoNode, threshold: int = None) -> vs.VideoNode:
        from havsfunc import GrainFactory3
        from adptvgrnMod import adptvgrnMod
        return adptvgrnMod(clip, lo=18, grainer=lambda g:
            GrainFactory3(g, g1str=threshold/100, g2str=threshold/100,
                          temp_avg=66, seed=422))
    # maybe we should also define our own debanding function rather than using debandshit
    # it would be nice to (optionally) use placebo.Deband.
    def _debandFactory(clip: vs.VideoNode, threshold: int = None) -> vs.VideoNode:
        ref = depth(clip, 16) if clip.format.bits_per_sample != 16 else clip

        mask_pre = Blur(get_y(ref))
        mask = detail_mask(mask_pre, sigma=False, rad=5)
        deband = f3kpf(ref,
                       threshold=threshold,
                       f3kdb_args=dict(use_neo=True, sample_mode=4))

        if thr[1] not in (None, 0) and threshold >= thr[1]:
            # using a stronger mask for the second pass
            extra_mask = detail_mask(mask_pre, sigma=False, rad=5)
            extra_deband = dumb3kdb(deband, threshold=int(threshold / f3kdb_scale),
                                    use_neo=True, sample_mode=4)
            deband = core.std.MaskedMerge(extra_deband, deband, extra_mask)

        merge = core.std.MaskedMerge(deband, ref, mask)

        grain = grainer(merge) if grainer is not None \
            else _noiseFactory(merge, threshold=threshold)

        return depth(grain, clip.format.bits_per_sample)

    def _propFactory(clip: vs.VideoNode,
                     src_prop: str = "CAMBI",
                     target_prop: str = None,
                     ref: None | vs.VideoNode = None):
        def _prop(n, f, clip):
            return core.std.SetFrameProp(clip, prop=f'{target_prop}',
                                         floatval=get_prop(f, src_prop, t=float))

        refClip = clip if ref is None else ref.resize.Point(clip.width, clip.height)
        setProp = core.std.FrameEval(clip, partial(_prop, clip=refClip), clip)
        copyProp = core.std.CopyFrameProps(clip, setProp)
        return copyProp

    def _findNearest(array: list, value: int | float):
        array = np.asarray(array)
        return (np.abs(array - value)).argmin()

    def _fun(n, f, clip, debands):
        val = f.props['CAMBI'] * f3kdb_scale
        return debands[_findNearest(array=array, value=val)] \
            if f.props['CAMBI'] > thr[0] else clip

    ref = depth(clip, 8, dither_type='ordered') \
        if clip.format.bits_per_sample > 8 else clip

    cambi = core.akarin.Cambi(ref, **cambi_dict)
    props = core.std.CopyFrameProps(clip, prop_src=cambi)

    # if we want to use thr as the lower bound maybe something like 
    # # if {all} in range > thr pass else for any in range < thr = thr
    array = range(*f3kdb_range, f3kdb_step) 
    debands = [_debandFactory(clip, x) for x in array]
    process = core.std.FrameEval(clip, partial(_fun, clip=props, debands=debands), props)

    if debug is True:
        def _debugProps(n, f, clip):
            val = f.props['CAMBI'] * f3kdb_scale

            score = np.asarray(array)[_findNearest(array=array, value=val)] \
                if f.props['CAMBI'] > thr[0] else 0

            props = ["CAMBI", 
                     "f3kdb_first", 
                     "f3kdb_second", 
                     "g1str", 
                     "g2str"]
            vals = [f.props['CAMBI'],
                    score,
                    score / 1.5 if thr[1] not in (None, 0) else 0,
                    score / 100 if grainer in (True, None) else 0,
                    score / 100 if grainer in (True, None) else 0]

            for prop, val in zip(props, vals):
                clip = core.std.SetFrameProp(clip, prop=prop, floatval=val)

            return clip

        process = core.std.FrameEval(props, partial(_debugProps, clip=process), props)
         # setting these twice is dumb. fix it.
         # also add something to drop g1str/g2str from returned props
         # if a custom noise generator is used
        return core.text.FrameProps(process,
                                    props=["CAMBI",
                                           "f3kdb_first",
                                           "f3kdb_second",
                                           "g1str",
                                           "g2str"])

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
        clip = core.std.StackVertical([outh.std.Crop(bottom=bottom + 1), clip.std.Crop(top=top, bottom=bottom + 1), outh.std.Crop(top=top)])
    elif top:
        clip = core.std.StackVertical([outh, clip.std.Crop(top=top)])
    elif bottom:
        clip = core.std.StackVertical([clip.std.Crop(bottom=bottom + 1), outh])
    if left and right:
        clip = core.std.StackHorizontal([outv.std.Crop(right=right + 1), clip.std.Crop(left=left, right=right + 1), outv.std.Crop(left=left)])
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


def ssimdown(clip: vs.VideoNode, preset: Optional[int] = None, repair: Optional[list[float]] = None, width: Optional[int] = None,
             height: Optional[int] = None, left: int = 0, right: int = 0, bottom: int = 0, top: int = 0, ar: str = 16 / 9,
             shader_path: Optional[str] = None, shader_str: Optional[str] = None, repair_fun: Optional[Dict[str, Any]] = None) -> vs.VideoNode:
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

    w = round(((clip.width - left - right) * rw) / 2) * 2
    h = round(((clip.height - top - bottom) * rh) / 2) * 2

    ind = clip.format.bits_per_sample

    clip = depth(clip, 16)

    shift = .25 - .25 * clip.width / w

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