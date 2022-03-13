import vapoursynth as vs
from typing import Callable, Optional, Dict, Any
from vsutil import get_y, depth, Range, scale_value, split, join

core = vs.core


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
            overlap = overlap // 2
            blksize = blksize // 2
            thSAD = thSAD // 2

            prefilt = rgvs.removegrain(clip, mode=4, planes=planes)
            super_r = core.mvsf.Super(prefilt, pel=pel, sharp=2, rfilter=4)
            analyse = core.mvsf.Recalculate(super_r, analyse, overlap=overlap, blksize=blksize)

        # Unforunately, we cannot make use of thSADC at this depth.
        # I don't generally recommend mvtools for chroma processing anyway.
        filtered = core.mvsf.Degrain(clip, super, analyse, thsad=thSAD, plane=plane, limit=1)
        return filtered

    if clip.format.bits_per_sample <= 16:
        return _CoolDegrain16(clip=clip, tr=tr, thSAD=thSAD, thSADC=thSADC,
                              planes=planes, blksize=blksize, overlap=overlap,
                              pel=pel, recalc=recalc, pf=pf)
    else:
        return _CoolDegrain32(clip=clip, tr=tr, thSAD=thSAD, thSADC=thSADC,
                              planes=planes, blksize=blksize, overlap=overlap,
                              pel=pel, recalc=recalc, pf=pf)


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
    removeNoise2 = core.mv.Degrain3(diffClip2, super=suClip,
                                    mvbw=b1vec1, mvfw=f1vec1,
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