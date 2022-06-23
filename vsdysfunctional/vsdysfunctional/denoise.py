import vapoursynth as vs
from typing import Any, Callable, Dict, Optional
from vsutil import scale_value
core = vs.core


def CoolDegrain(
        clip: vs.VideoNode, tr: int = 2, thSAD: int = 72, thSADC: Optional[int] = None,
        planes: list[int] = [0, 1, 2], blksize: Optional[int] = None,
        overlap: Optional[int] = None, pel: Optional[int] = None, recalc: bool = False,
        pf: Optional[Callable[[vs.VideoNode], vs.VideoNode]] = None
        ) -> vs.VideoNode:
    """
    Fairly sure this originates from Beatrice-Raws
    Ostensibly, this is a simplified version of SMDegrain;
    Can result in lesser quality output despite the additional percision.
    One downside to the current implementation is that when applied to Y'CbCr content
    calculations are done for all planes regardless of what we are actually denoising.
    Due to this, it may be best to split the clip into its respective planes.

    Raises:
        TypeError: CoolDegrain: This is not a clip.

        Warning: CoolDegrain: (32f) thSADC does not work at this depth.

        ValueError: CoolDegrain: (16i) tr must be between 1 and 3.

        ValueError: CoolDegrain: (32f) tr must be between 1 and 24.

    Returns:
        vs.VideoNode: Denoised clip
    """
    from zzfunc.util import vs_to_mv
    import rgvs

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

    def _CoolDegrain16(
            clip: vs.VideoNode, tr: int = 2, thSAD: int = 72, thSADC: Optional[int] = None,
            planes: list[int] = [0, 1, 2], blksize: Optional[int] = None,
            overlap: Optional[int] = None, pel: Optional[int] = None, recalc: bool = False,
            pf: Optional[Callable[[vs.VideoNode], vs.VideoNode]] = None
            ) -> vs.VideoNode:

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
            recalc_dict = dict(overlap=hoverlap, blksize=hblksize, thsad=hthsad)

            prefilt = rgvs.RemoveGrain(clip, mode=4, planes=planes)
            super_r = core.mv.Super(prefilt, pel=pel, sharp=2, rfilter=4)

            mvbw1 = core.mv.Recalculate(super_r, mvbw1, *recalc_dict)
            mvfw1 = core.mv.Recalculate(super_r, mvfw1, *recalc_dict)

            if tr >= 2:
                mvbw2 = core.mv.Recalculate(super_r, mvbw2, *recalc_dict)
                mvfw2 = core.mv.Recalculate(super_r, mvfw2, *recalc_dict)

            if tr >= 3:
                mvbw3 = core.mv.Recalculate(super_r, mvbw3, *recalc_dict)
                mvfw3 = core.mv.Recalculate(super_r, mvfw3, *recalc_dict)

        if tr == 1:
            filtered = core.mv.Degrain1(
                clip=clip, super=super, mvbw=mvbw1, mvfw=mvfw1, thsad=thSAD,
                thsadc=thSADC, plane=plane
                )

        elif tr == 2:
            filtered = core.mv.Degrain2(
                clip=clip, super=super, mvbw=mvbw1, mvfw=mvfw1, mvbw2=mvbw2,
                mvfw2=mvfw2, thsad=thSAD, thsadc=thSADC, plane=plane
                )

        elif tr == 3:
            filtered = core.mv.Degrain3(
                clip=clip, super=super, mvbw=mvbw1, mvfw=mvfw1, mvbw2=mvbw2,
                mvfw2=mvfw2, mvbw3=mvbw3, mvfw3=mvfw3, thsad=thSAD, thsadc=thSADC, plane=plane
                )

        return filtered

    def _CoolDegrain32(
            clip: vs.VideoNode, tr: int = 2, thSAD: int = 72, thSADC: Optional[int] = None,
            planes: list[int] = [0, 1, 2], blksize: Optional[int] = None,
            overlap: Optional[int] = None, pel: Optional[int] = None,
            recalc: bool = False, pf: Optional[Callable[[vs.VideoNode], vs.VideoNode]] = None
            ) -> vs.VideoNode:

        super = core.mvsf.Super(clip, pel=pel, sharp=2, rfilter=4)
        analyse = core.mvsf.Analyze(super, radius=tr, isb=True, overlap=overlap, blksize=blksize)

        # This seems more than useless, even with better prefilters (KNL, bilateral, ...).
        if recalc is True:
            overlap = overlap // 2
            blksize = blksize // 2
            thSAD = thSAD // 2

            prefilt = rgvs.Blur(clip, radius=2, planes=planes)
            super_r = core.mvsf.Super(prefilt, pel=pel, sharp=2, rfilter=4)
            analyse = core.mvsf.Recalculate(
                super_r, analyse, overlap=overlap, blksize=blksize
                )

        # Unforunately, we cannot make use of thSADC at this depth.
        # I don't generally recommend mvtools for chroma processing anyway.
        filtered = core.mvsf.Degrain(
            clip, super, analyse, thsad=thSAD, plane=plane, limit=1
            )

        return filtered

    if clip.format.bits_per_sample <= 16:
        return _CoolDegrain16(
            clip=clip, tr=tr, thSAD=thSAD, thSADC=thSADC,
            planes=planes, blksize=blksize, overlap=overlap,
            pel=pel, recalc=recalc, pf=pf
            )
    else:
        return _CoolDegrain32(
            clip=clip, tr=tr, thSAD=thSAD, thSADC=thSADC,
            planes=planes, blksize=blksize, overlap=overlap,
            pel=pel, recalc=recalc, pf=pf
            )


def unknownDideeDNR1(
        clip: vs.VideoNode,
        srch: Optional[Callable[[vs.VideoNode], vs.VideoNode]] = None,
        spat: Optional[Callable[[vs.VideoNode], vs.VideoNode]] = None,
        thSAD: int = 125,
        sharp: bool = True,
        ) -> vs.VideoNode:
    """
    https://forum.doom9.org/showthread.php?p=1076491#post1076491
    Precursor to TemporalDegrain
    Replaced FFT3D with DFTTest
    Returns:
        vs.VideoNode: Denoised clip
    """
    from havsfunc import ContraSharpening

    srch = srch or core.dfttest.DFTTest(clip, tbsize=1, sigma=8)
    spat = spat or srch

    # motion vector search (with very basic parameters)
    suClip = core.mv.Super(srch, pel=2, sharp=2)

    vect = []
    for x in range(3):
        vect.append(
            core.mv.Analyse(suClip, isb=True, delta=[x + 1], pelsearch=2, overlap=4)
        )
        vect.append(
            core.mv.Analyse(suClip, isb=False, delta=[x + 1], pelsearch=2, overlap=4)
        )

    # 1st MV-denoising stage. Usually here's some temporal-median filtering going on.
    # For simplicity, we just use MVDegrain.
    suClip = core.mv.Super(clip, pel=2, sharp=2)
    removeNoise = core.mv.Degrain3(
        clip,
        super=suClip,
        mvbw=vect[0],
        mvfw=vect[1],
        mvbw2=vect[2],
        mvfw2=vect[3],
        mvbw3=vect[4],
        mvfw3=vect[5],
        thsad=thSAD,
    )

    # limit NR1 to not do more than what "spat" would do
    limitNoise = core.std.Expr([clip, spat, removeNoise], "x y - abs x z - abs < y z ?")

    # 2nd MV-denoising stage. We use MVDegrain.
    suClip = core.mv.Super(limitNoise, pel=2, sharp=2)
    removeNoise2 = core.mv.Degrain3(
        limitNoise,
        super=suClip,
        mvbw=vect[0],
        mvfw=vect[1],
        mvbw2=vect[2],
        mvfw2=vect[3],
        mvbw3=vect[4],
        mvfw3=vect[5],
        thsad=thSAD,
    )

    if sharp:
        removeNoise2 = ContraSharpening(removeNoise2, clip)

    return removeNoise2


def MLMDegrain(
        clip: vs.VideoNode,
        mv_dict: None | dict = None) -> vs.VideoNode:
    import rgvs

    params = dict(
        global_=True,
        truemotion=False,
        blksize=16,
        overlap=8,
        search=5,
        searchparam=2,
        dct=5
    )

    if mv_dict is not None:
        params |= mv_dict

    half = core.resize.Bicubic(
        clip, width=clip.width / 2, height=clip.height / 2,
        filter_param_a=1/3, filter_param_b=1/3
    )
    half_up = core.resize.Bicubic(
        half, width=clip.width, height=clip.height,
        filter_param_a=1/3, filter_param_b=1/3
    )

    sup00 = rgvs.RemoveGrain(half, mode=11)
    sup00 = core.mv.Super(sup00, pel=2, sharp=1)
    sup01 = core.mv.Super(half, pel=2, sharp=1, levels=1)

    bv02 = core.mv.Analyse(sup00, isb=True, delta=2, **params)
    bv01 = core.mv.Analyse(sup00, isb=True, delta=1, **params)
    fv01 = core.mv.Analyse(sup00, isb=False, delta=1, **params)
    fv02 = core.mv.Analyse(sup00, isb=False, delta=2, **params)

    nr00 = core.mv.Degrain2(
        half, super=sup01, mvbw=bv01, mvfw=fv01, mvbw2=bv02, mvfw2=fv02
    ).resize.Bicubic(
        width=clip.width, height=clip.height,
        filter_param_a=1/3, filter_param_b=1/3
        )

    nr01_diff = core.std.MakeDiff(clip, half_up)
    nr01_diff = core.std.MergeDiff(nr00, nr01_diff)

    nr01a = core.akarin.Expr(
        [nr01_diff, clip], expr=["x 2 + y < x 2 + x 2 - y > x 2 - y ? ?"]
    )

    sup10 = core.flux.SmoothT(nr00, temporal_threshold=5)
    sup10 = rgvs.RemoveGrain(sup10, mode=11)
    sup10 = core.mv.Super(sup10, pel=2, sharp=1)

    sup11 = core.mv.Super(nr01a, pel=2, sharp=1, levels=1)

    bv13 = core.mv.Analyse(sup10, isb=True, delta=3, **params)
    bv12 = core.mv.Analyse(sup10, isb=True, delta=2, **params)
    bv11 = core.mv.Analyse(sup10, isb=True, delta=1, **params)
    fv11 = core.mv.Analyse(sup10, isb=False, delta=1, **params)
    fv12 = core.mv.Analyse(sup10, isb=False, delta=2, **params)
    fv13 = core.mv.Analyse(sup10, isb=False, delta=3, **params)

    nr01b = core.mv.Degrain3(
        nr01a, super=sup11,
        mvbw=bv11, mvfw=fv11,
        mvbw2=bv12, mvfw2=fv12,
        mvbw3=bv13, mvfw3=fv13
    )

    expr = core.akarin.Expr(
        [nr01b, clip], expr=["x y < x 1 + x y > x 1 - x ? ?"]
    )

    return expr
