import rgvs as RGToolsVS
import vapoursynth as vs
from functools import partial
from typing import Dict, Any
from vsutil import get_y, depth, Range, scale_value
core = vs.core


def __gauss(clip: vs.VideoNode, thresh: int = None) -> vs.VideoNode:
    gauss = core.fmtc.resample(clip, w=clip.width * 2, h=clip.height * 2,
                               kernel='gauss', a1=100)
    return core.fmtc.resample(gauss, w=clip.width, h=clip.height,
                              kernel='gauss', a1=thresh)


def __quickResample(clip: vs.VideoNode,
                    function: vs.VideoNode,
                    dither_type: str = 'none',
                    input_depth: int = 16) -> vs.VideoNode:

    down = depth(clip, input_depth, dither_type=dither_type)
    filtered = function(down)
    return depth(filtered, clip.format.bits_per_sample,
                 dither_type=dither_type)


def pickFn(function: vs.VideoNode, functionAlt: vs.VideoNode, bits: int = 16) -> None:
    return function if bits < 32 else functionAlt


def mvFuncComp(clip: vs.VideoNode, func: vs.VideoNode, **func_args) -> vs.VideoNode:
    """basic motion compensation via mvtools"""
    bits = clip.format.bits_per_sample

    mvSuper = pickFn(core.mv.Super, core.mvsf.Super, bits)(clip)

    vectorBck = pickFn(core.mv.Analyse, core.mvsf.Analyze, bits)(mvSuper, isb=True, delta=1, blksize=8, overlap=4)
    vectorFwd = pickFn(core.mv.Analyse, core.mvsf.Analyze, bits)(mvSuper, isb=False, delta=1, blksize=8, overlap=4)
    compBck = pickFn(core.mv.Compensate, core.mvsf.Compensate, bits)(clip, super=mvSuper, vectors=vectorBck)
    compFwd = pickFn(core.mv.Compensate, core.mvsf.Compensate, bits)(clip, super=mvSuper, vectors=vectorFwd)

    interleave = core.std.Interleave(clips=[compFwd, clip, compBck])
    process = func(interleave, **func_args)
    return core.std.SelectEvery(process, cycle=3, offsets=1)


def csharp(flt: vs.VideoNode, src: vs.VideoNode,
           mode: int = 20) -> vs.VideoNode:
    """
    Stolen from Zastin
    works ok
    """

    np = flt.format.num_planes
    blur = RGToolsVS.RemoveGrain(flt, mode=mode)
    return core.std.Expr([flt, src, blur],
                         ['x dup + z - x y min max x y max min', '', ''][:np])


def fastFreqMerge(lo: vs.VideoNode, hi: vs.VideoNode,
                  function: vs.VideoNode = __gauss,
                  **args) -> vs.VideoNode:

    hi_freq = core.std.MakeDiff(hi, function(hi, **args))
    return core.std.MergeDiff(function(lo, **args), hi_freq)


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

    if fast:
        sqrt = __quickResample(clip, lambda e: e.std.Expr(["x 5 * x * sqrt"]),
                               input_depth=clip.format.bits_per_sample)
        ret = core.std.MaskedMerge(clip, sqrt, clip.std.PlaneStats().adg.Mask())
    else:
        ret = core.retinex.MSRCP(clip, **msrcp_args) if clip.format.bits_per_sample <= 16 else \
            __quickResample(clip, partial(core.retinex.MSRCP, **msrcp_args), dither_type='none')

    max_value = scale_value(1, 32, clip.format.bits_per_sample, scale_offsets=True, range=Range.FULL)

    tcanny = core.tcanny.TCanny(ret, **tcanny_args).std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])
    expr = core.std.Expr([mask, tcanny], f'x y + {max_value} min')
    return depth(expr, clip.format.bits_per_sample, dither_type='none')
