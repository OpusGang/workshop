from functools import partial
from typing import Any, Dict

import rgvs as RGToolsVS
import vapoursynth as vs
from jvsfunc.misc import retinex as jretinex
from lvsfunc.scale import gamma2linear, linear2gamma
from vsutil import Range, depth, get_y, scale_value

core = vs.core


class LambdaFN():
    def __call__(
        self, clip: vs.VideoNode, *args: Any, **kwargs: Any
    ) -> vs.VideoNode:
        ...


class Map:
    def GMSD(
        clip: vs.VideoNode,
        ref: vs.VideoNode,
        c: float = 0.0026,
        downsample: bool = False
        ) -> vs.VideoNode:
        """
        https://github.com/WolframRhodium/muvsfunc/blob/master/muvsfunc.py#L3418
        """
        from muvsfunc import _IQA_downsample

        if downsample:
            res = clip.width, clip.height
            clip, ref = [_IQA_downsample(x) for x in (clip, ref)]

        clip1_dx = core.std.Convolution(
            clip, [1, 0, -1, 1, 0, -1, 1, 0, -1], divisor=1, saturate=False)
        clip1_dy = core.std.Convolution(
            clip, [1, 1, 1, 0, 0, 0, -1, -1, -1], divisor=1, saturate=False)
        clip1_grad_squared = core.std.Expr(
            [clip1_dx, clip1_dy], ['x dup * y dup * +'])

        clip2_dx = core.std.Convolution(
            ref, [1, 0, -1, 1, 0, -1, 1, 0, -1], divisor=1, saturate=False)
        clip2_dy = core.std.Convolution(
            ref, [1, 1, 1, 0, 0, 0, -1, -1, -1], divisor=1, saturate=False)
        clip2_grad_squared = core.std.Expr(
            [clip2_dx, clip2_dy], ['x dup * y dup * +'])

        # Compute the gradient magnitude similarity (GMS) map
        quality_map = core.std.Expr(
            [clip1_grad_squared, clip2_grad_squared],
            [f'2 x y * sqrt * {c} + x y + {c} + /'])

        if downsample:
            return core.resize.Lanczos(
                quality_map, width=res[0], height=res[1]
            )

        return quality_map

    def SSIM(
        clip: vs.VideoNode,
        ref: vs.VideoNode,
        downsample: bool = False,
        k1: float = 0.01,
        k2: float = 0.03,
        dynamic_range: float = 1
        ) -> vs.VideoNode:
        """
        https://github.com/WolframRhodium/muvsfunc/blob/master/muvsfunc.py#L3530
        """
        from muvsfunc import _IQA_downsample

        if downsample:
            res = clip.width, clip.height
            clip, ref = [_IQA_downsample(x) for x in (clip, ref)]

        c1 = (k1 * dynamic_range) ** 2
        c2 = (k2 * dynamic_range) ** 2

        fun = partial(core.tcanny.TCanny, sigma=1.5, mode=-1)

        mu1 = fun(clip)
        mu2 = fun(ref)
        mu1_sq = core.std.Expr([mu1], ['x dup *'])
        mu2_sq = core.std.Expr([mu2], ['x dup *'])
        mu1_mu2 = core.std.Expr([mu1, mu2], ['x y *'])
        sigma1_sq_pls_mu1_sq = fun(core.std.Expr([clip], ['x dup *']))
        sigma2_sq_pls_mu2_sq = fun(core.std.Expr([ref], ['x dup *']))
        sigma12_pls_mu1_mu2 = fun(core.std.Expr([clip, ref], ['x y *']))

        if c1 > 0 and c2 > 0:
            expr = f'2 x * {c1} + 2 y x - * {c2} + * z a + {c1} + b c - d e - + {c2} + * /'
            expr_clips = [
                mu1_mu2, sigma12_pls_mu1_mu2, mu1_sq,
                mu2_sq, sigma1_sq_pls_mu1_sq, mu1_sq,
                sigma2_sq_pls_mu2_sq, mu2_sq
                ]
            ssim_map = core.std.Expr(expr_clips, [expr])

        else:
            denominator1 = core.std.Expr(
                [mu1_sq, mu2_sq],
                [f'x y + {c1} +'])
            denominator2 = core.std.Expr(
                [sigma1_sq_pls_mu1_sq, mu1_sq, sigma2_sq_pls_mu2_sq, mu2_sq],
                [f'x y - z a - + {c2} +'])

            numerator1_expr = f'2 z * {c1} +'
            numerator2_expr = f'2 a z - * {c2} +'
            expr = f'x y * 0 > {numerator1_expr} {numerator2_expr} * x y * / x 0 = not y 0 = and {numerator1_expr} x / {1} ? ?'
            ssim_map = core.std.Expr(
                [denominator1, denominator2, mu1_mu2, sigma12_pls_mu1_mu2], [expr])

        if downsample:
            return core.resize.Lanczos(
                ssim_map, width=res[0], height=res[1]
            )

        return ssim_map


def gauss(clip: vs.VideoNode, thresh: int = None) -> vs.VideoNode:
    gauss = core.fmtc.resample(clip, w=clip.width * 2, h=clip.height * 2,
                               kernel='gauss', a1=100)
    return core.fmtc.resample(gauss, w=clip.width, h=clip.height,
                              kernel='gauss', a1=thresh)


def __quickResample(clip: vs.VideoNode,
                    function: LambdaFN,
                    dither_type: str = 'none',
                    input_depth: int = 16) -> vs.VideoNode:

    down = depth(clip, input_depth, dither_type=dither_type)
    filtered = function(down)
    return depth(filtered, clip.format.bits_per_sample,
                 dither_type=dither_type)


def pickFn(function: LambdaFN, functionAlt: LambdaFN, bits: int = 16) -> None:
    return function if bits < 32 else functionAlt


def mvFuncComp(clip: LambdaFN, func: LambdaFN, **func_args) -> vs.VideoNode:
    """basic motion compensation via mvtools"""
    bits = clip.format.bits_per_sample

    mvSuper = pickFn(core.mv.Super, core.mvsf.Super, bits)(clip)

    vectorBck = pickFn(core.mv.Analyse, core.mvsf.Analyze, bits)(mvSuper, isb=True, delta=1, blksize=16, overlap=8)
    vectorFwd = pickFn(core.mv.Analyse, core.mvsf.Analyze, bits)(mvSuper, isb=False, delta=1, blksize=16, overlap=8)
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
    return core.std.Expr(
        [flt, src, blur],
        ['x dup + z - x y min max x y max min', '', ''][:np]
        )


def fastFreqMerge(lo: vs.VideoNode, hi: vs.VideoNode,
                  function: vs.VideoNode = None,
                  **args) -> vs.VideoNode:
    if function is None:
        function = partial(gauss, thresh=6)

    hi_freq = core.std.MakeDiff(hi, function(hi, **args))
    return core.std.MergeDiff(function(lo, **args), hi_freq)


def retinex(clip: vs.VideoNode,
            mask: vs.VideoNode,
            fast: bool = True,
            tcanny_dict: None | dict = None) -> vs.VideoNode:
    """
    maskPre = vsutil.get_y(clip)
    mask = vsmask.edge.Kirsch().get_mask(maskPre)
    edgeKirsch = retinex(clip, mask, fast=True, msrcp_dict=dict(op=5))
    """

    tcanny_args: Dict[str, Any] = dict(sigma=1, mode=1, op=2)
    if tcanny_dict is not None:
        tcanny_args |= tcanny_dict

    if clip.format.num_planes > 1:
        clip, mask = [get_y(x) for x in (clip, mask)]

    if fast:
        ret = __quickResample(
            clip, partial(core.std.Expr, expr=["x 5 * x * sqrt"]),
            input_depth=clip.format.bits_per_sample)
    else:
        ret = jretinex(clip, sigmas=[50, 200, 350], fast=True)

    max_value = scale_value(1, 32, clip.format.bits_per_sample, scale_offsets=True, range=Range.FULL)

    tcanny = core.tcanny.TCanny(ret, **tcanny_args).std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])
    expr = core.std.Expr([mask, tcanny], f'x y + {max_value} min')
    return depth(expr, clip.format.bits_per_sample, dither_type='none')


class toLinear:
    def expr(self, clip: vs.VideoNode) -> vs.VideoNode:
        return gamma2linear(
            clip, curve=vs.TransferCharacteristics.TRANSFER_BT709
        )

    def fmtc(self, clip: vs.VideoNode) -> vs.VideoNode:
        return core.fmtc.transfer(
            clip, transs='1886', transd='linear', bits=32
        )

    def zimg(self, clip: vs.VideoNode) -> vs.VideoNode:
        return core.resize.Bicubic(
            clip, matrix_in_s='709', transfer_in_s='709', transfer_s='linear'
        )

    def sigmoid(self, clip: vs.VideoNode) -> vs.VideoNode:
        return core.fmtc.transfer(
            clip, transs='linear', transd='sigmoid'
        )


class fromLinear:
    def expr(self, clip: vs.VideoNode) -> vs.VideoNode:
        return linear2gamma(
            clip, curve=vs.TransferCharacteristics.TRANSFER_BT709
        )

    def fmtc(self, clip: vs.VideoNode) -> vs.VideoNode:
        return core.fmtc.transfer(
            clip, transs='linear', transd='1886'
        )

    def zimg(self, clip: vs.VideoNode) -> vs.VideoNode:
        return core.resize.Bicubic(
            clip, transfer_in_s='linear', transfer_s='709'
        )

    def sigmoid(self, clip: vs.VideoNode) -> vs.VideoNode:
        return core.fmtc.transfer(
            clip, transs='sigmoid', transd='linear'
        )


def linearize(clip: vs.VideoNode, function: vs.VideoNode = None,
              matrix: int = 709, sigmoid: bool = True,
              sig_c: float = 5.0, sig_t: float = 0.5):

    ref = clip
    if ref.format != vs.RGB:
        clip = core.resize.Bicubic(
            clip, format=vs.RGBS, matrix_in_s=matrix,
            dither_type='error_diffusion')
    # technically this should be 1886 instead of 709
    linear = gamma2linear(
        clip, vs.TransferCharacteristics.TRANSFER_BT709, sigmoid=sigmoid, cont=sig_c, thr=sig_t
    )

    resample = function(linear)

    return linear2gamma(
        resample, vs.TransferCharacteristics.TRANSFER_BT709, sigmoid=sigmoid, cont=sig_c, thr=sig_t
        )
