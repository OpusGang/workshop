from functools import partial

import vapoursynth as vs
from vardefunc import RGBPlanes
from vsrgtools import repair
from vstools import normalize_planes
from vsutil import iterate, scale_value

core = vs.core


def rtn(number, multiple):
    return multiple * round(number / multiple)


def lim_blur(clip: vs.VideoNode, radius: float = 3, limit: int = 5) -> vs.VideoNode:
    """
    https://forum.doom9.org/showpost.php?p=1586287&postcount=5
    """
    r_width, r_height = [rtn((x / radius) / 4, 2) for x in (clip.width, clip.height)]
    limit = scale_value(limit, 8, clip.format.bits_per_sample)
    bicubic = dict(filter_param_a=1/3, filter_param_b=1/3), dict(filter_param_a=1, filter_param_b=0)

    resize = core.resize.Bicubic(
        clip, width=r_width, height=r_height, **bicubic[0]
        ).resize.Bicubic(width=clip.width, height=clip.height, **bicubic[1])

    return core.akarin.Expr(
        [clip, resize], expr=[f"x {limit} + y < x {limit} + x {limit} - y > x {limit} - y ? ?"]) 


def sharp(clip: vs.VideoNode, radius: int = 12, planes: list | int = [0, 1, 2]) -> vs.VideoNode:
    """
    output is not the same
    https://forum.doom9.org/showpost.php?p=1595531&postcount=5
    """

    val1 = 1.0  # broken with 1.62
    blur = core.ctmf.CTMF(clip, radius=radius)
    rep = iterate(blur, partial(repair, clip, mode=1), 8)

    expr_string = f"x x y - abs {val1} 2 pow / 1 {val1} / pow {val1} 3 pow * x y - x y - abs {val1} + / * +"
    return core.akarin.Expr([clip, rep], [
        expr_string if i in normalize_planes(clip, planes) else ''
        for i in range(clip.format.num_planes)
    ], opt=True)


def msu(clip: vs.VideoNode) -> vs.VideoNode:
    """
    https://forum.doom9.org/showpost.php?p=1621981&postcount=12
    """
    if clip.format.color_family != vs.RGB:
        raise Exception("clip must be RGB")

    val = scale_value(255, 8, clip.format.bits_per_sample)

    with RGBPlanes(clip) as rgb:
        rgb.B = core.std.Expr([rgb.B, rgb.R], f"x y 2 pow * {val} 2 pow /")
        rgb.G = core.std.Expr([rgb.G, rgb.R], f"x y 2 pow * {val} 2 pow /")
        rgb.R = core.std.Expr([rgb.R],        f"x 3 pow {val} 2 pow /")

    return rgb.clip


def unknown_dnr_2(clip: vs.VideoNode) -> vs.VideoNode:
    """https://forum.doom9.org/showpost.php?p=1618157&postcount=15"""
    return clip


def unknown_dnr_3(clip: vs.VideoNode) -> vs.VideoNode:
    """https://forum.doom9.org/showpost.php?p=1571490&postcount=22"""
    return clip


def mcdegrainsharp(clip: vs.VideoNode) -> vs.VideoNode:
    """
    https://forum.doom9.org/showpost.php?p=1508638&postcount=12
    https://gist.github.com/4re/b5399b1801072458fc80
    """
    return clip


def mix3(
    dark: vs.VideoNode,
    medium: vs.VideoNode,
    bright: vs.VideoNode,
    thrs: list = [24, 56, 128, 160],
    ref: vs.VideoNode = None,
    blur: vs.VideoNode = core.std.BoxBlur
    ) -> vs.VideoNode:
    """https://forum.doom9.org/showpost.php?p=1578966&postcount=2"""

    if ref is None:
        ref = dark

    thrs = [scale_value(i, 8, ref.format.bits_per_sample) for i in (*thrs, 255)]
    ref = blur(ref)

    msk1 = core.std.Expr(ref, [f"x {thrs[0]} < 0 x {thrs[1]} > {thrs[4]} {thrs[4]} {thrs[1]} {thrs[0]} - / x {thrs[0]} - * ? ?"])
    msk2 = core.std.Expr(ref, [f"x {thrs[2]} < 0 x {thrs[3]} > {thrs[4]} {thrs[4]} {thrs[3]} {thrs[2]} - / x {thrs[2]} - * ? ?"])

    return core.std.MaskedMerge(
        dark, medium, msk1).std.MaskedMerge(bright, msk2)


def nonlin_usm(
    clip: vs.VideoNode,
    z: float = 6.0,
    pow: float = 1.6,
    str: float = 1.0,
    rad: float = 9.0,
    ldmp: float = 0.001
    ) -> vs.VideoNode:
    """
    thanks jackoneill
    https://forum.doom9.org/showpost.php?p=1555234&postcount=46
    https://forum.doom9.org/showpost.php?p=1798079&postcount=3
    """
    g_width = int(clip.width / rad / 4 + 0.5) * 4
    g_height = int(clip.height / rad / 4 + 0.5) * 4

    g = core.resize.Bicubic(clip, width=g_width, height=g_height)
    g = core.resize.Bicubic(g, width=clip.width, height=clip.height, filter_param_a=1, filter_param_b=0)

    expression = f"x x y - abs {z} / 1 {pow} / pow {z} * {str} * x y - 2 pow x y - 2 pow {ldmp} + / * x y - x y - abs 0.001 + / * +"

    return core.std.Expr(clips=[clip, g], expr=[expression, ''])
