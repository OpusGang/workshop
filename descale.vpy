import vapoursynth as vs
import vsutil
import kagefunc
core = vs.core

src = core.ffms2.Source('0000.m2ts')
b32 = vsutil.depth(src, 32)

height = 720
width = vsutil.get_w(height)
depth = src.format.bits_per_sample

y, u, v = kagefunc.split(b32)

# Bilinear
descale_a = core.descale.Debilinear(y, width, height).resize.Bilinear(src.width, src.height)
# Mitchell-Netravali
descale_b = core.descale.Debicubic(y, width, height, b=1/3, c=1/3).resize.Bicubic(src.width, src.height, filter_param_a=1/3, filter_param_b=1/3)
# Sharp Bicubic
descale_c = core.descale.Debicubic(y, width, height, b=0, c=1).resize.Bicubic(src.width, src.height, filter_param_a=0, filter_param_b=1)
# B-Spline
descale_d = core.descale.Debicubic(y, width, height, b=1, c=0).resize.Bicubic(src.width, src.height, filter_param_a=1, filter_param_b=0)
# Catmull-rom
descale_e = core.descale.Debicubic(y, width, height, b=0, c=1/2).resize.Bicubic(src.width, src.height, filter_param_a=0, filter_param_b=1/2)
# Spline36
descale_f = core.descale.Despline36(y, width, height).resize.Spline36(src.width, src.height)

out = [b32, descale_a, descale_b, descale_c, descale_d, descale_e, descale_f]

for pos, val in enumerate(out):
    val = core.std.ShufflePlanes([val, b32], [0, 1, 2], vs.YUV).set_output(pos)
