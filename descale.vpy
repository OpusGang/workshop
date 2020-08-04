from vapoursynth import core
import vsutil
import kagefunc as kgf
import awsmfunc as awsm

src = core.ffms2.Source(r'/path/to/source.mkv')
src = vsutil.depth(src, 32)

y, u, v = kgf.split(src)
height = 762
width = vsutil.get_w(height)

# Bilinear
descale_a = core.descale.Debilinear(y, width, height).resize.Bilinear(1920, 1080)
descale_a = kgf.join([descale_a, u, v])
descale_a = vsutil.depth(descale_a, 8)
descale_a = awsm.FrameInfo(descale_a, "Bilinear")
# Mitchell-Netravali
descale_b = core.descale.Debicubic(y, width, height, b=1/3, c=1/3).resize.Bicubic(1920, 1080, filter_param_a=1/3, filter_param_b=1/3)
descale_b = kgf.join([descale_b, u, v])
descale_b = vsutil.depth(descale_b, 8)
descale_b = awsm.FrameInfo(descale_b, "Mitchell-Netravali")
# Sharp Bicubic
descale_c = core.descale.Debicubic(y, width, height, b=0, c=1).resize.Bicubic(1920, 1080, filter_param_a=0, filter_param_b=1)
descale_c = kgf.join([descale_c, u, v])
descale_c = vsutil.depth(descale_c, 8)
descale_c = awsm.FrameInfo(descale_c, "Sharp Bicubic")
# B-Spline
descale_d = core.descale.Debicubic(y, width, height, b=1, c=0).resize.Bicubic(1920, 1080, filter_param_a=1, filter_param_b=0)
descale_d = kgf.join([descale_d, u, v])
descale_d = vsutil.depth(descale_d, 8)
descale_d = awsm.FrameInfo(descale_d, "B-Spline")
# Catmull-rom
descale_e = core.descale.Debicubic(y, width, height, b=0, c=1/2).resize.Bicubic(1920, 1080, filter_param_a=0, filter_param_b=1/2)
descale_e = kgf.join([descale_e, u, v])
descale_e = vsutil.depth(descale_e, 8)
descale_e = awsm.FrameInfo(descale_e, "Catmull-rom")
# Spline36
descale_f = core.descale.Despline36(y, width, height).resize.Spline36(1920, 1080)
descale_f = kgf.join([descale_f, u, v])
descale_f = vsutil.depth(descale_f, 8)
descale_f = awsm.FrameInfo(descale_f, "Spline36")

src = vsutil.depth(src, 8)
test = core.std.Interleave([src, descale_a, src, descale_b, src, descale_c, src, descale_d, src, descale_e, src, descale_f])
test.set_output()