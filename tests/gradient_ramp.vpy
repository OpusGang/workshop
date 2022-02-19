import vapoursynth as vs
import numpy as np
core = vs.core

def gradientRamp():
    generate = [core.std.BlankClip(width=1, height=1, format=vs.RGBS, color=[n]*3,
                                   length=240, keep=True) for n in np.arange(-0.07, 1, 0.001)]

    stack = core.std.StackHorizontal(generate)
    return core.resize.Bilinear(stack, width=800, height=400)

gradientRamp().fmtc.bitdepth(bits=8, dmode=8).set_output()