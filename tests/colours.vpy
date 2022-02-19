import vapoursynth as vs
import vsutil
core = vs.core

import numpy as np

# Authored by ???
# Pretty sure I found this in a gist somewhere, but I cannot remember nor find the original
# My apologies.

def makeline(y, v):
    return core.std.StackHorizontal([core.std.BlankClip(None,1,1,length=240,format=vs.YUV444PS, 
            color=[y,u,v]) for u in np.arange(-0.5,0.5,1/10)])

def makesquare(y):
    return core.std.StackVertical([makeline(y, v) for v in np.arange(0.5,-0.5,-1/10)])

rainbow = makesquare(0.5).resize.Spline36(800,800, format=vs.YUV444P8, dither_type='ordered').set_output()