import vapoursynth as vs
core = vs.core

from typing import Callable, Optional


def coolgrain(clip: vs.VideoNode, strength: list[Optional[int], Optional[int]] = [5,0], radius: int = 3, luma_scaling: float = 12.0,
              cutoff: Optional[float] = None, invert: bool = False) -> vs.VideoNode:
    from vsutil import depth, scale_value
    
    if isinstance(strength, int): strength=[strength, strength]

    bits = clip.format.bits_per_sample
    if clip.format.bits_per_sample != 32: clip = depth(clip, 32)
    
    # generate and process at half resolution
    blank = core.std.BlankClip(clip, width=clip.width / 2, height=clip.height / 2, color=[scale_value(127, 8, 32)]*clip.format.num_planes)
    
    grain = core.grain.Add(blank, var=strength[0], uvar=strength[1], seed=444)
    average = core.misc.AverageFrames(grain, weights=[1] * (2 * radius + 1)) 
    
    # Lover CPU usage via placebo
    diff = core.std.MakeDiff(blank, average).placebo.Resample(width=clip.width, height=clip.height, filter='robidoux', param1=0, param2=0)

    merge = core.std.Expr([clip, diff], ["x y +"])

    if luma_scaling > 0:
        mask = core.adg.Mask(core.std.PlaneStats(clip))

        if invert is True: mask = core.std.Invert(mask)
        merge = core.std.MaskedMerge(clip, merge, mask)
        
    # clip just above legal range
    if cutoff is not None:
        merge = core.std.MaskedMerge(clip, merge, core.std.Binarize(clip, scale_value(cutoff, 8, 32, scale_offsets=True)))
        
    return depth(merge, bits, dither_type='none')


def CoolDegrainSF(clip: vs.VideoNode, tr: int = 1, thSAD: int = 48, planes: list[int] = [0,1,2], blksize: int = None, overlap: int = None,
                  pel: int = None, recalc: bool = False, pf: Optional[Callable[[vs.VideoNode, vs.VideoNode], vs.VideoNode]] = None) -> vs.VideoNode:
    from vsutil import depth, plane
    from zzfunc.util import vs_to_mv
    import rgvs
        
    bits = clip.format.bits_per_sample
    if clip.format.bits_per_sample != 32:
        clip = depth(clip, 32)
    
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
        
    if tr not in range(1, 25):
        raise ValueError('tr must be between 1 and 24.')
    
    pfclip = pf if pf is not None else clip
    
    super = core.mvsf.Super(clip, pel=pel, sharp=2, rfilter=4)
    analyse = core.mvsf.Analyze(super, radius=tr, isb=True, overlap=overlap, blksize=blksize)

    # This seems useless
    if recalc is True:
        hoverlap = overlap // 2
        hblksize = blksize // 2
        hthsad = thSAD // 2
        
        prefilt = rgvs.removegrain(clip, mode=4, planes=0)
        super_r = core.mvsf.Super(prefilt, pel=pel, sharp=2, rfilter=4)
        analyse = core.mvsf.Recalculate(super_r, analyse, overlap=overlap, blksize=blksize)
    
    filter = core.mvsf.Degrain(clip, super, analyse, thsad=thSAD, plane=vs_to_mv(planes))

    return depth(filter, bits)


def horribleDNR(clip: vs.VideoNode, prefilter: Optional[Callable[..., vs.VideoNode]] = None,
                postfilter: Optional[Callable[..., vs.VideoNode]] = None, radius: int = 2) -> vs.VideoNode:
    from G41Fun import DetailSharpen
    
    if prefilter is None: prefilter = lambda x: core.bilateral.Bilateral(x, sigmaS=0.8, sigmaR=0.05)
    if postfilter is None: postfilter = lambda x: DetailSharpen(x)

    protEdges = core.std.Prewitt(clip).std.Maximum()
    removeNoise = prefilter(clip)
    maskEdges = core.std.MaskedMerge(removeNoise, clip, protEdges)

    storeNoise = core.std.MakeDiff(clip, maskEdges, planes=[0,1,2])
    avgNoise = core.misc.AverageFrames(storeNoise, weights=[1] * (2 * radius + 1))
    sharpNoise = core.std.MaskedMerge(avgNoise, postfilter(avgNoise), maskEdges)
    
    # contrasharp expr stolen from havsfunc
    neutral = 1 << (clip.format.bits_per_sample - 1)
    limitSharp = core.std.Expr([clip, sharpNoise], expr=[f'x {neutral} - abs y {neutral} - abs < x y ?'])

    return core.std.MergeDiff(limitSharp, maskEdges)


def bm3dGPU(clip: vs.VideoNode, sigma: int = 3, ref: Optional[Callable[..., vs.VideoNode]] = None,
            profile: Optional[str] = None, fast: bool = False) -> vs.VideoNode:
    from vsutil import get_y, depth
    from havsfunc import SMDegrain

    # CUDA implementation only has a few settings implemented as of writing
    if profile is None   : profile = 'fast'
    if profile == 'fast' : block_step, bm_range, radius, ps_num, ps_range = 7,   7,  1,   2,  5
    if profile == 'lc'   : block_step, bm_range, radius, ps_num, ps_range = 5,   9,  2,   2,  5
    if profile == 'np'   : block_step, bm_range, radius, ps_num, ps_range = 3,  12,  3,   2,  6
    if profile == 'high' : block_step, bm_range, radius, ps_num, ps_range = 2,  16,  4,   2,  8
    if profile == 'vn'   : block_step, bm_range, radius, ps_num, ps_range = 6,  12,  4,   2,  6
    
    # This is dumb, but it saves some CPU cycles - havsfunc wrapper may be inefficient 
    if ref is None:
        def ref(clip: vs.VideoNode) -> vs.VideoNode:
            luma = get_y(clip)
            if luma.format.bits_per_sample != 16: luma = depth(luma, 16)
            return core.std.ShufflePlanes([SMDegrain(luma, tr=2, thSAD=300, RefineMotion=True, plane=0), luma], [0,0,0], vs.YUV)

    opp32 = core.bm3d.RGB2OPP(core.resize.Point(clip, format=vs.RGBS, matrix_in_s='709', range_in_s='limited', range_s='full'), sample=1)
    reference = core.bm3d.RGB2OPP(core.resize.Point(ref(clip), format=vs.RGBS, matrix_in_s='709'), sample=1)

    denoise = core.bm3dcuda_rtc.BM3D(opp32, ref=reference, sigma=[sigma, 0, 0], fast=fast, extractor_exp=8, transform_2d_s='DCT', transform_1d_s='DCT',
    block_step=block_step, bm_range=bm_range, radius=radius, ps_num=ps_num, ps_range=ps_range)
    if radius > 0: 
        denoise = core.bm3d.VAggregate(denoise, radius=radius, sample=1)

    shuffle = core.std.ShufflePlanes([denoise, opp32], [0,1,2], vs.YUV)
    rgb32 = core.bm3d.OPP2RGB(shuffle, sample=1)
    return core.resize.Point(rgb32, format=clip.format, matrix_s='709', range_s='limited', range_in_s='full', dither_type='error_diffusion')


def FDOG(clip: vs.VideoNode, retinex: bool = True, div: list[float] = 2, bits: int = 16, sigma: float = 1.5, opencl: bool = False) -> vs.VideoNode:
    from vsutil import depth, get_depth, get_y

    if isinstance(div, int): div=[div, div]
    
    def __FDOG(clip: vs.VideoNode) -> vs.VideoNode:    
        lma = depth(get_y(clip), bits)
        gx = core.std.Convolution(lma, [1, 1, 0, -1, -1, 2, 2, 0, -2, -2, 3, 3, 0, -3, -3, 2, 2, 0, -2, -2, 1, 1, 0, -1, -1], divisor=div[0], saturate=False)
        gy = core.std.Convolution(lma, [-1, -2, -3, -2, -1, -1, -2, -3, -2, -1, 0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 1, 2, 3, 2, 1], divisor=div[1], saturate=False)
        return core.std.Expr([gx, gy], 'x dup * y dup * + sqrt')

    def __retinex_fdog(clip: vs.VideoNode, sigma=sigma, sigma_rtx=[50, 200, 350], opencl=opencl) -> vs.VideoNode:
        tcanny = core.tcanny.TCannyCL if opencl else core.tcanny.TCanny
        luma = get_y(clip)
        luma = depth(luma, bits)
        fdog = __FDOG(luma)
        max_value = 1 if luma.format.sample_type == vs.FLOAT else (1 << get_depth(luma)) - 1
        ret = depth(core.retinex.MSRCP(depth(luma, 16), sigma=sigma_rtx, upper_thr=0.005), bits)
        tcanny = tcanny(ret, mode=1, sigma=sigma).std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])
        return core.std.Expr([fdog, tcanny], f'x y + {max_value} min')

    if retinex is True: return depth(__retinex_fdog(clip), clip.format.bits_per_sample, dither_type='none')
    else: return depth(__FDOG(clip), clip.format.bits_per_sample, dither_type='none')


def bbcfcalc(clip, top=0, bottom=0, left=0, right=0, radius=None, thr=32768, blur=999):
    from vsutil import depth
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
    from vsutil import scale_value, split, join
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


def ssimdown(clip: vs.VideoNode, preset: Optional[str] = None, width: Optional[int] = None, height: Optional[int] = None,
             left: int = 0, right: int = 0, bottom: int = 0, top: int = 0, ar: str = 16 / 9, 
             shader_path: Optional[str] = None, shader_str = Optional[Callable[..., str]]) -> vs.VideoNode:
    """
    ssimdownscaler wrapper to resize chroma with spline36 and optional (hopefully working) side cropping
    only works with 420 atm since 444 would probably add krigbilateral to the mix
    """
    from vsutil import depth, split, join
    import math

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

    if preset == width == height == None:
        preset = 1080

    if preset:
        if clip.width / clip.height > ar:
            return ssimdown(clip, width=ar * preset, left=left, right=right, top=top, bottom=bottom)
        else:
            return ssimdown(clip, height=preset, left=left, right=right, top=top, bottom=bottom)

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

    y = clip.placebo.Shader(shader_s=shader, width=w, height=h, filter="mitchell") # pretty sure these don't need to be set: , linearize=0, sigmoidize=0)

    # I hope the shifts are correctly set
    u = u.resize.Spline36(w / 2, h / 2, src_left=shift + c[0], src_width=u.width - c[0] - c[1], src_top=c[2], src_height=u.height - c[2] - c[3])
    v = v.resize.Spline36(w / 2, h / 2, src_left=shift + c[0], src_width=v.width - c[0] - c[1], src_top=c[2], src_height=v.height - c[2] - c[3])

    return depth(join([y, u, v]), ind)