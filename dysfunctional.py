import vapoursynth as vs
core = vs.core

def STGraino(clip, planes: [0,1,2],
            pre = [1,2,2], 
            str: float = [1,4,4], 
            show_diff = False,
            **kwargs) -> vs.VideoNode:
    """
    KNLMeansCL documentation can be found here: https://github.com/Khanattila/KNLMeansCL/wiki/Filter-description
    STPresso: https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/master/havsfunc.py#L3535
    STPresso args can be parsed directly into this wrapper, i.e;
    grn = dsf.STGraino(b16, planes=[0,1,2], pre=[1,2,2], str = [1.5,4,4], limit = 4, bias = 65, tbias = 50, tthr = 32, show_diff=True)
    -
    TODO: parse knlm args, per-plane show_diff
    """
    from vsutil import split, join, depth as Depth
    from havsfunc import STPresso
    from rgvs import RemoveGrain

    rgb = RGBtoYUV(clip)
    b16 = Depth(rgb, 16)
    p = split(b16)

    if 0 in planes:
        if pre[0] == 1:
            p[0] = p[0].knlm.KNLMeansCL(h=str[0])
        elif pre[0] == 2:
            p[0] = RemoveGrain(p[0], mode=str[0])
    if 1 in planes:
        if pre[0] == 1:
            p[1] = p[1].knlm.KNLMeansCL(h=str[1])
        elif pre[0] == 2:
            p[1] = RemoveGrain(p[1], mode=str[1])
    if 2 in planes:
        if pre[0] == 1:
            p[2] = p[2].knlm.KNLMeansCL(h=str[2])
        elif pre[0] == 2:
            p[2] = RemoveGrain(p[2], mode=str[2])

    dns = join(p)
    dif = core.std.MakeDiff(b16, dns, planes=planes)
    rgn = STPresso(dif, **kwargs, planes=planes)
    mrg = core.std.MergeDiff(dns, rgn, planes=planes)

    if show_diff is True:
        return dif
    else: 
        return Depth(mrg, clip.format.bits_per_sample)


def RGBtoYUV(clip):
    if clip.format.color_family == vs.RGB:
        return core.resize.Point(clip, format=vs.YUV444P16, matrix_s='709', dither_type='error_diffusion')
    else:
        return clip


def FDOG(clip: vs.VideoNode, retinex=True, div=2, bits=16, sigma=1.5, opencl=False) -> vs.VideoNode:
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


def ssimdown(clip, preset=None, width=None, height=None, left=0, right=0, bottom=0, top=0, ar=16 / 9):
    """
    ssimdownscaler wrapper to resize chroma with spline36 and optional (hopefully working) side cropping
    only works with 420 atm since 444 would probably add krigbilateral to the mix
    """
    from vsutil import depth, split, join
    import math
    import urllib.request
    shader = urllib.request.urlopen("https://gist.githubusercontent.com/igv/36508af3ffc84410fe39761d6969be10/raw/ac09db2c0664150863e85d5a4f9f0106b6443a12/SSimDownscaler.glsl")
    shader = shader.read()
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

    y = clip.placebo.Shader(shader_s=shader, width=w, height=h, filter="mitchell") # pretty sure these don't need to be set: , linearize=0, sigmoidize=0)

    # I hope the shifts are correctly set
    u = u.resize.Spline36(w / 2, h / 2, src_left=shift + c[0], src_width=u.width - c[0] - c[1], src_top=c[2], src_height=u.height - c[2] - c[3])
    v = v.resize.Spline36(w / 2, h / 2, src_left=shift + c[0], src_width=u.width - c[0] - c[1], src_top=c[2], src_height=u.height - c[2] - c[3])

    return depth(join([y, u, v]), ind)


