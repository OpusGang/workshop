import os
from weakref import ref
import requests
import matplotlib.pyplot as plt
import numpy as np
import csv
import seaborn
import pandas as pd
import plotly.graph_objects as go
import warnings

from vstools import fallback, mod_x, vs, core, MatrixT, Matrix, clip_async_render, merge_clip_props
from awsmfunc import run_scenechange_detect, SceneChangeDetector
from muvsfunc import SSIM, GMSD
from mvsfunc import PlaneCompare, PlaneStatistics

from enum import Enum
from functools import partial
from itertools import cycle
from scipy.signal import savgol_filter
from scipy.stats import hmean, pmean
from statistics import geometric_mean, mean, median


class ReductionMode:
    class Crop:
        def __init__(self, percentage):
            self.percentage = percentage
            
    class Downsample:
        def __init__(self, percentage):
            self.percentage = percentage

    class Fun:
        pass


class ColourSpace(Enum):
    YCBCR = 0
    RGB = 1


class Complex(Enum):
    PSNR = (0, ['psnr_y', 'psnr_cb', 'psnr_cr'])
    PSNR_HVS = (1, ['psnr_hvs', 'psnr_hvs_y', 'psnr_hvs_cb', 'psnr_hvs_cr'])
    SSIM = (2, ['SSIM_Y', 'SSIM_Cb', 'SSIM_Cr'])
    SSIM_MS = (3, ['float_ms_ssim'])
    CIEDE = (4, ['ciede2000'])
    SSIMULACRA1 = (5, ['_SSIMULACRA1'])
    SSIMULACRA2 = (6, ['_SSIMULACRA2'])
    BUTTER = (7, ['_FrameButteraugli'])
    GMSD = (8, ['GMSD_Y', 'GMSD_Cb', 'GMSD_Cr'])
    WADIQAM = (9, ['Frame_WaDIQaM_FR'])

    def __init__(self, value, prop):
        self._value_ = value
        self.prop = prop

    def __int__(self):
        return self.value

    def __iter__(self):
        return iter(self.prop)


class Simple(Enum):
    MAE = (0, ['PlaneMAE_Y', 'PlaneMAE_Cb', 'PlaneMAE_Cr'])
    RMSE = (1, ['PlaneRMSE_Y', 'PlaneRMSE_Cb', 'PlaneRMSE_Cr'])
    COVARIANCE = (2, ['PlaneCov_Y', 'PlaneCov_Cb', 'PlaneCov_Cr'])
    CORRELATION = (3, ['PlaneCorr_Y', 'PlaneCorr_Cb', 'PlaneCorr_Cr'])

    def __init__(self, value, prop):
        self._value_ = value
        self.prop = prop

    def __int__(self):
        return self.value

    def __iter__(self):
        return iter(self.prop)


class NoRef(Enum):
    CAMBI = (0, ['CAMBI'])
    WADIQAM = (1, ['Frame_WaDIQaM_NR'])

    def __init__(self, value, prop):
        self._value_ = value
        self.prop = prop

    def __int__(self):
        return self.value

    def __iter__(self):
        return iter(self.prop)


class Metric:
    class FullRef:
        Simple = Simple
        Complex = Complex
        NoRef = NoRef


class Smooth:
    @staticmethod
    def Savgol(polyorder: int = 3):
        # FIX THIS YOU FUCKING RETARD
        def fun(data: list[int | float], window_length: int):
            return savgol_filter(data, 10, polyorder)

        fun.__name__ = 'Savitzky-Golay'
        return fun

    @staticmethod
    def HMean():
        f = partial(hmean)
        f.__name__ = 'Harmonic mean'
        return f

    @staticmethod
    def PMean(p: int = 1):
        f = partial(pmean, p=p)
        f.__name__ = 'Power mean'
        return f

    @staticmethod
    def GMean():
        f = partial(geometric_mean)
        f.__name__ = 'Geometric Mean'
        return f

    @staticmethod
    def AMean():
        f = partial(mean)
        f.__name__ = 'Arithmetic Mean'
        return f
    
    @staticmethod
    def Median():
        f = partial(median)
        f.__name__ = 'Median'
        return f


class CSVPropThing:
    def __init__(self, filepath):
        self.filepath = filepath
        self.props_from_csv = []
        self.scenechanges = []
        self.palette = seaborn.color_palette('pastel', as_cmap=True)
        self.colours = cycle(self.palette)
        self.scene_data = []
        self.read_data = []
        self.frames = None
        self.window_size = None
        self.ranges = None
        
        url = 'https://raw.githubusercontent.com/h4pZ/rose-pine-matplotlib/main/themes/rose-pine-dawn.mplstyle' # noqa
        style_file = 'rose-pine.mplstyle'

        if not os.path.exists(style_file):
            response = requests.get(url)
            with open(style_file, 'w') as f:
                f.write(response.text)
        
        plt.style.use(style_file)

    def _rolling_average(self, data: list[int | float], window: int, func):
        if func.__name__ == 'Savitzky-Golay':
            return func(data, window)

        data = pd.Series(data)
        return data.rolling(window).apply(func, raw=True)

    def _fix_inf(self, numbers):
        try:
            max_value = max(num for num in numbers if num != float('inf'))
        except ValueError:
            return numbers

        return [num if num != float('inf') else max_value for num in numbers]

    def _group_by_scene(self, prop: str | list[str] | Metric = Metric.FullRef.Complex.PSNR):

        with open('scenechanges.txt', 'r') as f:
            ranges = [int(line) for line in f]

        df = pd.read_csv(self.filepath)

        start_end = [[ranges[i], ranges[i+1]-1] for i in range(len(ranges)-1)]
        scores = [df.iloc[start:end][prop].median() for (start,end) in start_end]

        self.ranges = start_end
        self.scenechanges = scores
        return self.scenechanges

    def _read(self, prop: str | list[str] | Metric = Metric.FullRef.Complex.PSNR):
        tmp = []
        
        with open(self.filepath, mode="r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                _prop = float(row[f'{prop}'])
                tmp.append(_prop)

        fix_inf = self._fix_inf(tmp)
        off = np.percentile(fix_inf, 95)
        fix_out = [i if i <= off else off for i in fix_inf]

        return fix_out
    
    def write(
        self,
        clip: vs.VideoNode, scenechange_clip: vs.VideoNode = None,
        async_requests: int = 1, scenechange: bool = True,
        overwrite: bool = False
    ):
        if os.path.exists(self.filepath) and overwrite is False:
            raise ValueError("File exists")

        if scenechange:     # change this so scenechange and props are collected at the same time
            run_scenechange_detect(
                clip=fallback(scenechange_clip, clip),
                tonemap=False, output='scenechanges.txt',
                detector=SceneChangeDetector.AvScenechange
                )

        data = clip_async_render(
            clip, outfile=None, progress='Getting frame props...',
            callback=lambda _, f: f.props.copy(), async_requests=async_requests # noqa
            )

        for item in data:
            item["_PictType"] = item["_PictType"].decode("utf-8")

        with open(self.filepath, mode="w", newline="") as csvfile:
            fieldnames = list(data[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in data:
                writer.writerow(row)

    def histogram(self, prop: str | list[str] | Metric = Metric.FullRef.Complex.PSNR):
        if isinstance(prop, str):
            prop = [prop]

        data = [self._read(prop=p) for p in prop]

        for p, d in zip(prop, data):
            plt.hist(d, 30, color=next(self.colours), label=f'{p}', alpha=0.5, stacked=True)

        plt.xlabel('score')
        plt.ylabel('count')

        plt.legend(loc='upper right')
        plt.show()

    def plot(
        self,
        prop: str | list[str] | Metric = Metric.FullRef.Complex.PSNR,
        per_scene: bool = True,
        smoothmode: list[Smooth] = [Smooth.Savgol(polyorder=3), Smooth.HMean()]
    ):
        if isinstance(prop, str):
            prop = [prop]
        # broken lol
        if isinstance(smoothmode, Smooth):
            smoothmode = [smoothmode]

        self.scene_data = [self._group_by_scene(prop=p) for p in prop]

        self.read_data = [self._read(prop=p) for p in prop]

        data = self.scene_data if per_scene else self.read_data

        if not per_scene:
            warnings.warn("Loading may be slow for large datasets")

        self.frames = list(range(len(data[0])))
        self.window_size = int(0.10 * len(data[0]))

        fig = go.Figure()

        for idx, (d, p) in enumerate(zip(data, prop)):
            fig.add_trace(go.Scattergl(
                x=self.frames,
                y=d,
                legendgroup="raw",
                legendgrouptitle_text=f"Raw data",
                mode='markers',
                name=f'{p}',
                text=self.ranges,
                customdata=[[f'{p}']] * len(self.frames),
                hovertemplate='%{customdata} = %{y}<br>scene = %{x}<br>frames = %{text}<extra></extra>'
            ))

            for method in smoothmode:
                fig.add_trace(go.Scatter(
                x=self.frames,
                y=self._rolling_average(d, self.window_size, method),
                visible=True if method.__name__ == 'Savitzky-Golay' else 'legendonly',
                legendgroup=f"{method}",
                legendgrouptitle_text=f"{method.__name__}",
                legendrank=1001,
                mode='lines',
                marker=dict(size=6),
                name=f'{p}',
                customdata=[[f'{p}']] * len(self.frames),
                text=[method.__name__] * len(self.frames),
                hovertemplate='%{customdata} = %{y}<br>scene = %{x}<br>method: %{text}<extra></extra>'
                ))

        ### PRINT EXISTING PROPS ON ERROR???
        ### VALUEERROR(PROP 'Y_SSIM' DOES NOT EXIST)
        ### VALID PROPS = [_FRAMENUM, _FRAMETYPE, _CAMBI]
        ### TYPERROR ONLY INT AND FLOAT TYPE ACCEPTED
        fig.update_layout(
            template='seaborn',
            legend_title="Data points",
            xaxis_title="frames",
            yaxis_title="score",
            xaxis=dict(
                rangeslider=dict(
                    visible=True
                    )
                ),
            yaxis=dict(fixedrange=False),
            legend=dict(groupclick="toggleitem")
            )

        fig.show()
    
    
def _prop_transfer(
    n: int, f: list[vs.VideoFrame],
    clip: vs.VideoNode,
    src_prop: str = 'PlaneSSIM',
    out_prop: str = 'SSIM'
) -> vs.VideoFrame:
    if clip.format.color_family == vs.YUV:
        postfix = ['_Y', '_Cb', '_Cr']
    else:
        postfix = ['_R', '_G', '_B']

    props = {}
    for i in range(3):
        prop_value = f[i].props[f'{src_prop}']
        props[f"{out_prop}{postfix[i]}"] = prop_value

    return clip.std.SetFrameProps(**props)


def calc_diff(
    reference: vs.VideoNode,
    distorted: vs.VideoNode,
    reduce: ReductionMode | None = ReductionMode.Fun(),
    metric: Metric = Metric.FullRef.Complex.SSIM_MS,
    matrix: MatrixT = Matrix.BT709,
    model_path: str = None,
    **args
) -> vs.VideoNode:
    
    if isinstance(reduce, ReductionMode.Crop):

        crop_left_right = int(reference.width * (reduce.percentage / 2) / 100)
        crop_top_bottom = int(reference.height * (reduce.percentage / 2) / 100)

        crop_left_right, crop_top_bottom = [
            i + (4 - i % 4) % 4 for i in [crop_left_right, crop_top_bottom]
        ]

        reference, distorted = [
            clip.std.Crop(
                left=crop_left_right,
                right=crop_left_right,
                top=crop_top_bottom,
                bottom=crop_top_bottom
                ) for clip in (reference, distorted)
        ]

    elif isinstance(reduce, ReductionMode.Downsample):

        new_width = reference.width - 2 * int(reference.width * (reduce.percentage / 2) / 100)
        new_height = reference.height - 2 * int(reference.height * (reduce.percentage / 2) / 100)

        new_width -= new_width % 4
        new_height -= new_height % 4

        reference, distorted = [
            clip.resize.Spline64(new_width, new_height)
            for clip in (reference, distorted)
        ]

    if isinstance(reduce, ReductionMode.Fun):

        for clip in [reference, distorted]:

            crop_width = clip.width // 2
            crop_height = clip.height // 2

            clips = []

            corners = [
                (0, crop_width, 0, crop_height),  # top left
                (crop_width, 0, 0, crop_height),  # top right
                (crop_width, 0, crop_height, 0),  # bottom right
                (0, crop_width, crop_height, 0)   # bottom left
            ]

            for corner in corners:
                cropped_clip = clip[::4].std.Crop(
                    left=corner[0], right=corner[1],
                    top=corner[2], bottom=corner[3]
                )
                clips.append(cropped_clip)

            if clip is reference:
                reference = core.std.Interleave(clips)
            else:
                distorted = core.std.Interleave(clips)

    if metric in (
        Metric.FullRef.Complex.PSNR,
        Metric.FullRef.Complex.PSNR_HVS,
        Metric.FullRef.Complex.SSIM,
        Metric.FullRef.Complex.SSIM_MS,
        Metric.FullRef.Complex.GMSD
    ):
        if reference.format.color_family != vs.YUV:
            _reference, _distorted = [
                core.resize.Spline64(i, format=vs.YUV420P12, matrix=matrix, dither_type='error_diffusion')
                for i in (reference, distorted)
            ]
        else:
            _reference, _distorted = reference, distorted
            
        if metric == Metric.FullRef.Complex.SSIM:
            ssim = [SSIM(clip1=_reference, clip2=_distorted, plane=i) for i in range(3)]
            measure = _reference.std.FrameEval(
                prop_src=ssim, eval=partial(_prop_transfer, clip=reference, src_prop='PlaneSSIM', out_prop='SSIM'))

        elif metric == Metric.FullRef.Complex.GMSD:
            gmsd = [GMSD(clip1=_reference, clip2=_distorted, plane=i) for i in range(3)]
            measure = _reference.std.FrameEval(
                prop_src=gmsd, eval=partial(_prop_transfer, clip=_reference, src_prop='PlaneGMSD', out_prop='GMSD'))
        else:
            measure = core.vmaf.Metric(reference=_reference, distorted=_distorted, feature=metric.value)

    elif metric in (
        Metric.FullRef.Complex.SSIMULACRA1,
        Metric.FullRef.Complex.SSIMULACRA2,
        Metric.FullRef.Complex.BUTTER,
        Metric.FullRef.Complex.WADIQAM
    ):

        if reference.format.color_family != vs.RGB:
            _reference, _distorted = [
                core.resize.Spline64(i, format=vs.RGB48, matrix_in=matrix, dither_type='error_diffusion')
                for i in (reference, distorted)
            ]
        else:
            _reference, _distorted = reference, distorted
            
        if metric == Metric.FullRef.Complex.SSIMULACRA1:
            measure = core.julek.SSIMULACRA(reference=_reference, distorted=_distorted, feature=1, **args)
            
        elif metric == Metric.FullRef.Complex.SSIMULACRA2:
            measure = core.julek.SSIMULACRA(reference=_reference, distorted=_distorted, feature=0, **args)
            
        elif metric == Metric.FullRef.Complex.BUTTER:
            measure = core.julek.Butteraugli(reference=_reference, distorted=_distorted, **args)

        elif metric == Metric.FullRef.Complex.WADIQAM:
            from vs_wadiqam_chainer import wadiqam_fr

            if model_path is None:
                raise ValueError("model_path is required for WADIQAM")
            
            rw = [mod_x(i, 32) for i in (_reference.height, _reference.width)]            
            _reference, _distorted = [c.resize.Spline64(rw[0], rw[1]) for c in (_reference, _distorted)]

            measure = wadiqam_fr(_reference, _distorted, model_folder_path=model_path, dataset='tid', top='patchwise', max_batch_size=2040)

        measure = measure.std.SetFrameProp(prop='_Matrix', intval=matrix)


    elif metric in Metric.FullRef.Simple:

        enum_map = {
            Simple.MAE: ('mae', 'PlaneMAE'),
            Simple.RMSE: ('rmse', 'PlaneRMSE'),
            Simple.COVARIANCE: ('cov', 'PlaneCov'),
            Simple.CORRELATION: ('corr', 'PlaneCorr')
        }

        _metrics = {
            'mae':  False,
            'rmse': False,
            'cov':  False,
            'corr': False,
            'psnr': False,
        }

        key, prop = enum_map[metric]
        if key in _metrics:
            _metrics[key] = True    
        
            simple = [PlaneCompare(clip1=reference, clip2=distorted, plane=i, **_metrics) for i in range(3)]
            measure = reference.std.FrameEval(
                prop_src=simple, eval=partial(_prop_transfer, clip=reference, src_prop=f'{prop}', out_prop=f'{prop[5:]}'))

    else:
        measure = core.std.PlaneStats(reference, distorted, **args)

    print(f'internal res: {measure.width, measure.height}')
    return merge_clip_props(reference, measure)
