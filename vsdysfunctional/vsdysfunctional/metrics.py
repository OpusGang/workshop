import requests
import matplotlib.pyplot as plt
import numpy as np
import csv
import seaborn

from vstools import vs, core, MatrixT, Matrix, clip_async_render, merge_clip_props
from awsmfunc import run_scenechange_detect, SceneChangeDetector
from muvsfunc import SSIM, GMSD

from enum import Enum
from functools import partial
from itertools import cycle
import pandas as pd


class Statistic(Enum):
    MEAN = 'mean'
    GEOMETRIC_MEAN = 'geometric_mean'
    GROUPED_MEDIAN = 'grouped_median'
    HARMONIC_MEAN = 'harmonic_mean'


class Metric(Enum):
    PSNR = (0, ['psnr_y', 'psnr_cb', 'psnr_cr'])
    PSNR_HVS = (1, ['psnr_hvs', 'psnr_hvs_y', 'psnr_hvs_cb', 'psnr_hvs_cr'])
    SSIM = (2, ['Y_SSIM', 'Cb_SSIM', 'Cr_SSIM'])
    SSIM_MS = (3, ['float_ms_ssim'])
    CIEDE = (4, ['ciede2000'])
    SSIMULACRA1 = (5, ['_SSIMULACRA1'])
    SSIMULACRA2 = (6, ['_SSIMULACRA2'])
    BUTTER = (7, ['_FrameButteraugli'])
    GMSD = (8, ['Y_GMSD', 'Cb_GMSD', 'Cr_GMSD'])

    # todo calc psnr from (0.8 * PSNR_Y) + (0.1 * PSNR_Cb) + (0.1 * PSNR_Cr)
    # _psnr = lambda psnr: (0.8 * psnr[0]) + (0.1 * psnr[1]) + (0.1 * psnr[2])
    def __init__(self, value, prop):
        self._value_ = value
        self.prop = prop

    def __int__(self):
        return self.value

    def __iter__(self):
        return iter(self.prop)


def calculate_mean(
    y: list[float],
    window_size: int,
    statistic: Statistic
) -> list[float]:

    result = []
    y = np.array(y)

    for i, in np.ndindex(y.shape):
        start = max(0, i - window_size)
        end = min(len(y), i + window_size + 1)
        window = y[start:end]

        if statistic == Statistic.MEAN:
            result.append(sum(window) / len(window))
        elif statistic == Statistic.GEOMETRIC_MEAN:
            result.append(np.prod(window) ** (1 / len(window)))
        elif statistic == Statistic.GROUPED_MEDIAN:
            result.append(np.median(window))
        elif statistic == Statistic.HARMONIC_MEAN:
            result.append(len(window) / np.sum(1.0 / window))

    return result


class CSVPropThing:
    def __init__(self, filepath):
        self.filepath = filepath

        url = 'https://raw.githubusercontent.com/h4pZ/rose-pine-matplotlib/main/themes/rose-pine-dawn.mplstyle' # noqa
        style_file = '/tmp/rose-pine.mplstyle'
        response = requests.get(url)
        with open(style_file, 'w') as f:
            f.write(response.text)

        plt.style.use(style_file)

    def _group_by_scene(self, prop: str | list[str] = 'psnr_y'):
        df = pd.read_csv(self.filepath)

        with open('scenechanges.txt', 'r') as f:
            ranges = [int(line) for line in f.read().splitlines()]

        scores = []

        for i in range(len(ranges) - 1):
            start = ranges[i]
            end = ranges[i + 1] - 1

            df_subset = df.iloc[start:end]

            mean = df_subset[f'{prop}'].median()

            scores.append(mean)

        return scores

    def _read(self, prop: str | list[str] = 'psnr_y'):
        data_from_csv = []

        with open(self.filepath, mode="r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                _prop = float(row[f'{prop}'])
                data_from_csv.append(_prop)

        return data_from_csv
    
    def write(self, clip: vs.VideoNode, async_requests: int = 1, scenechange: bool = True):
        if scenechange:
            run_scenechange_detect(
                clip,
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

    def plot(self, prop: str | list[str] = 'psnr_y', per_scene: bool = True):
        if isinstance(prop, str):
            prop = [prop]

        if per_scene:
            data = [self._group_by_scene(prop=p) for p in prop]
        else:
            data = [self._read(prop=p) for p in prop]

        frames = range(len(data[0]))

        window_size = int(0.05 * len(data[0]))

        _colours = seaborn.color_palette('pastel', as_cmap=True)
        _palette = cycle(_colours)

        for idx, (d, p) in enumerate(zip(data, prop)):
            palette = next(_palette)

            mean = [calculate_mean(d, window_size, statistic) for statistic in Statistic]

            plt.scatter(frames, d, label=p, color=palette, linewidth=0.1)

            plt.plot(
                frames, mean[3],
                linestyle='dashed', color=palette,
                label=f'{p} Harmonic Mean'
                )

        plt.xlabel('scene' if per_scene else 'frame')
        plt.ylabel('score')

        if per_scene:
            with open('scenechanges.txt', 'r') as f:
                numbers = [int(line) for line in f.read().splitlines()]

            ranges = []
            result = []

            for i in range(len(numbers) - 1):

                start = numbers[i]
                end = numbers[i + 1]

                if end >= start:
                    ranges.append(range(start, end))

            numbers = [list(r) for r in ranges]

            for num_list in numbers:
                range_str = str(num_list[0]) + "-" + str(num_list[-1])
                result.append(range_str)

            plt.xticks(ticks=[x for x in range(len(result))], labels=result)
            plt.xticks(rotation=-45, fontsize=8)

        plt.legend(loc='upper right')
        plt.show()

    def histogram(self, prop: str | list[str] = 'psnr_y'):
        if isinstance(prop, str):
            prop = [prop]

        data = [self._read(prop=p) for p in prop]

        _colours = seaborn.color_palette('pastel', as_cmap=True)
        _palette = cycle(_colours)

        for p, d in zip(prop, data):
            plt.hist(d, 30, color=next(_palette), label=f'{p}', alpha=0.5, stacked=True)

        plt.xlabel('score')
        plt.ylabel('count')

        plt.legend(loc='upper right')
        plt.show()
        
    def kde(self, prop: str | list[str] = 'psnr_y'):
        if isinstance(prop, str):
            prop = [prop]

        data = [self._read(prop=p) for p in prop]

        _colours = seaborn.color_palette('pastel')
        _palette = cycle(_colours)

        for p, d in zip(prop, data):
            seaborn.kdeplot(d, color=next(_palette), label=f'{p}')

        plt.xlabel('score')
        plt.ylabel('count')

        plt.legend(loc='upper right')
        plt.show()


def _PlaneSSIMTransfer(
    n: int, f: list[vs.VideoFrame],
    clip: vs.VideoNode,
    src_prop: str = 'PlaneSSIM',
    out_prop: str = 'SSIM'
) -> vs.VideoFrame:

    props = {}
    for i in range(3):
        prop_value = f[i].props[f'{src_prop}']
        prefix = ['Y_', 'Cb_', 'Cr_'][i]
        props[f"{prefix}{out_prop}"] = prop_value

    return clip.std.SetFrameProps(**props)


def metrics(
    reference: vs.VideoNode,
    distorted: vs.VideoNode,
    metric: Metric = Metric.SSIM_MS,
    matrix: MatrixT = Matrix.BT709,
    **args
) -> vs.VideoNode:

    if metric in (Metric.PSNR, Metric.PSNR_HVS, Metric.SSIM, Metric.SSIM_MS, Metric.GMSD):
        if reference.format.color_family != vs.YUV:
            _reference, _distorted = [core.resize.Spline64(
                i, format=vs.YUV420P12, matrix=matrix, dither_type='error_diffusion'
                ) for i in (reference, distorted)]
        else:
            _reference, _distorted = reference, distorted

        if metric == Metric.SSIM:
            ssim = [SSIM(clip1=_reference, clip2=_distorted, plane=i) for i in range(3)]
            measure = core.std.FrameEval(
                _reference, prop_src=ssim,
                eval=partial(_PlaneSSIMTransfer, clip=reference, src_prop='PlaneSSIM', out_prop='SSIM')
                )

        elif metric == Metric.GMSD:
            gmsd = [GMSD(clip1=_reference, clip2=_distorted, plane=i) for i in range(3)]
            measure = core.std.FrameEval(
                _reference, prop_src=gmsd,
                eval=partial(_PlaneSSIMTransfer, clip=_reference, src_prop='PlaneGMSD', out_prop='GMSD')
                )
        else:
            measure = core.vmaf.Metric(reference=_reference, distorted=_distorted, feature=metric.value)

    elif metric in (Metric.SSIMULACRA1, Metric.SSIMULACRA2, Metric.BUTTER):
        if reference.format.color_family != vs.RGB:
            _reference, _distorted = [core.resize.Spline64(
                i, format=vs.RGB24, matrix_in=matrix, dither_type='error_diffusion'
                ) for i in (reference, distorted)]
        else:
            _reference, _distorted = reference, distorted

        if metric == Metric.SSIMULACRA1:
            measure = core.julek.SSIMULACRA(reference=_reference, distorted=_distorted, feature=1, **args)
        elif metric == Metric.SSIMULACRA2:
            measure = core.julek.SSIMULACRA(reference=_reference, distorted=_distorted, feature=0, **args)
        elif metric == Metric.BUTTER:
            measure = core.julek.Butteraugli(reference=_reference, distorted=_distorted, **args)
 
    else:
        measure = core.std.PlaneStats(reference, distorted, **args)
 
    return merge_clip_props(reference, measure)
