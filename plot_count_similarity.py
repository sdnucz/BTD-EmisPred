"""Draw the shaded count-fingerprint lollipop plot from calculated similarities."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('outputs/count_similarity/Count_similarity.csv'))
    parser.add_argument('--output', type=Path, default=Path('outputs/count_similarity/figures'))
    parser.add_argument('--font-file', type=Path, help='Optional local Times New Roman TTF file.')
    args = parser.parse_args()
    if args.font_file:
        font_manager.fontManager.addfont(args.font_file)
        family = font_manager.FontProperties(fname=args.font_file).get_name()
    else:
        family = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams.update({'font.family': family, 'font.size': 18, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'axes.linewidth': 1,
                         'axes.spines.top': False, 'axes.spines.right': False})
    data = pd.read_csv(args.input)
    required = {'compound', 'cohort', 'max_sparse_count_tanimoto'}
    if not required.issubset(data.columns) or data.empty:
        raise ValueError('Input must be the nonempty Count_similarity.csv output.')
    values = data.max_sparse_count_tanimoto.to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Similarities must be finite values between zero and one.')
    if not set(data.cohort).issubset({'SI6', 'own3'}):
        raise ValueError('Plot cohorts must be SI6 or own3.')
    palette = {'SI6': '#599DB7', 'own3': '#EFA775'}
    backgrounds = {'SI6': '#EAF5F8', 'own3': '#FFF5E9'}
    colors = [palette[c] for c in data.cohort]
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(15, 5.2))
    fig.subplots_adjust(left=.09, right=.98, bottom=.28, top=.77)
    # Shade contiguous cohort blocks, preserving the supplied compound order.
    start = 0
    for end in range(1, len(data) + 1):
        if end == len(data) or data.cohort.iloc[end] != data.cohort.iloc[start]:
            ax.axvspan(start - .5, end - .5, color=backgrounds[data.cohort.iloc[start]], zorder=0)
            if end < len(data):
                ax.axvline(end - .5, ls=(0, (3, 4)), color='#CCCCCC', lw=.8, zorder=1)
            start = end
    ax.vlines(x, 0, values, colors=colors, lw=1.5, alpha=.75, zorder=2)
    ax.scatter(x, values, s=235, facecolors='white', edgecolors=colors, linewidths=1.1, alpha=.5, zorder=3)
    ax.scatter(x, values, s=130, c=colors, edgecolors='white', linewidths=1, zorder=4)
    for i, value in enumerate(values):
        ax.text(i, value + .055, f'{value:.3f}', ha='center', va='bottom', fontsize=17)
    ax.set_ylim(0, 1.15)
    ax.set_xlim(-.5, len(data) - .5)
    ax.set_yticks([0, .25, .5, .75, 1], ['0', '0.25', '0.50', '0.75', '1.00'])
    ax.set_ylabel('Maximum Tanimoto similarity', fontsize=18, labelpad=12)
    ax.set_title('Count Morgan fingerprint', loc='left', fontsize=22, pad=12)
    ax.set_xticks(x, data.compound, rotation=38, ha='right', rotation_mode='anchor')
    ax.tick_params(axis='both', labelsize=17, direction='out', length=4)
    labels = {'SI6': 'Supplementary Fig. 6 compounds', 'own3': 'Our three compounds'}
    handles = [Line2D([], [], color=palette[c], marker='o', lw=1.2, markersize=9, label=labels[c])
               for c in dict.fromkeys(data.cohort)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.55, .99), ncol=2, frameon=False, fontsize=18)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in [*ax.texts, *ax.get_xticklabels(), *ax.get_yticklabels(), ax.yaxis.label, ax._left_title]:
        bounds = text.get_window_extent(renderer)
        if bounds.x0 < 0 or bounds.y0 < 0 or bounds.x1 > fig.bbox.width or bounds.y1 > fig.bbox.height:
            raise ValueError(f'Text outside canvas: {text.get_text()}')
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(args.output / f'Count_similarity_lollipop.{extension}', dpi=350, facecolor='white')
    fig.savefig(args.output / 'Preview.png', dpi=110, facecolor='white')
    plt.close(fig)
    print(f'Exported count-fingerprint lollipop plot to {args.output}')


if __name__ == '__main__':
    main()
