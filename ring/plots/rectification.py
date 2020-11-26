import itertools

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

from ring import plots as p
from ring.gui import retrieve_image
from ring.rectification import SplineApproximation


class PlotSplineApproximation(SplineApproximation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit_xy(self, ax=None):
        if ax is None:
            ax = plt.gca()

        palette = itertools.cycle(sns.color_palette())

        t = np.linspace(0, 2 * np.pi, num=len(self._poly.exterior.xy[0]))
        xdata, ydata = self._poly.exterior.xy
        pts = np.array([self.f(_t) for _t in t])

        # ax.scatter(t, xdata, linewidth=1, marker='o', facecolors='red',  label='x')
        # ax.scatter(t, ydata, linewidth=1, marker='o', facecolors='blue', label='y')
        ax.plot(t, pts[:, 0], linewidth=1, linestyle='-', marker='o', ms=2, c=next(palette), label='x')
        ax.plot(t, pts[:, 1], linewidth=1, linestyle='-', marker='o', ms=2, c=next(palette), label='y')
        ax.set_title('Spline Approximation')
        ax.set_ylabel('Coordinate $[pix]$')
        ax.set_xlabel('$\\theta \, [rad]$')
        ax.set_aspect('auto')
        ax.legend()

    def fit_polygon(self, ax=None):
        if ax is None:
            ax = plt.gca()

        palette = itertools.cycle(sns.color_palette())

        t = np.linspace(0, 2 * np.pi, num=len(self._poly.exterior.xy[0]))
        pts = np.array([self.f(_t) for _t in t])
        poly = p.render_polygon(self._poly, zorder=100, ax=ax, c=next(palette))
        fit_color = next(palette)
        ax.plot(pts[:, 0], pts[:, 1], linewidth=1, linestyle='-', c=fit_color)
        fit = mlines.Line2D([], [], color=fit_color, linestyle='--', marker=None)

        ax.legend([poly, fit], ["Polygon", "Spline"], loc='best')

        ax.set_title('Spline approximation')
        ax.set_xlabel('X $[pix]$')
        ax.set_ylabel('Y $[pix]$')

    def grid(self, dna_ch=0, act_ch=2, width_dl=1, n_dl=5, n_theta=10, ax=None,
             draw_boundary=True, bcolor='blue', draw_grid=True, draw_tangents=True,
             annotate_angles=True):
        if ax is None:
            ax = plt.gca()

        c = self._poly.centroid
        xdata, ydata = self._poly.exterior.xy
        ax.set_xlim([min(xdata) - 2 * self.pix_per_um, max(xdata) + 2 * self.pix_per_um])
        ax.set_ylim([min(ydata) - 2 * self.pix_per_um, max(ydata) + 2 * self.pix_per_um])

        image = retrieve_image(self.images, channel=dna_ch, number_of_channels=self.nChannels,
                               zstack=self.zstack, number_of_zstacks=self.nZstack, frame=0)
        ax.imshow(image, origin='lower', cmap='gray')

        if draw_boundary:
            # p.render_polygon(self._poly, zorder=10, ax=ax)
            pts = np.array([self.f(_t) for _t in np.linspace(0, 2 * np.pi, num=len(xdata))])
            ax.plot(pts[:, 0], pts[:, 1], linewidth=1, linestyle='-', c=bcolor)

        theta = np.linspace(0, 2 * np.pi, num=n_theta)
        # fx, fy = self.f(theta)
        # ax.scatter(fx, fy, linewidth=1, marker='o', edgecolors='yellow', facecolors='none', label='x')

        # plot normals and tangents
        dl_arr = np.linspace(-1, 1, num=n_dl) * self.pix_per_um
        for i, t in enumerate(theta):
            x0, y0 = self.f(t)

            if draw_grid:
                o = self.normal_angle(t)
                fnx, fny = np.array([(x0 + dl * np.cos(o), y0 + dl * np.sin(o)) for dl in dl_arr]).T
                ax.scatter(fnx, fny, linewidth=1, c='white', marker='o', s=2, zorder=100)

            if draw_tangents:
                th = self.tangent_angle(t)
                ftx, fty = np.array([(x0 + dl * np.cos(th), y0 + dl * np.sin(th)) for dl in dl_arr]).T
                ax.plot(ftx, fty, linewidth=1, linestyle='-', c='red')

            if annotate_angles:
                margin = 20
                ax.annotate((f"{i} ({t:.2f}): "
                             f"T{np.rad2deg(th):.0f}º  "
                             f"N{np.rad2deg(o):.0f}º "
                             f"{np.sin(o):.2f} {np.cos(o):.2f}"), xy=(x0, y0),
                            xytext=(min(xdata) - margin if x0 < c.x else max(xdata) + margin, y0),
                            color="red", size=10,
                            horizontalalignment='right' if x0 < c.x else 'left',
                            arrowprops=dict(arrowstyle='->', color='white', lw=0.5))

        ax.set_title("Grid on image")
        ax.set_xlabel('X $[pix]$')
        ax.set_ylabel('Y $[pix]$')
