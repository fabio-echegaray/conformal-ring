import numpy as np
import pyvista as pv

import matplotlib.pyplot as plt

from ring import plots as p
from ring.gui import retrieve_image
from ring.rectification import SplineApproximation


class PlotSplineApproximation(SplineApproximation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit_xy(self, ax=None):
        if ax is None:
            ax = plt.gca()

        t = np.linspace(0, 2 * np.pi, num=len(self._poly.exterior.xy[0]))
        xdata, ydata = self._poly.exterior.xy
        pts = np.array([self.f(_t) for _t in t])

        # ax.scatter(t, xdata, linewidth=1, marker='o', facecolors='red',  label='x')
        # ax.scatter(t, ydata, linewidth=1, marker='o', facecolors='blue', label='y')
        ax.plot(t, pts[:, 0], linewidth=1, linestyle='-', marker='o', ms=2, c='red')
        ax.plot(t, pts[:, 1], linewidth=1, linestyle='-', marker='o', ms=2, c='blue')
        ax.set_title('Spline Approximation')
        ax.set_ylabel('Coordinate [pix]')
        ax.set_xlabel('$\\theta [au]$')
        ax.legend()

    def fit_polygon(self, ax=None):
        if ax is None:
            ax = plt.gca()

        t = np.linspace(0, 2 * np.pi, num=len(self._poly.exterior.xy[0]))
        pts = np.array([self.f(_t) for _t in t])
        p.render_polygon(self._poly, zorder=10, ax=ax)
        ax.plot(pts[:, 0], pts[:, 1], linewidth=1, linestyle='-', c='blue')
        ax.set_title('Spline approximation')
        ax.set_xlabel('X [pix]')
        ax.set_ylabel('Y [pix]')

    def grid(self, dna_ch=0, act_ch=2, width_dl=1, n_dl=5, n_theta=10, ax=None,
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

        # p.render_polygon(self._poly, zorder=10, ax=ax)
        pts = np.array([self.f(_t) for _t in np.linspace(0, 2 * np.pi, num=len(xdata))])
        ax.plot(pts[:, 0], pts[:, 1], linewidth=1, linestyle='-', c='blue')
        ax.set_title("Grid on image")

        theta = np.linspace(0, 2 * np.pi, num=n_theta)
        fx, fy = self.f(theta)
        ax.scatter(fx, fy, linewidth=1, marker='o', edgecolors='yellow', facecolors='none', label='x')

        # plot normals and tangents
        dl_arr = np.linspace(-1, 1, num=n_dl) * self.pix_per_um
        for i, t in enumerate(theta):
            x0, y0 = self.f(t)

            o = self.normal_angle(t)
            th = self.tangent_angle(t)
            ftx, fty = np.array([(x0 + dl * np.cos(th), y0 + dl * np.sin(th)) for dl in dl_arr]).T
            ax.plot(ftx, fty, linewidth=1, linestyle='-', c='red', marker='<')

            margin = 20
            if annotate_angles:
                ax.annotate((f"{i} ({t:.2f}): "
                             f"T{np.rad2deg(th):.0f}º  "
                             f"N{np.rad2deg(o):.0f}º "
                             f"{np.sin(o):.2f} {np.cos(o):.2f}"), xy=(x0, y0),
                            xytext=(min(xdata) - margin if x0 < c.x else max(xdata) + margin, y0),
                            color="red", size=10,
                            horizontalalignment='right' if x0 < c.x else 'left',
                            arrowprops=dict(arrowstyle='->', color='white', lw=0.5))

        # calculate minimum number of theta steps to cover all pixels radially
        n_dl = np.ceil(width_dl * 2 * self.pix_per_um / np.sqrt(2)).astype(int) * 2 + 1

        # calculate minimum number of theta steps to cover all pixels tangentially
        def calc_theta(n_th):
            theta_rng = np.linspace(0, 2 * np.pi, num=n_th)
            points = np.array([[*self.f(theta), 0] for theta in theta_rng])
            spline_pv = pv.Spline(points, n_th).compute_arc_length()
            dl = max(np.diff(spline_pv.get_array("arc_length")))
            return theta_rng, spline_pv, dl

        theta, spline, dl = calc_theta(n_theta)
        while dl > 1:
            n_theta += 10
            theta, spline, dl = calc_theta(n_theta)

        self.log.debug(f"Test: optimal n_dl={n_dl}, n_theta={n_theta}")
        ring = spline.ribbon(width=width_dl * self.pix_per_um, normal=(0, 0, 1))

        pts = iter(ring.points)
        thit = iter(theta)
        _ = next(thit)
        p0, p1 = next(pts), next(pts)
        for p2, p3, th in zip(*[pts] * 2, thit):
            x1 = np.linspace(p0[:2], p1[:2], num=n_dl)
            x2 = np.linspace(p2[:2], p3[:2], num=n_dl)
            for i, j in zip(x1, x2):
                ax.plot([i[0], j[0]], [i[1], j[1]], linewidth=1, linestyle='-', c='white', marker='o', ms=2, alpha=0.5)
            p0 = p2
            p1 = p3

        ax.set_xlabel('X [pix]')
        ax.set_ylabel('Y [pix]')
