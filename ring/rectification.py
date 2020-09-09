import logging
import time

import numpy as np
import pyvista as pv
from shapely.geometry import Polygon
from skimage.transform import PiecewiseAffineTransform, warp
from scipy.interpolate import UnivariateSpline

import matplotlib.pyplot as plt

from ring import plots as p
from gui._image_loading import retrieve_image
from ring.measure import FileImageMixin


def timeit(method):
    log = logging.getLogger('timeit')

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            log.debug('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


# FIXME: how to make a clockwise strictly increasing curve? This is:
#  1 - how to homogenize the points so it won't have more dense areas?
#  2 - how to make the parametrization function of the arclength?

class BaseApproximation(FileImageMixin):
    log = logging.getLogger('BaseApproximation')

    def __init__(self, polygon: Polygon, image):
        super(BaseApproximation, self).__init__()
        self._load_image(image)
        self._poly = polygon

        self._fn = lambda x: np.array([0, 0])
        self._dfn_dt = lambda x: np.array([0, 0])

    def _load_image(self, img):
        if type(img) is str:
            self.file = img
        elif issubclass(type(img), FileImageMixin):
            self.fileimage_from(img)
        else:
            raise Exception("Couldn't load image file.")

    def approximate_fn(self):
        pass

    def f(self, t):
        return self._fn(t)

    def tangent_angle(self, t):
        dx, dy = self._dfn_dt(t)
        if dy != 0:
            return np.arctan2(dy, dx)
        else:
            return np.nan

    def normal_angle(self, t):
        dx, dy = self._dfn_dt(t)
        if dx.any() != 0:
            # as arctan2 argument order is  y, x (and as we're doing a rotation) -> x=-dy y=dx)
            return np.arctan2(dx, -dy)
        else:
            return np.nan


@timeit
def harmonic_approximation(polygon: Polygon, n=3):
    from symfit import Eq, Fit, cos, parameters, pi, sin, variables

    def fourier_series(x, f, n=0):
        """
        Returns a symbolic fourier series of order `n`.

        :param n: Order of the fourier series.
        :param x: Independent variable
        :param f: Frequency of the fourier series
        """
        # Make the parameter objects for all the terms
        a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
        sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
        # Construct the series
        series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                          for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
        return series

    x, y = variables('x, y')
    w, = parameters('w')
    fourier = fourier_series(x, f=w, n=n)
    model_dict = {y: fourier}
    print(model_dict)

    # Extract data from argument
    # FIXME: how to make a clockwise strictly increasing curve?
    xdata, ydata = polygon.exterior.xy
    t = np.linspace(0, 2 * np.pi, num=len(xdata))

    constr = [
        # Ge(x, 0), Le(x, 2 * pi),
        Eq(fourier.subs({x: 0}), fourier.subs({x: 2 * pi})),
        Eq(fourier.diff(x).subs({x: 0}), fourier.diff(x).subs({x: 2 * pi})),
        # Eq(fourier.diff(x, 2).subs({x: 0}), fourier.diff(x, 2).subs({x: 2 * pi})),
        ]
    print(constr)

    fit_x = Fit(model_dict, x=t, y=xdata, constraints=constr)
    fit_y = Fit(model_dict, x=t, y=ydata, constraints=constr)
    fitx_result = fit_x.execute()
    fity_result = fit_y.execute()
    print(fitx_result)
    print(fity_result)

    # Define function that generates the curve
    def curve_lambda(_t):
        return np.array(
            [
                fit_x.model(x=_t, **fitx_result.params).y,
                fit_y.model(x=_t, **fity_result.params).y
                ]
            ).ravel()

    # code to test if fit is correct
    plot_fit(polygon, curve_lambda, t, title='Harmonic Approximation')

    return curve_lambda


class SplineApproximation(BaseApproximation):
    log = logging.getLogger('SplineApproximation')

    def __init__(self, polygon: Polygon, image):
        super(SplineApproximation, self).__init__(polygon, image)
        self.approximate_fn()

    @timeit
    def approximate_fn(self):
        # Extract data from argument
        # FIXME: how to make a clockwise strictly increasing curve?
        cols, rows = self._poly.exterior.xy
        t = np.linspace(0, 2 * np.pi, num=len(cols))

        # Define spline minimizer function
        splx = UnivariateSpline(t, cols)
        sply = UnivariateSpline(t, rows)
        # splx.set_smoothing_factor(2 * self.pix_per_um)
        # sply.set_smoothing_factor(2 * self.pix_per_um)

        # Define spline 1st order derivative
        dsplx_dt = splx.derivative()
        dsply_dt = sply.derivative()

        # Define function that generates the curve
        self._fn = lambda o: np.array([splx(o), sply(o)])
        self._dfn_dt = lambda o: np.array([dsplx_dt(o), dsply_dt(o)])


_inc_theta = 500


class FunctionRectification:
    log = logging.getLogger('FunctionRectification')

    # rectify the image using the approximated function directly
    def __init__(self, curve: BaseApproximation, dl=1, pix_per_dl=1, pix_per_arclen=1):
        self._model = curve

        self.dl = dl
        self.n_dl = None
        self.arc_dl = np.inf
        self.n_theta = _inc_theta  # min(n_theta) = 4
        self.pix_per_dl = pix_per_dl
        self.pix_per_arclen = pix_per_arclen

        self.out_rows = None
        self.out_cols = None

        self.spline = None
        self.ring = None

    def _calc_theta(self):
        self.theta_rng = np.linspace(0, 2 * np.pi, num=self.n_theta)
        points = np.array([[c, r, 0] for c, r in zip(*self._model._poly.exterior.xy)])
        spline = pv.Spline(points, self.n_theta)  # FIXME: Move to SplineApproximation as here is not intuitive.
        self.arc_dl = max(np.diff(spline.get_array("arc_length")))

        self.spline = spline

    def _calc(self):
        # calculate minimum number of theta steps to cover all pixels radially
        self.n_dl = np.ceil(self.dl * 2 * self._model.pix_per_um / np.sqrt(2)).astype(int) * 2 + 1

        # calculate minimum number of theta steps to cover all pixels tangentially
        self._calc_theta()
        while self.arc_dl > 1:
            self.log.debug(f"step n_dl={self.n_dl}, n_theta={self.n_theta}, arc_dl={self.arc_dl}")
            self.n_theta += _inc_theta
            self._calc_theta()

        self.ring = self.spline.ribbon(width=self.dl * self._model.pix_per_um, normal=(0, 0, 1))
        self.out_rows = self.n_dl * self.pix_per_dl
        self.out_cols = self.n_theta * self.pix_per_arclen

        self.log.info(f"optimal n_dl={self.n_dl}, n_theta={self.n_theta}, arc_dl={self.arc_dl}, "
                      f"out_rows={self.out_rows}, out_cols={self.out_cols}")

    @timeit
    def rectify(self, image):
        def rect_fn(cr: np.array):
            # hack solution with the parameters we already know instead of using the input cr
            k_dl = 0
            out = np.empty(cr.shape)
            pts = iter(self.ring.points)
            thit = iter(self.theta_rng)
            for p0, p1, th in zip(*[pts] * 2, thit):
                x1 = np.linspace(p0[:2], p1[:2], num=self.n_dl)
                out[k_dl:k_dl + self.n_dl] = x1
                k_dl += self.n_dl

            return out

        self._calc()
        return warp(image, rect_fn, output_shape=(self.out_rows, self.out_cols), preserve_range=True)


class TestFunctionRectification(FunctionRectification):
    def plot_rectification(self):
        import matplotlib.pyplot as plt
        image = retrieve_image(self._model.images, channel=0, number_of_channels=self._model.nChannels,
                               zstack=self._model.zstack, number_of_zstacks=self._model.nZstack, frame=0)

        rows, cols = image.shape[0], image.shape[1]
        out = self.rectify(image)

        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})
        ax1.imshow(image, origin='lower')
        # ax1.plot(self.src[:, 0], self.src[:, 1], '.b')
        ax1.axis((0, cols, rows, 0))

        # ext = (0, 2 * np.pi, -self.dl, self.dl)
        ax2.imshow(out, origin='upper', interpolation='none', aspect='equal')  # , extent=ext)
        # ax2.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        # ax2.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
        # ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
        # ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter(r'%g $\pi$'))

        fig = plt.figure()
        ax = fig.gca()
        # ext = [0, t_dom.max(), dl_dom.max() * 2, 0]
        plt.imshow(out, origin='upper', aspect='auto')  # , extent=ext)
        ax.set_title('Image rectification using original function')

        plt.show(block=False)


class PiecewiseLinearRectification:
    log = logging.getLogger('PiecewiseLinearRectification')

    # rectify the image using a piecewise affine transform from skimage library
    def __init__(self, curve: BaseApproximation, dl=1, n_dl=10, n_theta=50, pix_per_dl=100, pix_per_theta=100):
        self._model = curve
        self.dl_dom = None
        self.t_dom = None
        self.src = None
        self.dst = None
        self.out_rows = None
        self.out_cols = None
        self.dst_rows = None
        self.dst_cols = None

        self.estimated = False

        self.dl = dl
        self.n_dl = n_dl
        self.n_theta = n_theta
        self.pix_per_dl = pix_per_dl
        self.pix_per_theta = pix_per_theta

    @timeit
    def _estimate_transform(self):
        if self.estimated:
            return

        self.estimated = True

        # define the ending points of the transformation
        self.dl_dom = np.linspace(-self.dl, self.dl, num=self.n_dl) * self._model.pix_per_um
        self.t_dom = np.linspace(0, 2 * np.pi, num=self.n_theta)
        self.dst_rows, self.dst_cols = np.meshgrid(self.dl_dom, self.t_dom)

        # calculate the original points
        self.src_rows = self.dst_rows.copy()
        self.src_cols = self.dst_cols.copy()
        for i in range(self.src_cols.shape[0]):
            t = self.src_cols[i, 0]
            x0, y0 = self._model.f(t)
            o = self._model.normal_angle(t)
            sin_o = np.sin(o)
            cos_o = np.cos(o)

            for j in range(self.src_rows.shape[1]):
                dl = self.src_rows[i, j]
                self.log.debug(f"debug i={j},  j={i}, dl={dl:.2f}   src_cols[i, j]-t={self.src_cols[i, j] - t:.3f}")
                self.src_cols[i, j] = x0 + dl * cos_o
                self.src_rows[i, j] = y0 + dl * sin_o

        # rescale the point of the dst mesh to match output image
        self.out_rows = self.dl_dom.size * self.pix_per_dl
        self.out_cols = self.t_dom.size * self.pix_per_theta
        self.dst_rows = np.linspace(0, self.out_rows, self.dl_dom.size)
        self.dst_cols = np.linspace(0, self.out_cols, self.t_dom.size)
        self.dst_rows, self.dst_cols = np.meshgrid(self.dst_rows, self.dst_cols)

        # convert meshes to (N,2) vectors
        self.src = np.dstack([self.src_cols.flat, self.src_rows.flat])[0]
        self.dst = np.dstack([self.dst_cols.flat, self.dst_rows.flat])[0]

        self.transform = PiecewiseAffineTransform()
        self.transform.estimate(self.dst, self.src)

    @timeit
    def rectify(self, image):
        self._estimate_transform()
        return warp(image, self.transform, output_shape=(self.out_rows, self.out_cols))  # , order=2)


class TestPiecewiseLinearRectification(PiecewiseLinearRectification):
    def plot_rectification(self):
        import matplotlib.pyplot as plt
        image = retrieve_image(self._model.images, channel=0, number_of_channels=self._model.nChannels,
                               zstack=self._model.zstack, number_of_zstacks=self._model.nZstack, frame=0)

        rows, cols = image.shape[0], image.shape[1]
        out = self.rectify(image)

        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})
        ax1.imshow(image, origin='lower')
        # ax1.plot(self.src[:, 0], self.src[:, 1], '.b')
        ax1.axis((0, cols, rows, 0))

        ax2.imshow(out, origin='lower')
        # ax2.plot(self.transform.inverse(self.src)[:, 0], self.transform.inverse(self.src)[:, 1], '.b')

        fig = plt.figure()
        ax = fig.gca()
        # ext = [0, t_dom.max(), dl_dom.max() * 2, 0]
        plt.imshow(out, origin='lower', aspect='auto')  # , extent=ext)
        ax.set_title('Image rectification using piecewise linear transform')

        plt.show(block=False)


class TestSplineApproximation(SplineApproximation):
    def test_fit(self):
        t = np.linspace(0, 2 * np.pi, num=len(self._poly.exterior.xy[0]))
        plot_fit(self._poly, self.f, t, title='Spline Approximation')

    def grid(self, dna_ch=0, act_ch=2, width_dl=1, n_dl=5, n_theta=10, ax=None):
        if ax is None:
            ax = plt.gca()

        c = self._poly.centroid
        xdata, ydata = self._poly.exterior.xy
        ax.set_xlim([min(xdata) - 2 * self.pix_per_um, max(xdata) + 2 * self.pix_per_um])
        ax.set_ylim([min(ydata) - 2 * self.pix_per_um, max(ydata) + 2 * self.pix_per_um])

        image = retrieve_image(self.images, channel=dna_ch, number_of_channels=self.nChannels,
                               zstack=self.zstack, number_of_zstacks=self.nZstack, frame=0)
        ax.imshow(image, origin='lower', cmap='gray')

        p.render_polygon(self._poly, zorder=10, ax=ax)
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
            theta = np.linspace(0, 2 * np.pi, num=n_th)
            points = np.array([[*self.f(theta), 0] for theta in theta])
            spline = pv.Spline(points, n_th).compute_arc_length()
            dl = max(np.diff(spline.get_array("arc_length")))
            return theta, spline, dl

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

    def plot_grid(self, dna_ch=0, act_ch=2, width_dl=1, n_dl=5, n_theta=10, ax=None):
        self.grid(dna_ch=dna_ch, act_ch=act_ch, width_dl=width_dl, n_dl=n_dl, n_theta=n_theta, ax=ax)
        plt.show(block=False)

        # p = pv.Plotter()
        # mesh = ring
        # p.add_mesh(mesh, show_edges=True, color='white')
        # p.add_mesh(pv.PolyData(mesh.points), color='red', point_size=10, render_points_as_spheres=True)
        # p.add_point_labels(mesh.points, ["%d" % i for i in range(mesh.n_points)], font_size=30, tolerance=0.1)
        # p.show()


def plot_fit(polygon: Polygon, fit_fn, t, title=""):
    import matplotlib.pyplot as plt
    import ring.plots as p

    xdata, ydata = polygon.exterior.xy
    pts = np.array([fit_fn(_t) for _t in t])

    fig = plt.figure(10)
    ax = fig.gca()

    ax.scatter(t, xdata, linewidth=1, marker='o', edgecolors='red', facecolors='none', label='x')
    ax.scatter(t, ydata, linewidth=1, marker='o', edgecolors='blue', facecolors='none', label='y')
    ax.plot(t, pts[:, 0], linewidth=1, linestyle='-', c='red')
    ax.plot(t, pts[:, 1], linewidth=1, linestyle='-', c='blue')
    ax.set_title(title)
    ax.legend()

    fig = plt.figure()
    ax = fig.gca()
    # plt.imshow(image, origin='lower')
    # n_um = affinity.scale(n, xfact=self.um_per_pix, yfact=self.um_per_pix, origin=(0, 0, 0))

    p.render_polygon(polygon, zorder=10, ax=ax)
    ax.plot(pts[:, 0], pts[:, 1], linewidth=1, linestyle='-', c='blue')
    ax.set_title('Spline approximation')

    plt.show(block=False)
