import numpy as np
from lmfit import Minimizer
from lmfit import Parameters
from skimage.filters import rank
from skimage.morphology import disk

from surface._base import BaseFit


class PlaneFit(BaseFit):
    _prop_keys: dict = {}

    def __init__(self, a, b, c, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._a = a
        self._b = b
        self._c = c

        self._prop_keys.update({
            "a": "self._a",
            "b": "self._b",
            "c": "self._c",
        })

    def state(self):
        return self._x0, self._y0, self._z0, self._roll, self._ptch, self._yaw, self._a, self._b, self._c

    def _grid(self):
        s = self._spac
        self._xl = [x * s - self._x0 for x in range(int(self._w / s) + 1)]
        self._yl = [y * s - self._y0 for y in range(int(self._h / s) + 1)]
        self._xv, self._yv = np.meshgrid(self._xl, self._yl)
        self._zv = np.zeros_like(self._xv)

    def eval_surf(self):
        if self._surf_eval:
            return

        xv, yv = self._xv, self._yv
        zv = -1 / self._c * (self._a * xv + self._b * yv)

        pts, pts_flt = self.rigid_transform_points(xv, yv, zv)

        with self.calculating_semaphore:
            self.pts = pts
            self.xl, self.yl, self.zl = pts
            # self.xlo, self.ylo, self.zlo = pts

        self._surf_eval = True

    def project_2d(self):
        self.eval_surf()

        # out = np.nanmedian(self.projected_img_2d), self._img_changes
        # out = np.nanmean(self.projected_img_2d), self._img_changes
        # footprint = disk(10 * self._ppu)
        footprint = disk(60)
        img_8bit = ((self.projected_img_2d - self.projected_img_2d.min()) / (
                self.projected_img_2d.ptp() / 255.0)).astype(np.uint8)
        # percentile_result = rank.mean_percentile(img_8bit, footprint=footprint, p0=0.1, p1=0.9)
        # bilateral_result = rank.mean_bilateral(img_8bit, footprint=footprint, s0=500, s1=500)
        median_result = rank.median(img_8bit, footprint=footprint)
        # mean_result = rank.mean(img_8bit, footprint=footprint)
        # out = np.sum(self.projected_img_2d), np.ptp(self.projected_img_2d), np.sum(mean_result), np.sum(median_result), self._img_changes
        out = np.sum(self.projected_img_2d), np.ptp(self.projected_img_2d), 0, np.sum(median_result), self._img_changes
        return out

    def _obj_fn_minimize_0(self, p):
        self._eval_params(p)

        sum, ptp, mean, median, chg = self.project_2d()
        # sleep(0.1)
        out = np.nan_to_num(100000 / (chg + sum + 1000 * ptp + 10 * median), posinf=1e5)
        # plane should be in at least 70% of the volume
        # if chg < self._h * self._w * 0.7:
        #     out *= 1e2
        xv = np.array([p[n].value for n in p.keys()])
        xv_str = np.array2string(xv, precision=1, suppress_small=True, floatmode='fixed')
        print(f"testing f({xv_str})=100000/({chg}+{sum}+1000*{ptp}+10*{median})={out}")

        return out

    def optimize_parameters_0(self):
        """
        This optimization step searches for the best position to place the ellipsoid so it contains most of the voxels
        :return: MinimizerResult
        """
        self.sample_spacing = 10  # this will acquire a calculating_semaphore and will recompute grid

        params = Parameters()
        params.add('x0', value=0, vary=False)
        params.add('y0', value=0, vary=False)
        params.add('z0', value=0, vary=True)
        params.add('a', value=self._a, vary=False)
        params.add('b', value=self._b, vary=False)
        params.add('c', value=self._c, vary=False)
        params.add('roll', value=0, vary=False)
        params.add('pitch', value=0, vary=False)
        params.add('yaw', value=0, vary=False)

        params['x0'].min = -self._w * self._ppu
        params['x0'].max = self._w * self._ppu
        params['y0'].min = -self._h * self._ppu
        params['y0'].max = self._h * self._ppu
        params['z0'].min = -self._nz / 2 * self._ppu
        params['z0'].max = self._nz / 2 * self._ppu

        params['a'].min = 0
        params['a'].max = 0.1
        params['b'].min = 0
        params['b'].max = 0.1
        params['c'].min = 0
        params['c'].max = 0.1

        print("brute force search for z")
        self._delay = 0.1
        fitter = Minimizer(self._obj_fn_minimize_0, params)
        self._result0 = fitter.minimize(method='brute', params=params, Ns=self._nz)

        # repeat search one last time, only now at full resolution
        print("search with bassinhoping")
        self._delay = None
        p1 = self._result0.params
        p1['a'].vary = True
        p1['b'].vary = True
        self._result0 = fitter.minimize(method='basinhopping', params=p1, stepsize=0.01)
        # self._result0 = fitter.minimize(method='emcee', params=params, nwalkers=10000)
        # self._result0 = fitter.minimize(method='dual_annealing', params=params)

        # repeat search one last time, only now at full resolution
        print("performing last search (last one!)")
        self.sample_spacing = 3  # this will acquire a calculating_semaphore and will recompute grid

        p1 = self._result0.params
        # p1['x0'].min = p1['x0'].value - self._w / 8
        # p1['x0'].max = p1['x0'].value + self._w / 8
        # p1['y0'].min = p1['y0'].value - self._h / 8
        # p1['y0'].max = p1['y0'].value + self._h / 8
        p1['z0'].min = p1['z0'].value - self._nz / 2
        p1['z0'].max = p1['z0'].value + self._nz / 2

        # p1['a'].min = p1['a'].value - 0.1 * p1['a'].value
        # p1['a'].max = p1['a'].value + 0.1 * p1['a'].value
        # p1['b'].min = p1['b'].value - 0.1 * p1['b'].value
        # p1['b'].max = p1['b'].value + 0.1 * p1['b'].value
        # p1['c'].min = p1['c'].value - 0.1 * p1['c'].value
        # p1['c'].max = p1['c'].value + 0.1 * p1['c'].value

        fitter = Minimizer(self._obj_fn_minimize_0, p1)
        self._result0 = fitter.minimize(method='bgfs', params=p1)

        self._eval_params(self._result0.params)
        print(self._result0)
        print(self._result0.params, self._result0.residual)

        return self._result0
