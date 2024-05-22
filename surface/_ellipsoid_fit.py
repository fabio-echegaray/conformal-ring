import numpy as np
from lmfit import Minimizer
from lmfit import Parameters

from surface._base import BaseFit


class EllipsoidFit(BaseFit):

    def __init__(self, vtk_ellipsoid, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._vtk_ellipsoid = vtk_ellipsoid
        self._a = vtk_ellipsoid.x_radius
        self._b = vtk_ellipsoid.y_radius
        self._c = vtk_ellipsoid.z_radius

        self._a2 = self._a ** 2
        self._b2 = self._b ** 2
        self._c2 = self._c ** 2

        self._prop_keys.update({
            "x_radius": "self._a",
            "y_radius": "self._b",
            "z_radius": "self._c",
            "x_radius2": "self._a2",
            "y_radius2": "self._b2",
            "z_radius2": "self._c2",
        })

    def __setattr__(self, name, value):
        if name in self._prop_keys.keys():
            exec(f"{self._prop_keys[name]} = {value}")
            if name[2:8] == "radius":
                exec(f"{self._prop_keys[name]}2 = {value ** 2}")
            self.ask_recalc()
        else:
            super().__setattr__(name, value)

    def state(self):
        return self._x0, self._y0, self._z0, self._roll, self._ptch, self._yaw, self._a, self._b, self._c

    def _grid(self):
        s = self._spac
        if self.local_search is not None:
            ls = self.local_search
            self._xl = [x * s - (self._w + ls) / 2 for x in range(int((self._w + ls) / s))]
            self._yl = [y * s - (self._h + ls) / 2 for y in range(int((self._h + ls) / s))]
        else:
            self._xl = [x * s - self._a - self._x0 for x in range(int(self._a * 3 / s))]
            self._yl = [y * s - self._b - self._y0 for y in range(int(self._b * 3 / s))]
        self._xv, self._yv = np.meshgrid(self._xl, self._yl)
        self._zv = np.zeros_like(self._xv)

    def eval_surf(self):
        if self._surf_eval:
            return

        xv, yv = self._xv, self._yv

        z2 = 1 - xv ** 2 / self._a2 - yv ** 2 / self._b2
        z = self._c * np.sqrt(z2)

        pts, pts_flt = self.rigid_transform_points(np.vstack([xv, xv]), np.vstack([yv, yv]), np.vstack([z, -z]))

        with self.calculating_semaphore:
            self.pts = pts.astype(int)
            self.xl, self.yl, self.zl = pts
            self.xlo, self.ylo, self.zlo = pts_flt

            self._surf_eval = True

    def project_2d(self):
        self.eval_surf()

        # assert len(self.xl) == len(self.yl) and len(self.xl) == len(self.zl), "something happened in project_2d"
        out = np.nanmean(self.projected_img_2d), self._img_changes
        return out

    def _obj_fn_minimize_0(self, p):
        self._eval_params(p)

        # ab_ratio = self._a / self._b
        ab_rel = self._b > self._a
        ac_ratio = self._a / self._c
        # ac_rel = self._a > self._c
        sr = 10 * np.abs(1 - ac_ratio)
        sr += 10000 if not ab_rel else 0

        self.eval_surf()
        s0 = np.sqrt(
            (self._x0 - self._w / 2) ** 2 +
            (self._y0 - self._h / 2) ** 2 +
            (self._z0 - self._c - self._nz / 2) ** 2)
        zz = self._pts[2]
        zhits = np.nansum(np.logical_and(zz >= 0, zz <= self._nz))

        out = s0 + sr + np.nan_to_num(10 / zhits, posinf=1e5) if zhits > 0 else s0 + 1e5
        xv = np.array([p[n].value for n in p.keys()])
        xv_str = np.array2string(xv, precision=1, suppress_small=False, floatmode='fixed')
        print(f"testing 0 f({xv_str})= {s0} + 10/{zhits} ={out:0.6E}  ac_ratio={ac_ratio}")

        return out

    def _obj_fn_minimize_1(self, p):
        self._eval_params(p)

        # ab_ratio = self._a / self._b
        # ab_rel = self._b > self._a

        s, chg = self.project_2d()

        xx, yy, zz = self._pts
        xx -= self._x0 + self._w / 2
        yy -= self._y0 + self._h / 2
        zz -= self._z0 + self._nz / 2

        sdiff = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
        dist_to_vol = np.nanmin(sdiff)

        zhits = np.nansum(np.logical_and(zz >= 0, zz <= self._nz))
        zin_dist = 100 * (self._w * self._h / self._spac ** 2 - zhits)
        in_z = zin_dist

        o0_den = 0.01 * chg + s
        out = np.nan_to_num(100 / o0_den, posinf=1e5) + 0.0001 * (dist_to_vol + in_z) if o0_den > 0 else 1e6
        xv = np.array([p[n].value for n in p.keys()])
        xv_str = np.array2string(xv, precision=1, suppress_small=True, floatmode='fixed')
        print(f"testing 1 f({xv_str})=100/(0.01* {chg} + {s} )+  0.0001* {dist_to_vol + in_z:0.2f} ={out:0.6E}")

        return out

    def optimize_parameters_0(self):
        """
        This optimization step searches for the best position to place the ellipsoid so it contains most of the voxels
        :return: MinimizerResult
        """
        params = Parameters()
        params.add('x0', value=self._w / 2, vary=True)
        params.add('y0', value=self._h / 2, vary=True)
        params.add('z0', value=self._c + self._nz / 2, vary=True)
        params.add('x_radius', value=self._a, vary=True)
        params.add('y_radius', value=self._b, vary=True)
        params.add('z_radius', value=self._c, vary=True)
        params.add('roll', value=0, vary=True)
        params.add('pitch', value=0, vary=False)
        params.add('yaw', value=0, vary=True)

        params['x0'].min = -100 * self._ppu
        params['x0'].max = 100 * self._ppu
        params['y0'].min = -100 * self._ppu
        params['y0'].max = 100 * self._ppu
        params['z0'].min = -100 * self._ppu
        params['z0'].max = 100 * self._ppu

        params['x_radius'].min = 0.1
        params['x_radius'].max = 400 * self._ppu
        params['y_radius'].min = 0.1
        params['y_radius'].max = 1000 * self._ppu
        params['z_radius'].min = 0.1
        params['z_radius'].max = 400 * self._ppu
        for r in ['roll', 'pitch', 'yaw']:
            params[r].min = -10
            params[r].max = 10

        fitter = Minimizer(self._obj_fn_minimize_0, params)
        self._result0 = fitter.minimize(method='bgfs', params=params)

        self._eval_params(self._result0.params)
        print(self._result0)
        print(self._result0.params, self._result0.residual)

        return self._result0

    def optimize_parameters_1(self):
        """
        This optimization step searches for the best position to place the ellipsoid so it contains most of the voxels
        :return: MinimizerResult
        """
        p0 = self._result0.params
        params = Parameters()
        params.add('x0', value=p0['x0'].value, vary=True)
        params.add('y0', value=p0['y0'].value, vary=True)
        params.add('z0', value=p0['z0'].value, vary=True)
        params.add('x_radius', value=p0['x_radius'].value, vary=True)
        params.add('y_radius', value=p0['y_radius'].value, vary=True)
        params.add('z_radius', value=p0['z_radius'].value, vary=True)
        params.add('roll', value=p0['roll'].value, vary=True)
        params.add('pitch', value=p0['pitch'].value, vary=False)
        params.add('yaw', value=p0['yaw'].value, vary=True)

        params['x0'].min = p0['x0'].value - self._w / 8
        params['x0'].max = p0['x0'].value + self._w / 8
        params['y0'].min = p0['y0'].value - self._h / 8
        params['y0'].max = p0['y0'].value + self._h / 8
        params['z0'].min = p0['z0'].value - self._nz / 8
        params['z0'].max = p0['z0'].value + self._nz / 8

        params['x_radius'].min = p0['x_radius'].value - 0.1 * p0['x_radius'].value
        params['x_radius'].max = p0['x_radius'].value + 0.1 * p0['x_radius'].value
        params['y_radius'].min = p0['y_radius'].value - 0.1 * p0['y_radius'].value
        params['y_radius'].max = p0['y_radius'].value + 0.1 * p0['y_radius'].value
        params['z_radius'].min = p0['z_radius'].value - 0.1 * p0['z_radius'].value
        params['z_radius'].max = p0['z_radius'].value + 0.1 * p0['z_radius'].value
        for r in ['roll', 'pitch', 'yaw']:
            val = p0[r].value if p0[r].value > 0 else 0.1
            params[r].min = params[r].value - 0.5 * val
            params[r].max = params[r].value + 0.5 * val

        with self.calculating_semaphore:
            self.local_search = 200
            self._grid()

        fitter = Minimizer(self._obj_fn_minimize_1, params)
        # self._result1 = fitter.minimize(method='bgfs', params=params)
        self._result1 = fitter.minimize(method='basinhopping', params=params)  # , niter=10 ** 4, niter_success=1000)

        # repeat search one last time, only now at full resolution
        print("performing last search (last one!)")
        self.sample_spacing = 4  # this will acquire a calculating_semaphore and will recompute grid

        p1 = self._result1.params
        p1['x0'].min = p1['x0'].value - self._w / 8
        p1['x0'].max = p1['x0'].value + self._w / 8
        p1['y0'].min = p1['y0'].value - self._h / 8
        p1['y0'].max = p1['y0'].value + self._h / 8
        p1['z0'].min = p1['z0'].value - self._nz / 8
        p1['z0'].max = p1['z0'].value + self._nz / 8

        p1['x_radius'].min = p1['x_radius'].value - 0.1 * p1['x_radius'].value
        p1['x_radius'].max = p1['x_radius'].value + 0.1 * p1['x_radius'].value
        p1['y_radius'].min = p1['y_radius'].value - 0.1 * p1['y_radius'].value
        p1['y_radius'].max = p1['y_radius'].value + 0.1 * p1['y_radius'].value
        p1['z_radius'].min = p1['z_radius'].value - 0.1 * p1['z_radius'].value
        p1['z_radius'].max = p1['z_radius'].value + 0.1 * p1['z_radius'].value
        for r in ['roll', 'pitch', 'yaw']:
            val = p0[r].value if p0[r].value > 0 else 0.1
            p1[r].min = p1[r].value - 0.2 * val
            p1[r].max = p1[r].value + 0.2 * val

        fitter = Minimizer(self._obj_fn_minimize_1, p1)
        self._result1 = fitter.minimize(method='bgfs', params=p1)

        self._eval_params(self._result1.params)
        print(self._result1)
        print(self._result1.params, self._result1.residual)

        return self._result1
