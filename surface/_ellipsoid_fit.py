from threading import Semaphore

import numpy as np
from lmfit import Minimizer
from lmfit import Parameters
from scipy.spatial.transform import Rotation as R


class EllipsoidFit:
    _prop_keys: dict = {}

    def __init__(self, vtk_ellipsoid, pix_per_um, xyz_0=(0, 0, 0), sample_spacing=1):
        self._ppu = pix_per_um

        self._spac = sample_spacing

        self._vtk_ellipsoid = vtk_ellipsoid
        self._a = vtk_ellipsoid.x_radius
        self._b = vtk_ellipsoid.y_radius
        self._c = vtk_ellipsoid.z_radius

        self._vol = None
        self._w = 0
        self._h = 0
        self._nz = 0
        self._dtype = None

        self._a2 = self._a ** 2
        self._b2 = self._b ** 2
        self._c2 = self._c ** 2
        self._x0 = xyz_0[0]
        self._y0 = xyz_0[1]
        self._z0 = xyz_0[2]
        self._X = None
        self._xl = None
        self._yl = None
        self._xv = None
        self._yv = None
        self._zv = None

        # euler angles
        self._roll = 0
        self._ptch = 0
        self._yaw = 0
        self._r: R = None
        self._R: np.array = None
        self._Ri: np.array = None

        self._surf_eval = False

        self.xl = None
        self.yl = None
        self.zl = None
        self._pts = None
        self.pts = None
        self._projected_img_2d = None
        self._img_2d_calculated = False
        self._img_changes = 0
        self._dist_to_vol = np.inf
        self.calculating_semaphore = Semaphore()
        self.stop = False

        self._prop_keys = {
            "x0": "self._x0",
            "y0": "self._y0",
            "z0": "self._z0",
            "roll": "self._roll",
            "pitch": "self._ptch",
            "yaw": "self._yaw",
            "x_radius": "self._a",
            "y_radius": "self._b",
            "z_radius": "self._c",
            "x_radius2": "self._a2",
            "y_radius2": "self._b2",
            "z_radius2": "self._c2",
        }

    def __setattr__(self, name, value):
        if name in self._prop_keys.keys():
            exec(f"{self._prop_keys[name]} = {value}")
            if name[2:8] == "radius":
                exec(f"{self._prop_keys[name]}2 = {value ** 2}")
            with self.calculating_semaphore:
                self._surf_eval = False
                self._img_2d_calculated = False
        else:
            super().__setattr__(name, value)

    def state(self):
        return self._x0, self._y0, self._z0, self._a, self._b, self._c, self._roll, self._ptch, self._yaw

    @property
    def sample_spacing(self):
        return self._spac

    @sample_spacing.setter
    def sample_spacing(self, spacing: int):
        self._spac = spacing
        with self.calculating_semaphore:
            self._projected_img_2d = None
            self._img_2d_calculated = False

            self._grid(self._spac)

        self.eval_surf()

    @property
    def volume(self):
        return self._vol

    @volume.setter
    def volume(self, vol: np.array):
        self._vol = vol
        self._nz, self._h, self._w = vol.shape
        self._dtype = vol.dtype

        with self.calculating_semaphore:
            self._projected_img_2d = None
            self._img_2d_calculated = False

            self._grid(self._spac)

        self.eval_surf()

    def _grid(self, spacing):
        self._xl = [x * spacing - self._a - self._x0 for x in range(int(self._a * 3 / spacing))]
        self._yl = [y * spacing - self._b - self._y0 for y in range(int(self._b * 3 / spacing))]
        self._xv, self._yv = np.meshgrid(self._xl, self._yl)
        self._zv = np.zeros_like(self._xv)

    def eval_surf(self):
        if self._surf_eval:
            return

        self._r = R.from_euler('ZXY', [self._ptch, self._yaw, self._roll], degrees=True)
        self._R = self._r.as_matrix()
        self._Ri = self._r.inv().as_matrix()

        xv0, yv0 = self._xv, self._yv

        z2 = 1 - xv0 ** 2 / self._a2 - yv0 ** 2 / self._b2
        z = self._c * np.sqrt(z2)

        xx0 = np.array([self._x0, self._y0, self._z0])[:, None, None]
        rot10 = xx0 + np.einsum('ji, mni -> jmn', self._R, np.dstack([xv0, yv0, z]))
        rot11 = xx0 + np.einsum('ji, mni -> jmn', self._R, np.dstack([xv0, yv0, -z]))
        self._pts = np.array([np.concatenate((rot10[0], rot11[0])).ravel(),
                              np.concatenate((rot10[1], rot11[1])).ravel(),
                              np.concatenate((rot10[2], rot11[2])).ravel()])

        def _r(r):
            rgx = np.logical_and(r[0] >= 0, r[0] <= self._w)
            rgy = np.logical_and(r[1] >= 0, r[1] <= self._h)
            # idx = np.where(np.logical_and(rgx, rgy))
            r[0][~np.logical_and(rgx, rgy)] = np.nan
            r[1][~np.logical_and(rgx, rgy)] = np.nan
            r[2][~np.logical_and(rgx, rgy)] = np.nan

        self._pts_out = self._pts.copy()
        _r(self._pts_out)

        with self.calculating_semaphore:
            self.pts = self._pts.copy().astype(int)
            self.xl, self.yl, self.zl = self.pts
            self.xlo, self.ylo, self.zlo = self._pts_out

            self._surf_eval = True

    @property
    def projected_img_2d(self):
        if self._img_2d_calculated:
            return self._projected_img_2d
        if self._projected_img_2d is None:
            # self._projected_img_2d = np.zeros(shape=(self._w, self._h), dtype=self._dtype)
            self._projected_img_2d = np.zeros(shape=(int(self._w / self._spac), int(self._h / self._spac)),
                                              dtype=self._dtype)
        else:
            self._projected_img_2d[:, :] = 0

        xl, yl, zl = self._pts_out
        zf = np.array(zl)
        zix = np.logical_and(~np.isnan(zf), np.logical_and(0 <= zf, zf < self._nz))
        zf = np.floor(zf[zix]).astype(int)
        xf = np.array(xl).astype(int)[zix]
        yf = np.array(yl).astype(int)[zix]
        self._img_changes = 0
        changes = len(zf)
        if changes > 0:
            for xi, yi, zi in zip(xf, yf, zf):
                rx, ry = int(xi / self._spac), int(yi / self._spac)
                # if 0 <= xi < self._w and 0 <= yi < self._h and 0 <= zi < self._nz:
                try:
                    self._projected_img_2d[rx, ry] = self._vol[zi, xi, yi]
                except IndexError as e:
                    print(e)
                self._img_changes += 1

        # self._projected_img_2d = transform.resize(img_tmp, output_shape=self._projected_img_2d.shape)
        self._img_2d_calculated = True
        return self._projected_img_2d

    def project_2d(self):
        self.eval_surf()

        assert len(self.xl) == len(self.yl) and len(self.xl) == len(self.zl), "something happened in project_2d"
        return np.median(self.projected_img_2d), self._img_changes

    def _eval_params(self, p: Parameters):
        for name in p.keys():
            # print(f"self.{name} = {p[name].value}")
            exec(f"self.{name} = {p[name].value}")

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

    def optimize_parameters(self):
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
        result = fitter.minimize(method='bgfs', params=params)
        # result = fitter.minimize(method='basinhopping', params=params)  # , niter=10 ** 4, niter_success=1000)

        self._eval_params(result.params)
        print(result)
        print(result.params, result.residual)

        return result
