from threading import Semaphore

import numpy as np
import vtk
from lmfit import Minimizer
from lmfit import Parameters
from scipy.spatial.transform import Rotation as R
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkIOImage import vtkPNGWriter



class EllipsoidFit:
    _prop_keys: dict = {}

    def __init__(self, vtk_ellipsoid, pix_per_um, xyz_0=(0, 0, 0), sample_spacing=1, resampled_thickness=1):
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
        self._resampled_img_2d = None
        self._projected_img_2d = None
        self._img_2d_thick = min(resampled_thickness, 5)
        self._img_2d_calculated = False
        self._img_changes = 0
        self._dist_to_vol = np.inf
        self.local_search = None
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
            self._resampled_img_2d = None
            self._projected_img_2d = None
            self._img_2d_calculated = False

            self._grid()

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
            self._img_2d_calculated = False
            self._resampled_img_2d = None
            self._projected_img_2d = None

            self._grid()

        self.eval_surf()

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
        with self.calculating_semaphore:
            if self._img_2d_calculated:
                return self._projected_img_2d

            zrng = np.arange(start=-self._img_2d_thick / 2, stop=self._img_2d_thick / 2, step=1, dtype=np.int8)
            if self._projected_img_2d is None:
                self._resampled_img_2d = np.zeros(
                    shape=(int(self._img_2d_thick), int(self._w / self._spac), int(self._h / self._spac)),
                    dtype=self._dtype)
                self._projected_img_2d = np.zeros(shape=(int(self._w / self._spac), int(self._h / self._spac)),
                                                  dtype=self._dtype)
            else:
                self._resampled_img_2d[:, :, :] = 0
                self._projected_img_2d[:, :] = 0

            xl, yl, zl = self._pts_out
            zf = np.array(zl)
            zix = np.logical_and(~np.isnan(zf), np.logical_and(0 <= zf, zf < self._nz))
            zf = np.floor(zf[zix]).astype(int)
            xf = np.array(xl)[zix].astype(int)
            yf = np.array(yl)[zix].astype(int)

            self._img_changes = 0
            changes = len(zf)
            if changes == 0:
                return self._projected_img_2d

            neo_rx = np.floor(np.repeat(xf[:, None], len(zrng), axis=1) / self._spac).astype(int)
            neo_ry = np.floor(np.repeat(yf[:, None], len(zrng), axis=1) / self._spac).astype(int)
            neo_xl = np.repeat(xf[:, None], len(zrng), axis=1)
            neo_yl = np.repeat(yf[:, None], len(zrng), axis=1)
            neo_zl = np.repeat(zf[:, None], len(zrng), axis=1) + zrng
            neo_rz = np.zeros_like(neo_zl) + np.array(list(range(len(zrng))))
            nrz, nrx, nry = self._resampled_img_2d.shape
            for rx, ry, rz, xi, yi, zi in zip(neo_rx.ravel(), neo_ry.ravel(), neo_rz.ravel(),
                                              neo_xl.ravel(), neo_yl.ravel(), neo_zl.ravel()):
                try:
                    # if 0 <= xi < self._w and 0 <= yi < self._h and 0 <= zi < self._nz:
                    if xi < self._w and yi < self._h and zi < self._nz and \
                            rx < nrx and ry < nry and rz < nrz:
                        self._resampled_img_2d[rz, rx, ry] = self._vol[zi, xi, yi]
                except IndexError as e:
                    print(e)
                self._img_changes += 1

            # self._projected_img_2d = transform.resize(img_tmp, output_shape=self._projected_img_2d.shape)
            self._projected_img_2d = np.median(self._resampled_img_2d, axis=0)
            self._img_2d_calculated = True
            return self._projected_img_2d

    def project_2d(self):
        self.eval_surf()

        # assert len(self.xl) == len(self.yl) and len(self.xl) == len(self.zl), "something happened in project_2d"
        with self.calculating_semaphore:
            out0 = np.nansum(self._projected_img_2d), self._img_changes
        out = np.nansum(self.projected_img_2d), self._img_changes
        return out

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
        out = np.nan_to_num(100 / o0_den, posinf=1e5) + 0.1 * (dist_to_vol + in_z) if o0_den > 0 else 1e6
        xv = np.array([p[n].value for n in p.keys()])
        xv_str = np.array2string(xv, precision=1, suppress_small=True, floatmode='fixed')
        print(f"testing 1 f({xv_str})=100/(0.01* {chg} + {s} )+ {dist_to_vol + in_z:0.2f} ={out:0.6E}")

        return out

    def _accept_sol(self, x, f, accept):
        return self.stop

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
