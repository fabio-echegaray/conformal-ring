import itertools
from threading import Semaphore

import numpy as np
from scipy.optimize import basinhopping
from scipy.spatial.transform import Rotation as R


class Plane:
    _prop_keys: dict = {}

    def __init__(self, xyz_0=(0, 0, 0), sample_spacing=1):
        self._spac = sample_spacing

        self._vol = None
        self._w = 0
        self._h = 0
        self._nz = 0
        self._dtype = None

        self._x0 = xyz_0[0]
        self._y0 = xyz_0[1]
        self._z0 = xyz_0[2]
        self._X = None

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
        self.projected_img_2d = None
        self.calculating_semaphore = Semaphore()

        self._prop_keys = {
            "x0": "self._x0",
            "y0": "self._y0",
            "z0": "self._z0",
            "roll": "self._roll",
            "pitch": "self._ptch",
            "yaw": "self._yaw",
        }

    def __setattr__(self, name, value):
        if name in self._prop_keys.keys():
            exec(f"{self._prop_keys[name]} = {value}")
            if name[2:8] == "radius":
                exec(f"{self._prop_keys[name]}2 = {value ** 2}")
            self._surf_eval = False
        else:
            super().__setattr__(name, value)

    def state(self):
        return self._x0, self._y0, self._z0, self._roll, self._ptch, self._yaw

    @property
    def volume(self):
        return self._vol

    @volume.setter
    def volume(self, vol: np.array):
        self._vol = vol
        self._nz, self._h, self._w = vol.shape
        self._dtype = vol.dtype

        self.projected_img_2d = np.zeros(shape=(self._w, self._h), dtype=self._dtype)
        self.eval_surf()

    def eval_surf(self):
        if self._surf_eval:
            return

        self._r = R.from_euler('ZXY', [self._ptch, self._yaw, self._roll], degrees=True)
        self._R = self._r.as_matrix()
        self._Ri = self._r.inv().as_matrix()

        x0 = int(self._w / 2)
        y0 = int(self._h / 2)
        xl = [x * self._spac - x0 for x in range(int(self._w / self._spac))]
        yl = [y * self._spac - y0 for y in range(int(self._h / self._spac))]
        xv, yv = np.meshgrid(xl, yl)
        zv = np.zeros_like(xv)

        rot0 = np.einsum('ji, mni -> jmn', self._R, np.dstack([xv, yv, zv]))
        # rot1 = np.einsum('ji, mni -> jmn', self._Ri, np.dstack(rot0))
        self.test = rot0

        out = rot0
        # out[0] += x0
        # out[1] += y0
        self._pts = np.array([out[0].ravel(), out[1].ravel(), out[2].ravel()])
        with self.calculating_semaphore:
            self.pts = self._pts.copy().astype(int)
            self.xl, self.yl, self.zl = self.pts

        self._surf_eval = True

    def project_2d(self):
        self.eval_surf()

        self.projected_img_2d[:, :] = 0

        zf = np.array(self.zl)
        zix = np.logical_and(~np.isnan(zf), np.logical_and(0 <= zf, zf < self._nz))
        zf = np.floor(zf[zix]).astype(int)
        changes = len(zf)
        if changes > 0:
            for xi, yi, zi in zip(np.array(self.xl / self._spac).astype(int)[zix],
                                  np.array(self.yl / self._spac).astype(int)[zix], zf):
                # if 0 <= xi < self._w and 0 <= yi < self._h and 0 <= zi < self._nz:
                self.projected_img_2d[xi, yi] = self._vol[zi, xi, yi]
                changes += 1

        assert len(self.xl) == len(self.yl) and len(self.xl) == len(self.zl), "something happened in project_2d"
        return np.sum(self.projected_img_2d), changes

    def _obj_fn_minimize(self, xv):
        self.x0, self.y0, self.z0, self.x_radius, self.y_radius, self.z_radius, self.u0, self.u1, self.u2, self.theta = xv
        s, chg = self.project_2d()
        out = 1 / (0.1 * chg + s)
        print(f"testing f({xv})=1/(0.1*{chg}+{s})={out}")

        return out

    def optimize_parameters(self):
        param_bounds = ((0, 2 * self._w), (0, 2 * self._h), (0, 10 * self._nz),
                        (0, 3 * self._w), (0, 3 * self._h), (0, 100 * self._nz),
                        (0, 1), (0, 1), (0, 1), (-np.pi, np.pi))
        x0 = [250, 250, 100, 200, 500, 200, 0, 0, 0, np.pi]
        # res = basinhopping(self._obj_fn_minimize, x0, minimizer_kwargs={'bounds': param_bounds, 'args': (yn, Np)})
        res = basinhopping(self._obj_fn_minimize, x0, stepsize=10, T=10, minimizer_kwargs={'bounds': param_bounds})
        # res = basinhopping(self._obj_fn_minimize, x0)
        print(res.x)
        # objf = self._obj_fn_minimize(res.x, yn)
        return res
