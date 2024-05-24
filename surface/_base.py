from threading import Semaphore

import numpy as np
import vtk
from lmfit import Parameters
from scipy.spatial.transform import Rotation as R
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkIOImage import vtkPNGWriter


class BaseFit:
    """
    This class defines the basic structure for derived classes that need to perform some surface fitting onto data.
    """
    _prop_keys: dict = {}

    def __init__(self, *args, pix_per_um=1, xyz_0=(0, 0, 0), sample_spacing=1, resampled_thickness=1, **kwargs):

        self._ppu = pix_per_um
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
        self._xl = None
        self._yl = None
        self._xv = None
        self._yv = None
        self._zv = None
        self.xl = None  # all calculated x points as a list
        self.yl = None  # all calculated y points as a list
        self.zl = None  # all calculated z points as a list
        self.xlo = None  # filtered points x for output
        self.ylo = None  # filtered points y for output
        self.zlo = None  # filtered points z for output
        self._pts = None
        self.pts = None
        self._pts_out = None

        # euler angles
        self._roll = 0
        self._ptch = 0
        self._yaw = 0
        self._r: R = None
        self._R: np.array = None
        self._Ri: np.array = None

        self._surf_eval = False

        self._resampled_img_2d = None
        self._projected_img_2d = None
        self._img_2d_thick = min(resampled_thickness, 5)
        self._img_2d_calculated = False
        self._img_changes = 0
        self._dist_to_vol = np.inf
        self.local_search = None
        self.calculating_semaphore = Semaphore()
        self.stop = False

        self._prop_keys.update({
            "x0": "self._x0",
            "y0": "self._y0",
            "z0": "self._z0",
            "roll": "self._roll",
            "pitch": "self._ptch",
            "yaw": "self._yaw",
        })

    def __setattr__(self, name, value):
        if name in self._prop_keys.keys():
            exec(f"{self._prop_keys[name]} = {value}")
            self.ask_recalc()
        else:
            super().__setattr__(name, value)

    def state(self):
        return self._x0, self._y0, self._z0, self._roll, self._ptch, self._yaw

    def ask_recalc(self):
        with self.calculating_semaphore:
            self._resampled_img_2d = None
            self._projected_img_2d = None
            self._img_2d_calculated = False
            self._surf_eval = False

            self._r = None
            self._R = None
            self._Ri = None

    @property
    def sample_spacing(self):
        return self._spac

    @sample_spacing.setter
    def sample_spacing(self, spacing: int):
        self._spac = spacing
        self.ask_recalc()
        self._grid()
        self.eval_surf()

    def _grid(self):
        s = self._spac
        if self.local_search is not None:
            ls = self.local_search
            self._xl = [x * s - (self._w + ls) / 2 for x in range(int((self._w + ls) / s))]
            self._yl = [y * s - (self._h + ls) / 2 for y in range(int((self._h + ls) / s))]
        else:
            self._xl = [x * s - self._w / 2 - self._x0 for x in range(int(self._w / s) + 1)]
            self._yl = [y * s - self._h / 2 - self._y0 for y in range(int(self._h / s) + 1)]
        self._xv, self._yv = np.meshgrid(self._xl, self._yl)
        self._zv = np.zeros_like(self._xv)

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

    def eval_surf(self):
        pass

    def rigid_transform_points(self, xv, yv, zv):
        """
        Rotate and translate points. Then, filter the subset that is within the constraints of the volume.
        :param xv:
        :param yv:
        :param zv:
        :return: np.array of all transformed points and also an np.array of the points filtered within the boundaries
                 of the volume.
        """
        if self._r is None:
            with self.calculating_semaphore:
                self._r = R.from_euler('ZXY', [self._ptch, self._yaw, self._roll], degrees=True)
                self._R = self._r.as_matrix()
                self._Ri = self._r.inv().as_matrix()

        xx0 = np.array([self._x0, self._y0, self._z0])[:, None, None]
        rot = xx0 + np.einsum('ji, mni -> jmn', self._R, np.dstack([xv, yv, zv]))
        _pts = np.array([rot[0].ravel(), rot[1].ravel(), rot[2].ravel()])

        def _r(r):
            rgx = np.logical_and(r[0] >= 0, r[0] <= self._w)
            rgy = np.logical_and(r[1] >= 0, r[1] <= self._h)
            # idx = np.where(np.logical_and(rgx, rgy))
            r[0][~np.logical_and(rgx, rgy)] = np.nan
            r[1][~np.logical_and(rgx, rgy)] = np.nan
            r[2][~np.logical_and(rgx, rgy)] = np.nan

        _pts_out = _pts.copy()
        _r(_pts_out)

        with self.calculating_semaphore:
            self._pts, self._pts_out = _pts, _pts_out

        return self._pts, self._pts_out

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

    def save_projection(self, name="projection.png"):
        # obtain projection of volumetric data onto 3D surface
        _s_spac = self.sample_spacing
        self.sample_spacing = 1

        # write image to PNG
        depth_array = numpy_support.numpy_to_vtk(self.projected_img_2d.ravel(), deep=True,
                                                 array_type=vtk.VTK_UNSIGNED_SHORT)
        depth_array.SetNumberOfComponents(1)

        imagedata = vtkImageData()
        imagedata.SetSpacing([1, 1, 1])
        imagedata.SetOrigin([-1, -1, -1])
        imagedata.SetDimensions(self._w, self._h, 1)
        imagedata.GetPointData().SetScalars(depth_array)

        writer = vtkPNGWriter()
        writer.SetInputData(imagedata)
        writer.SetFileName(name)
        writer.Write()

        self.sample_spacing = _s_spac

    def _eval_params(self, p: Parameters):
        for name in p.keys():
            # print(f"self.{name} = {p[name].value}")
            exec(f"self.{name} = {p[name].value}")
