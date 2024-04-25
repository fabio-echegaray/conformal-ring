import itertools
from pathlib import Path

import numpy as np
import vtk
from mayavi import mlab
from mayavi.modules.surface import Surface
from mayavi.sources.parametric_surface import ParametricSurface
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkIOImage import vtkPNGWriter

from fileops.export.config import create_cfg_file, read_config


class Ellipsoid:
    def __init__(self, vtk_ellipsoid, xyz_0=(0, 0, 0)):
        self._vtk_ellipsoid = vtk_ellipsoid
        self._a = vtk_ellipsoid.x_radius
        self._b = vtk_ellipsoid.y_radius
        self._c = vtk_ellipsoid.z_radius
        self._a2 = self._a ** 2
        self._b2 = self._b ** 2
        self._c2 = self._c ** 2
        self._x0 = xyz_0[0]
        self._y0 = xyz_0[1]
        self._z0 = xyz_0[2]

    def z(self, x, y):
        z2 = 1 - (x - self._x0) ** 2 / self._a2 - (y - self._y0) ** 2 / self._b2
        z = self._c * np.sqrt(z2)
        return [self._z0 - z, self._z0 + z]


if __name__ == "__main__":
    base_exp_pth = Path("/Users/fabio/data/")
    base_cache_pth = Path("/Users/fabio/dev/Python-AgentSegmentation/out")
    exp_name = "20231130 - JupMCh SqhGFP/CHX-500ug_1"
    exp_file = f"CHX-500ug_1_MMStack_Default_1.ome.tif"
    img_path = base_exp_pth / exp_name / exp_file

    cfg_path = Path('../test_vol.cfg')
    if not cfg_path.exists():
        create_cfg_file(path=cfg_path,
                        contents={
                            "DATA": {
                                "image": img_path.as_posix(),
                                "series": 0,  # TODO: change
                                "channel": [0, 1],  # TODO: change
                                "frame": 1
                            }
                        })
    cfg = read_config(cfg_path)

    # img_path = Path('./tiff/ch1/ch1_fr001.tiff')
    # if not img_path.exists():
    #     export_vtk(cfg, None)
    # # Load the data
    # img = load_image_file(Path('./tiff/ch1/ch1_fr001.tiff'))
    imgser = cfg.image_file.image_series(channel=1, zstack='all', frame=31, as_8bit=False)

    mlab.figure(1, bgcolor=(0, 0, 0), size=(500, 500))
    mlab.clf()

    data = imgser.images[0].reshape((imgser.zstacks, imgser.width, imgser.height))
    source = mlab.pipeline.scalar_field(data.T)
    source.spacing = [1, 1, 1]
    min = data.min()
    max = data.max()
    vol = mlab.pipeline.volume(source,
                               vmin=min + 0.3 * (max - min),
                               vmax=min + 0.9 * (max - min))

    # vol.volume.rotate_y(90)

    engine = mlab.get_engine()
    source = ParametricSurface()
    source.function = 'ellipsoid'
    engine.add_source(source)

    a, b, c = 90 * cfg.image_file.pix_per_um, 250 * cfg.image_file.pix_per_um, 90 * cfg.image_file.pix_per_um
    source.parametric_function.x_radius = a * 2
    source.parametric_function.y_radius = b * 2
    source.parametric_function.z_radius = c * 2
    x0, y0, z0 = a, b, 2 * c

    u = [1, 0, 1]
    x = [0, 0, 0]
    Du = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    rs = source.parametric_function.evaluate(u, x, Du)

    min_u, max_u = source.parametric_function.minimum_u, source.parametric_function.maximum_u
    min_v, max_v = source.parametric_function.minimum_v, source.parametric_function.maximum_v
    surface = Surface()
    source.add_module(surface)

    e = Ellipsoid(source.parametric_function, xyz_0=(x0, y0, z0))
    xl, yl, zl = [], [], []
    for xi, yi in itertools.product(range(cfg.image_file.width), range(cfg.image_file.height)):
        xl.append(xi)
        yl.append(yi)
        zl.append(e.z(xi, yi)[0])
    mlab.points3d(xl, yl, zl, [1] * len(xl), color=(1, 0, 1), scale_factor=1)

    # obtain projection of volumetric data onto 3D surface
    img = np.zeros((cfg.image_file.width, cfg.image_file.height), dtype=cfg.image_file.image(0).image.dtype)
    changes = 0
    for xi, yi, zi in zip(xl, yl, np.floor(np.array(zl)).astype(int)):
        if 0 <= zi <= cfg.image_file.n_zstacks:
            # print(data[xi, yi, zi])
            img[xi, yi] = data[zi, xi, yi]
            changes += 1

    # write image to PNG
    depth_array = numpy_support.numpy_to_vtk(img.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    depth_array.SetNumberOfComponents(1)

    imagedata = vtkImageData()
    imagedata.SetSpacing([1, 1, 1])
    imagedata.SetOrigin([-1, -1, -1])
    imagedata.SetDimensions(cfg.image_file.width, cfg.image_file.height, 1)
    imagedata.GetPointData().SetScalars(depth_array)

    writer = vtkPNGWriter()
    writer.SetInputData(imagedata)
    writer.SetFileName("projection.png")
    writer.Write()

    actor = surface.actor  # mayavi actor, actor.actor is tvtk actor
    # actor.property.ambient = 1 # defaults to 0 for some reason, ah don't need it, turn off scalar visibility instead
    actor.property.opacity = 0.1
    actor.property.color = (0, 1, 1)  # tuple(np.random.rand(3))
    actor.mapper.scalar_visibility = False  # don't colour ellipses by their scalar indices into colour map
    actor.property.backface_culling = True  # gets rid of weird rendering artifact when opacity is < 1
    actor.property.specular = 0.1
    # actor.property.frontface_culling = True
    actor.actor.orientation = np.array([1, 0, 0]) * 360  # in degrees
    actor.actor.origin = np.array([0, 0, 0])
    # actor.actor.position = np.array([0, 0, 0])
    actor.actor.scale = np.array([1, 1, 1])
    # actor.actor.origin = np.array([0, 0, 0])
    actor.actor.position = np.array([x0, y0, z0])
    # actor.actor.scale = np.array([e0, e1, e2])
    actor.enable_texture = True
    actor.property.representation = ['wireframe', 'surface'][1]

    # mlab.view(132, 54, 45, [21, 20, 21.5])
    # mlab.axes()
    mlab.orientation_axes()

    mlab.show()
