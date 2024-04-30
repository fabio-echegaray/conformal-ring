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
from threading import Thread
from scipy.spatial.transform import Rotation as R

from fileops.export.config import create_cfg_file, read_config

from surface import EllipsoidFit


class ThreadedAction(Thread):
    def __init__(self, ellipsoid: EllipsoidFit, **kwargs):
        Thread.__init__(self, **kwargs)
        self._e = ellipsoid

    def run(self):
        print("Fitting ellipsoid to data ...")
        self._e.optimize_parameters()
        print('done.')


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
                                "series": 0,
                                "channel": [0, 1],
                                "frame": 1
                            }
                        })
    cfg = read_config(cfg_path)

    imgser = cfg.image_file.image_series(channel=1, zstack='all', frame=31, as_8bit=False)

    mlab.figure(1, bgcolor=(0, 0, 0), size=(500, 500))
    mlab.clf()

    data = imgser.images[0].reshape((imgser.zstacks, imgser.width, imgser.height))
    source = mlab.pipeline.scalar_field(data.T)
    source.spacing = [1, 1, 10]
    min = data.min()
    max = data.max()
    vol = mlab.pipeline.volume(source,
                               vmin=min + 0.3 * (max - min),
                               vmax=min + 0.9 * (max - min))

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

    e = EllipsoidFit(source.parametric_function, xyz_0=(x0, y0, z0))
    e.volume = data
    e.eval_surf()
    points = mlab.points3d(e.xl, e.yl, e.zl, [1] * len(e.xl), color=(1, 0, 1), scale_factor=1)

    # obtain projection of volumetric data onto 3D surface
    # img, changes = e.project_2d()

    # write image to PNG
    # depth_array = numpy_support.numpy_to_vtk(img.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)
    # depth_array.SetNumberOfComponents(1)

    # imagedata = vtkImageData()
    # # imagedata.SetSpacing([1, 1, 1])
    # # imagedata.SetOrigin([-1, -1, -1])
    # imagedata.SetDimensions(cfg.image_file.width, cfg.image_file.height, 1)
    # imagedata.GetPointData().SetScalars(depth_array)
    #
    # writer = vtkPNGWriter()
    # writer.SetInputData(imagedata)
    # writer.SetFileName("projection.png")
    # writer.Write()

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
    actor.actor.scale = np.array([1, 1, 1])
    actor.actor.position = np.array([x0, y0, z0])
    actor.enable_texture = True
    actor.property.representation = ['wireframe', 'surface'][1]

    mlab.orientation_axes()

    # mlab.figure(2, bgcolor=(0, 0, 0), size=(500, 500))
    # mlab.imshow(img)

    te = ThreadedAction(e)
    te.start()


    @mlab.animate(delay=2000)
    def update_visualisation(srf, pts):
        while True:
            with e.calculating_semaphore:
                print(f'Updating Visualisation {np.round(e.state(), 1)} ({len(e.xl)} {len(e.yl)} {len(e.zl)})')
                assert len(e.xl) == len(e.yl) and len(e.xl) == len(e.zl), "strange cat"
                pts.mlab_source.set(x=e.xl, y=e.yl, z=e.zl)

            x0, y0, z0, a, b, c, u0, u1, u2, theta = e.state()

            r = R.from_quat([u0, u1, u2, theta])

            actor = srf.actor  # mayavi actor, actor.actor is tvtk actor
            actor.actor.orientation = r.as_rotvec(degrees=True)
            actor.actor.position = np.array([x0, y0, z0])
            # actor.actor.scale = np.array([a, b, c])

            yield


    update_visualisation(surface, points)
    mlab.show()
