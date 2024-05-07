from pathlib import Path
from threading import Thread

import numpy as np
from fileops.export.config import create_cfg_file, read_config
from mayavi import mlab
from mayavi.modules.surface import Surface
from mayavi.sources.parametric_surface import ParametricSurface
from skimage import transform

from surface import EllipsoidFit

np.set_printoptions(precision=2, linewidth=500)


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
    source.spacing = [1, 1, 1]
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
    a, b, c = 2 * a, b, 2 * c
    source.parametric_function.x_radius = a * 2
    source.parametric_function.y_radius = b * 2
    source.parametric_function.z_radius = c * 2
    x0, y0, z0 = a, b, 2 * c

    surface = Surface()
    source.add_module(surface)

    spac = 50
    e = EllipsoidFit(source.parametric_function, cfg.image_file.pix_per_um, xyz_0=(x0, y0, z0), sample_spacing=spac)
    e.volume = data
    x0, y0, z0, a, b, c, roll, pitch, yaw = e.state()

    e.eval_surf()
    points = mlab.points3d(e.xl, e.yl, e.zl, [1] * len(e.xl), color=(1, 0, 1), scale_factor=10)
    points_rect = mlab.points3d(e.xl, e.yl, e.zl, [1] * len(e.xl), color=(1, 0, 0), scale_factor=100)

    mlab.orientation_axes()

    actor = surface.actor  # mayavi actor, actor.actor is tvtk actor
    actor.property.opacity = 0.2
    actor.property.color = (0, 1, 1)  # tuple(np.random.rand(3))
    actor.mapper.scalar_visibility = False  # don't colour ellipses by their scalar indices into colour map
    actor.property.backface_culling = True  # gets rid of weird rendering artifact when opacity is < 1
    actor.property.specular = 0.1
    actor.actor.orientation = [yaw, roll, pitch]  # r.as_euler('ZXY', degrees=True)
    actor.actor.origin = np.array([0, 0, 0])
    actor.actor.scale = np.array([1, 1, 1])
    actor.actor.position = np.array([x0, y0, z0])
    actor.enable_texture = True
    actor.property.representation = ['wireframe', 'surface'][1]

    img_ = transform.resize(data[10], output_shape=[int(k / spac) for k in data[10].shape])
    mlab.figure(2, bgcolor=(0, 0, 0), size=(500, 500))
    # img = mlab.imshow(data[10])  # , colormap="gray")
    img = mlab.imshow(img_)

    te = ThreadedAction(e)
    te.start()


    @mlab.animate(delay=500)
    def update_visualisation(srf, pts):
        while True:
            x0, y0, z0, a, b, c, roll, pitch, yaw = e.state()
            # print(f'Updating Visualisation {np.round(e.state(), 1)} ({len(e.xl)} {len(e.yl)} {len(e.zl)})')

            e.eval_surf()
            pts.mlab_source.set(x=e.xl, y=e.yl, z=e.zl)
            points_rect.mlab_source.set(x=e.xlo, y=e.ylo, z=e.zlo)

            source.parametric_function.x_radius = a
            source.parametric_function.y_radius = b
            source.parametric_function.z_radius = c

            img.mlab_source.scalars = e.projected_img_2d

            for el in [srf, ]:
                el.actor.actor.orientation = [yaw, roll, pitch]  # r.as_euler('ZXY', degrees=True)
                el.actor.actor.position = np.array([x0, y0, z0])
                # actor.actor.scale = np.array([a, b, c])

            yield


    update_visualisation(surface, points)
    mlab.orientation_axes()
    mlab.show()
