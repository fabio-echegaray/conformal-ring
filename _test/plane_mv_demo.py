from pathlib import Path

import numpy as np
from PyQt5 import QtCore
from fileops.export.config import create_cfg_file, read_config
from mayavi import mlab
from skimage import transform
from skimage.filters import rank
from skimage.morphology import disk

from surface import PlaneFit

np.set_printoptions(precision=3, linewidth=500)

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
                                "frame": 38
                            }
                        })
    cfg = read_config(cfg_path)

    imgser = cfg.image_file.image_series(channel=1, zstack='all', frame=38, as_8bit=False)

    fig = mlab.figure(1, bgcolor=(0, 0, 0), size=(500, 500))
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

    # generate a plane at the middle of the volume
    x, y = np.mgrid[0:imgser.width:4j, 0:imgser.height:4j]
    z = np.zeros_like(x) + imgser.zstacks / 2
    plane = mlab.surf(x, y, z)

    spac = 5
    x0, y0, z0 = 0, 0, imgser.zstacks / 2
    e = PlaneFit(a=0, b=0, c=1, pix_per_um=cfg.image_file.pix_per_um, xyz_0=(x0, y0, z0),
                 sample_spacing=spac, resampled_thickness=3)
    e.volume = data
    x0, y0, z0, roll, pitch, yaw, a, b, c = e.state()

    points = mlab.points3d(e.xl, e.yl, e.zl, [1] * len(e.xl), color=(1, 0, 1), scale_factor=1)

    mlab.orientation_axes()
    mlab.axes()

    actor = plane.actor  # mayavi actor, actor.actor is tvtk actor
    actor.property.opacity = 0.1
    actor.property.color = (0, 1, 1)
    actor.mapper.scalar_visibility = False  # don't colour ellipses by their scalar indices into colour map
    actor.property.backface_culling = True  # gets rid of weird rendering artifact when opacity is < 1
    actor.property.specular = 0.1
    actor.actor.orientation = [yaw, roll, pitch]
    actor.actor.origin = np.array([0, 0, 0])
    actor.actor.scale = np.array([1, 1, 1])
    actor.actor.position = np.array([x0, y0, 10 * z0])
    actor.enable_texture = True
    actor.property.representation = ['wireframe', 'surface'][1]

    img_ = transform.resize(data[10], output_shape=[int(k / spac) for k in data[10].shape])
    fig2 = mlab.figure(2, bgcolor=(0, 0, 0), size=(500, 500))
    img = mlab.imshow(img_)
    mlab.view(azimuth=0, elevation=0, distance=276, focalpoint=(-0.5, -0.5, 0.0))


    def key_press(obj, event):
        key = obj.GetKeySym()  # works fine
        if key == "y":
            e.z0 = e._z0 + 1
        elif key == "h":
            e.z0 = e._z0 - 1
        if key == "i":
            e.y0 = e._y0 + 1
        elif key == "k":
            e.y0 = e._y0 - 1
        if key == "j":
            e.x0 = e._x0 + 1
        elif key == "l":
            e.x0 = e._x0 - 1
        elif key == "r":
            e.a = e._a + 0.001
        elif key == "f":
            e.a = e._a - 0.001
        elif key == "t":
            e.b = e._b + 0.001
        elif key == "g":
            e.b = e._b - 0.001
        e.eval_surf()


    fig.scene.interactor.remove_observers("KeyPressEvent")
    fig.scene.interactor.remove_observers("KeyReleaseEvent")
    fig.scene.interactor.add_observer("KeyPressEvent", key_press)


    @mlab.animate(delay=1000, ui=True)
    def update_visualisation():
        while not e.stop:
            footprint = disk(60)
            img_8bit = ((e.projected_img_2d - e.projected_img_2d.min()) / (
                    e.projected_img_2d.ptp() / 255.0)).astype(np.uint8)
            # percentile_result = rank.mean_percentile(img_8bit, footprint=footprint, p0=0.1, p1=0.9)
            bilateral_result = rank.mean_bilateral(img_8bit, footprint=footprint, s0=500, s1=500)
            median_result = rank.median(img_8bit, footprint=footprint)
            mean_result = rank.mean(img_8bit, footprint=footprint)

            print(f'Updating Visualisation {np.round(e.state(), 1)}\n'
                  f'sum {np.sum(e.projected_img_2d):0.1f} '
                  f'mean {np.mean(e.projected_img_2d):0.1f} '
                  f'median {np.median(e.projected_img_2d):0.1f} '
                  f'ptp {np.ptp(e.projected_img_2d):0.1f} '
                  f'mean_c {np.sum(mean_result):0.1f} '
                  f'median_c {np.sum(median_result):0.1f} '
                  f'bilateral_c {np.sum(bilateral_result):0.1f}')

            points.mlab_source.scalars = None
            points.mlab_source.reset(x=e.xl, y=e.yl, z=10 * e.zl)

            # z = -1 / c * (a * (x - x0) + b * (y - y0))
            # plane.mlab_source.set(x=x - x0, y=y - y0, z=10 * z)
            plane.mlab_source.set(x=e.xl, y=e.yl, z=10 * e.zl)

            img.mlab_source.scalars = np.fliplr(e.projected_img_2d)

            yield


    @QtCore.pyqtSlot()
    def on_finish():
        anim.timer.Stop()
        # mlab.close(all=True)
        print("all should be closed now.")
        e.save_projection()
        mlab.figure(3, bgcolor=(0, 0, 0), size=(500, 500))
        e.sample_spacing = 1
        mlab.imshow(e.projected_img_2d)


    anim = update_visualisation()
    mlab.orientation_axes()

    try:
        mlab.show()
    except KeyboardInterrupt:
        e.stop = True
