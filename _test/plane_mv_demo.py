import numpy as np
from mayavi import mlab
from mayavi.modules.surface import Surface
from mayavi.sources.parametric_surface import ParametricSurface
from scipy.spatial.transform import Rotation as R

from surface import Plane

np.set_printoptions(precision=2)

if __name__ == "__main__":
    mlab.figure(1, bgcolor=(0, 0, 0), size=(500, 500))
    mlab.clf()

    engine = mlab.get_engine()
    source = ParametricSurface()
    source.function = 'ellipsoid'
    engine.add_source(source)

    a, b, c = 512, 512 * 2, 300
    source.parametric_function.x_radius = a * 2
    source.parametric_function.y_radius = b * 2
    source.parametric_function.z_radius = c * 2
    x0, y0, z0 = 0, 0, 0

    surface = Surface()
    source.add_module(surface)

    pry = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 1]
    ])

    e = Plane(xyz_0=(x0, y0, z0), sample_spacing=50)
    e.pitch, e.roll, e.yaw = pry[0] * 90
    e.volume = np.empty(shape=(25, a * 4, b * 2))
    points = mlab.points3d(e.xl, e.yl, e.zl, [1] * len(e.xl), color=(1, 0, 1), scale_factor=10)
    minx, max = -a, a
    miny, maxy = -b, b
    px0y0 = mlab.points3d([minx], [miny], [0], [1], color=(1, 0, 0), scale_factor=100)
    px1y0 = mlab.points3d([max], [miny], [0], [1], color=(0, 1, 0), scale_factor=100)
    px0y1 = mlab.points3d([minx], [maxy], [0], [1], color=(0, 0, 1), scale_factor=100)
    px1y1 = mlab.points3d([max], [maxy], [0], [1], color=(0, 1, 1), scale_factor=100)

    # o = mlab.quiver3d(u1, -u2, u0, scale_factor=2 * b)
    o = mlab.quiver3d(0, 1, 0, scale_factor=2 * b)

    actor = surface.actor  # mayavi actor, actor.actor is tvtk actor
    actor.property.opacity = 0.2
    actor.property.color = (0, 1, 1)  # tuple(np.random.rand(3))
    actor.mapper.scalar_visibility = False  # don't colour ellipses by their scalar indices into colour map
    actor.property.backface_culling = True  # gets rid of weird rendering artifact when opacity is < 1
    actor.property.specular = 0.1
    # actor.actor.orientation = r.as_euler('YZX', degrees=True)
    actor.actor.origin = np.array([0, 0, 0])
    actor.actor.scale = np.array([1, 1, 1])
    actor.actor.position = np.array([x0, y0, z0])
    actor.property.representation = ['wireframe', 'surface'][1]

    t = mlab.text(.3, .05, "holi", width=.05)


    @mlab.animate(delay=100)
    def update_visualisation(srf, pts, o):
        theta = np.linspace(-np.pi, np.pi, 50)
        xr = np.linspace(-1000, 1000, 40)
        i = j = k = 0
        while True:
            i = (i + 1) % len(theta)
            j = (j + 1) % len(xr)
            if i == 0:
                k = (k + 1) % len(pry)
            x0, y0, z0 = 0, 0, 0

            pitch, roll, yaw = pry[k] * np.rad2deg(theta[i])
            # r = R.from_euler('YXZ', [pitch, roll, yaw], degrees=True)
            r = R.from_euler('ZXY', [yaw, roll, pitch], degrees=True)

            # order of angles follows [yaw, pitch, roll]
            # for el in [srf, o, px0y0, px0y1, px1y0, px1y1]:
            for el in [srf, o]:
                # el.actor.actor.orientation = r.as_euler('ZXY', degrees=True)
                el.actor.actor.orientation = [yaw, roll, pitch]  # r.as_euler('ZXY', degrees=True)
                el.actor.actor.position = np.array([x0, y0, z0])

            t.text = f"{k}"

            # recompute points on surface given new coordinates
            e.x0, e.y0, e.z0 = x0, y0, z0
            e.pitch, e.roll, e.yaw = pitch, roll, yaw

            e.eval_surf()

            pts.mlab_source.set(x=e.xl, y=e.yl, z=e.zl)

            d, h, w = np.array(e.test.shape) - 1
            px0y0.mlab_source.set(x=e.test[0, 0, 0], y=e.test[1, 0, 0], z=e.test[2, 0, 0])
            px0y1.mlab_source.set(x=e.test[0, 0, h], y=e.test[1, 0, h], z=e.test[2, 0, h])
            px1y0.mlab_source.set(x=e.test[0, w, 0], y=e.test[1, w, 0], z=e.test[2, w, 0])
            px1y1.mlab_source.set(x=e.test[0, w, h], y=e.test[1, w, h], z=e.test[2, w, h])

            yield


    update_visualisation(surface, points, o)
    axes = mlab.orientation_axes()

    mlab.view(azimuth=90, elevation=90, distance=10000, focalpoint=(-0., 1.05, 1.05))
    mlab.show()
