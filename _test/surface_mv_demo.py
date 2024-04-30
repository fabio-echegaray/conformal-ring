from threading import Thread

import numpy as np
from mayavi import mlab
from mayavi.modules.surface import Surface
from mayavi.sources.parametric_surface import ParametricSurface
from scipy.spatial.transform import Rotation as R

from surface import EllipsoidFit

np.set_printoptions(precision=2)


class ThreadedAction(Thread):
    def __init__(self, ellipsoid: EllipsoidFit, **kwargs):
        Thread.__init__(self, **kwargs)
        self._e = ellipsoid

    def run(self):
        print("Fitting ellipsoid to data ...")
        self._e.optimize_parameters()
        print('done.')


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

    u0, u1, u2, theta = 1, 0, 0, np.pi / 2
    r = R.from_quat([u0, u1, u2, theta / 2])

    e = EllipsoidFit(source.parametric_function, xyz_0=(x0, y0, z0), sample_spacing=10)
    e.volume = np.empty(shape=(25, 512, 512))
    e.u0, e.u1, e.u2, e.theta = u0, u1, u2, theta

    points = mlab.points3d(e.xl, e.yl, e.zl, [1] * len(e.xl), color=(1, 0, 1), scale_factor=10)
    o = mlab.quiver3d(0, 1, 0, scale_factor=2 * b)

    o.actor.actor.orientation = r.as_euler('XYZ', degrees=True)

    actor = surface.actor  # mayavi actor, actor.actor is tvtk actor
    actor.property.opacity = 0.2
    actor.property.color = (0, 1, 1)  # tuple(np.random.rand(3))
    actor.mapper.scalar_visibility = False  # don't colour ellipses by their scalar indices into colour map
    actor.property.backface_culling = True  # gets rid of weird rendering artifact when opacity is < 1
    actor.property.specular = 0.1
    actor.actor.orientation = r.as_euler('YZX', degrees=True)
    actor.actor.origin = np.array([0, 0, 0])
    actor.actor.scale = np.array([1, 1, 1])
    actor.actor.position = np.array([0, 0, 0])
    actor.property.representation = ['wireframe', 'surface'][1]


    @mlab.animate(delay=200)
    def update_visualisation(srf, pts, o):
        theta = np.linspace(0, 2 * np.pi, 360)
        xr = np.linspace(-1000, 1000, 40)
        i = j = 0
        while True:
            i = (i + 1) % len(theta)
            j = (j + 1) % len(xr)
            x0, x1, x2 = xr[j], 0, 0
            r = R.from_quat([u0, u1, u2, np.cos(theta[i] / 2)])

            # order of angles follows [pitch, roll, yaw]
            srf.actor.actor.orientation = r.as_euler('YZX', degrees=True)
            o.actor.actor.orientation = r.as_euler('YZX', degrees=True)

            # --------------------------------------------------------------
            #  DEBUG info
            # --------------------------------------------------------------
            print()
            tq = np.array([*pts.actor.actor.orientation_wxyz[1:4], pts.actor.actor.orientation_wxyz[0]])
            print(f"them bf {pts.actor.actor.orientation} {tq}")

            mq = np.array([*r.as_quat(canonical=False)[0:3], np.rad2deg(r.as_quat(canonical=False)[3])])
            tq = np.array([*pts.actor.actor.orientation_wxyz[1:4], pts.actor.actor.orientation_wxyz[0]])

            print(f"mine af {r.as_euler('YZX', degrees=True)} {mq}")

            r2 = R.from_euler('YZX', np.deg2rad(pts.actor.actor.orientation))
            mq2 = np.array([*r2.as_quat(canonical=False)[0:3], np.rad2deg(r2.as_quat(canonical=False)[3])])
            print(f"mine a2 {r2.as_euler('YZX', degrees=True)} {mq2}")

            print(f"them af {pts.actor.actor.orientation} {tq}")
            # dot product should yield 1 if q1==q2
            print(f"logical check if q1==q2 {np.dot(r.as_quat(canonical=False), r2.as_quat(canonical=False))}")

            # recompute points on surface given new coordinates
            e.x0, e.y0, e.z0 = x0, x1, x2
            e.u0, e.u1, e.u2 = u1, -u2, u0
            e.theta = theta[i]
            e.eval_surf()

            pts.mlab_source.set(x=e.xl, y=e.yl, z=e.zl)

            yield


    update_visualisation(surface, points, o)
    axes = mlab.orientation_axes()

    mlab.view(azimuth=90, elevation=90, distance=10000, focalpoint=(-0., 1.05, 1.05))
    mlab.show()
