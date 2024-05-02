import numpy as np
from mayavi import mlab
from mayavi.modules.surface import Surface
from mayavi.sources.parametric_surface import ParametricSurface
from scipy.spatial.transform import Rotation as R

from surface import EllipsoidFit

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

    w, h = 512, 512
    e = EllipsoidFit(source.parametric_function, xyz_0=(x0, y0, z0), sample_spacing=50)
    e.pitch, e.roll, e.yaw = pry[0] * 90
    e.volume = np.empty(shape=(25, w, h))
    points = mlab.points3d(e.xl, e.yl, e.zl, [1] * len(e.xl), color=(1, 0, 1), scale_factor=10)

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
        theta = np.linspace(- np.pi, np.pi, 100)
        xr = np.linspace(-1000, 1000, 40)
        i = j = k = 0
        while True:
            i = (i + 1) % len(theta)
            j = (j + 1) % len(xr)
            if i == 0:
                k = (k + 1) % len(pry)
            x0, y0, z0 = 0, 0, 0

            pitch, roll, yaw = pry[k] * np.rad2deg(theta[i])
            r = R.from_euler('ZXY', [yaw, roll, pitch], degrees=True)

            # order of angles follows [yaw, pitch, roll]
            for el in [srf, o]:
                # el.actor.actor.orientation = r.as_euler('ZXY', degrees=True)
                el.actor.actor.orientation = [yaw, roll, pitch]  # r.as_euler('ZXY', degrees=True)
                el.actor.actor.position = np.array([x0, y0, z0])

            t.text = f"{int(roll)}"

            # recompute points on surface given new coordinates
            e.x0, e.y0, e.z0 = x0, y0, z0
            e.pitch, e.roll, e.yaw = pitch, roll, yaw

            e.eval_surf()

            pts.mlab_source.set(x=e.xl, y=e.yl, z=e.zl)
            yield


    update_visualisation(surface, points, o)
    axes = mlab.orientation_axes()
    ax = mlab.axes(xlabel='X', ylabel='Y', zlabel='Z',
                   extent=(0, w, 0, h, 0, 100), opacity=1.0,
                   x_axis_visibility=True, y_axis_visibility=True, z_axis_visibility=True)

    mlab.view(azimuth=90, elevation=90, distance=10000, focalpoint=(-0., 1.05, 1.05))
    mlab.show()
