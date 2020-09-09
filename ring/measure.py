import itertools
import logging

import numpy as np
import pandas as pd
import seaborn as sns
from shapely.geometry.point import Point

from ring.gui import find_image, qpixmap_from, retrieve_image
from ring import measurements as m, plots as p


class FileImageMixin(object):
    log = logging.getLogger('FileImageMixin')

    def __init__(self):
        self._file = None
        self._meta = None
        self._zstack = 0

        self.images = None
        self.pix_per_um = None
        self.um_per_pix = None
        self.nFrames = 0
        self.nChannels = 0
        self.nZstack = 0

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, value: str):
        if value is not None:
            self.log.info('Loading %s' % value)
            self._file = value
            im = find_image(value)
            self._meta = im
            self.images = im.image
            self.pix_per_um = im.pix_per_um
            self.um_per_pix = im.um_per_pix
            self.nFrames = im.frames
            self.nChannels = im.channels
            self.nZstack = int(len(self.images) / self.nFrames / self.nChannels)
            self.log.info(f"Pixels per um: {self.pix_per_um:0.4f}, {im.image.dtype}")

    @property
    def zstack(self):
        return self._zstack

    @zstack.setter
    def zstack(self, value):
        self._zstack = value

    def fileimage_from(self, fi):
        assert issubclass(type(fi), FileImageMixin), "Not the right type to load values from!"
        self._file = fi.file
        self._zstack = fi.zstack
        self.images = fi.images
        self.pix_per_um = fi.pix_per_um
        self.um_per_pix = fi.um_per_pix
        self.nFrames = fi.nFrames
        self.nChannels = fi.nChannels
        self.nZstack = fi.nZstack


# noinspection PyPep8Naming
class Measure(FileImageMixin):
    log = logging.getLogger('Measure')
    _nlin = 20
    _colors = sns.husl_palette(_nlin, h=.5).as_hex()
    dl = 0.05

    def __init__(self):
        super(Measure, self).__init__()
        self._dnaChannel = 0
        self._rngChannel = 0

        self._dnaimage = None
        self._rngimage = None
        self._dnapixmap = None
        self._rngpixmap = None
        self.measurements = pd.DataFrame()

    @FileImageMixin.file.setter
    def file(self, value):
        if value is not None:
            super(Measure, type(self)).file.fset(self, value)
            self.measurements = pd.DataFrame()

    @FileImageMixin.zstack.setter
    def zstack(self, value):
        if value is not None:
            super(Measure, type(self)).zstack.fset(self, value)
            self._dnaimage = None
            self._rngimage = None
            self._dnapixmap = None
            self._rngpixmap = None

    @property
    def dnaChannel(self):
        return self._dnaChannel

    @dnaChannel.setter
    def dnaChannel(self, value):
        self._dnaChannel = value
        self._dnaimage = None
        self._dnapixmap = None

    @property
    def rngChannel(self):
        return self._rngChannel

    @rngChannel.setter
    def rngChannel(self, value):
        self._rngChannel = value
        self._rngimage = None
        self._rngpixmap = None

    @property
    def dnaimage(self):
        if self._dnaimage is None:
            if self.file is not None:
                self._dnaimage = retrieve_image(self.images, channel=self.dnaChannel, number_of_channels=self.nChannels,
                                                zstack=self.zstack, number_of_zstacks=self.nZstack, frame=0)
                self._dnapixmap = None
        return self._dnaimage

    @property
    def rngimage(self):
        if self._rngimage is None:
            if self.file is not None:
                self._rngimage = retrieve_image(self.images, channel=self.rngChannel, number_of_channels=self.nChannels,
                                                zstack=self.zstack, number_of_zstacks=self.nZstack, frame=0)
                self._rngpixmap = None
        return self._rngimage

    @property
    def dnapixmap(self):
        if self._dnapixmap is None:
            self._dnapixmap = qpixmap_from(self.dnaimage)
        return self._dnapixmap

    @property
    def rngpixmap(self):
        if self._rngpixmap is None:
            self._rngpixmap = qpixmap_from(self.rngimage)
        return self._rngpixmap

    @property
    def dwidth(self):
        return self._rngimage.shape[0] if self._rngimage is not None else 0

    @property
    def dheight(self):
        return self._rngimage.shape[1] if self._rngimage is not None else 0

    def measure_all_nuclei(self):
        __z_stack = self.zstack
        for zst in range(self.nZstack):
            self.zstack = zst
            self._measure_nuclei()

        self.zstack = __z_stack

    def _measure_nuclei(self):
        if self.dnaimage is None:
            return

        lbl, boundaries = m.nuclei_segmentation(self.dnaimage, simp_px=self.pix_per_um / 2)
        boundaries = m.exclude_contained(boundaries)
        self.log.debug(f"Processing z-stack {self.zstack} with {len(boundaries)} nuclei.")

        for nucleus in boundaries:
            nucbnd = nucleus["boundary"]
            _x, _y = np.array(nucbnd.centroid.coords).astype(np.int16)[0]
            # self.log.debug(f"({_x},{_y})")

            self.measurements = self.measurements.append(
                {
                    'x':     _x,
                    'y':     _y,
                    'z':     self.zstack,
                    'type':  'nucleus',
                    'value': nucbnd,
                    'id':    nucleus['id']
                    },
                ignore_index=True)

    def _measure_lines_around_nuclei(self, _id):
        if self.dnaimage is None:
            return

        nucleus = self.measurements[(self.measurements['type'] == 'nucleus') &
                                    (self.measurements['id'] == _id) &
                                    (self.measurements['z'] == self.zstack)
                                    ]
        nucbnd = nucleus["value"].iloc[0]
        _x, _y = np.array(nucbnd.centroid.coords).astype(np.int16)[0]
        self.log.debug(f"({_x},{_y})")

        lines = m.measure_lines_around_polygon(self.rngimage, nucbnd, rng_thick=4, dl=self.dl,
                                               n_lines=self._nlin, pix_per_um=self.pix_per_um)
        for k, ((ls, l), colr) in enumerate(zip(lines, itertools.cycle(self._colors))):
            if ls is not None:
                self.measurements = self.measurements.append(
                    {
                        'x':     _x,
                        'y':     _y,
                        'z':     self.zstack,
                        'type':  'line',
                        'value': l,
                        'id':    nucleus['id'].iloc[0],
                        'li':    k,
                        'c':     colr,
                        'ls0':   ls.coords[0],
                        'ls1':   ls.coords[1],
                        'd':     max(l) - min(l),
                        'sum':   np.sum(l)
                        },
                    ignore_index=True)

    def nucleus(self, *args):
        if len(args) == 1 and np.isscalar(args[0]):
            _id = args[0]
            return self._nucleus(int(_id))

        elif len(args) == 2 and np.all([np.issubdtype(type(a), np.integer) for a in args]):
            x, y = args[0], args[1]

            self.log.info(f"Searching nucleus at ({x},{y})")
            pt = Point(x, y)
            # search for nuclear boundary of interest
            if self.measurements.empty or len(self.measurements[(self.measurements['z'] == self.zstack)]) == 0:
                self._measure_nuclei()
            nuclei = self.measurements[
                (self.measurements['type'] == 'nucleus') & (self.measurements['z'] == self.zstack)]
            nuclei = nuclei[nuclei.apply(lambda row: row['value'].contains(pt), axis=1)]
            self.log.debug(f"{len(nuclei)} nuclei found.")
            assert len(nuclei) <= 1, "Found more than one result for query."

            if nuclei.empty:
                return pd.DataFrame()

            return self._nucleus(nuclei['id'].iloc[0])
        else:
            raise AttributeError("Parameter no recognized.")

    def _nucleus(self, _id):
        return get_from_df(self.measurements, 'nucleus', _id, self.zstack)

    @property
    def nuclei(self):
        if self.measurements.empty:
            self._measure_nuclei()

        return self.measurements[
            (self.measurements['type'] == 'nucleus') &
            (self.measurements['z'] == self.zstack)
            ]

    def lines(self, _id=None):
        if _id is not None:
            if self.measurements.empty:
                self._measure_lines_around_nuclei(_id)
            # check if lines were calculated previously
            lines = self.measurements[
                (self.measurements['id'] == _id) &
                (self.measurements['z'] == self.zstack)
                ]
            if len(lines) == 1:
                self._measure_lines_around_nuclei(_id)
            return get_from_df(self.measurements, 'line', _id, self.zstack)

        else:
            return get_from_df(self.measurements, 'line', _id, self.zstack)

    def drawMeasurements(self, ax):
        from shapely import affinity

        img = self.rngimage
        ext = (0, self.dwidth / self.pix_per_um, 0, self.dheight / self.pix_per_um)
        ax.imshow(img, extent=ext, origin='lower', interpolation='none', cmap='gray')

        for _, n in self.nuclei.iterrows():
            n_um = affinity.scale(n["value"], xfact=self.um_per_pix, yfact=self.um_per_pix, origin=(0, 0, 0))
            p.render_polygon(n_um, zorder=10, ax=ax)
            ax.text(n_um.centroid.x, n_um.centroid.y, int(n["id"]), color='w', fontdict={'size': 7},
                    ha='center', va='center_baseline')

        x0, y0 = 5, 5
        ax.plot([x0, x0 + 10], [y0, y0], c='w', lw=2)
        ax.text(x0, y0 + 5.5, '10 um', color='w', fontdict={'size': 7})


def get_from_df(df, _type, _id, z):
    if df.empty:
        return pd.DataFrame()
    if _id is None:
        _id = df["id"].unique()

    return df[(df['type'] == _type) & (df['id'].isin(np.array(_id).ravel())) & (df['z'] == z)]
