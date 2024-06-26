import os
import xml.etree.ElementTree
from datetime import datetime
from pathlib import Path
from typing import Tuple
import statistics
from aicsimageio.readers import BioformatsReader
from aicsimageio.readers.bioformats_reader import BioFile
from bs4 import BeautifulSoup as bs

import numpy as np
import pandas as pd
from ome_types import OME

from fileops.image.image_file import ImageFile
from fileops.image.imagemeta import MetadataImage
from fileops.logger import get_logger


class OMEImageFile(ImageFile):
    ome_ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    log = get_logger(name='OMEImageFile')

    def __init__(self, image_path: Path, **kwargs):
        super(OMEImageFile, self).__init__(image_path, **kwargs)

        self._rdr: BioformatsReader = None

        self.md, self.md_xml = self._get_metadata()
        self.all_series = self._rdr.scenes
        self.instrument_md = self.md.instruments
        self.objectives_md = None
        self.md_description = bs(self.md_xml, "lxml-xml")

        self._load_imageseries()

        self._fix_defaults(failover_dt=self._failover_dt, failover_mag=self._failover_mag)

    @staticmethod
    def has_valid_format(path: Path):
        return True

    @property
    def info(self) -> pd.DataFrame:
        fname_stat = Path(self.image_path).stat()
        fcreated = datetime.fromtimestamp(fname_stat.st_ctime).strftime("%a %b/%d/%Y, %H:%M:%S")
        fmodified = datetime.fromtimestamp(fname_stat.st_mtime).strftime("%a %b/%d/%Y, %H:%M:%S")
        series_info = list()
        for imageseries in self.md.findall('ome:Image', self.ome_ns):  # iterate through all series
            instrument = imageseries.find('ome:InstrumentRef', self.ome_ns)
            obj_id = imageseries.find('ome:ObjectiveSettings', self.ome_ns).get('ID')
            objective = self.md.find(f'ome:Instrument/ome:Objective[@ID="{obj_id}"]', self.ome_ns)
            imgseries_pixels = imageseries.findall('ome:Pixels', self.ome_ns)
            for isr_pixels in imgseries_pixels:
                size_x = float(isr_pixels.get('PhysicalSizeX'))
                size_y = float(isr_pixels.get('PhysicalSizeY'))
                size_z = float(isr_pixels.get('PhysicalSizeZ'))
                size_x_unit = isr_pixels.get('PhysicalSizeXUnit')
                size_y_unit = isr_pixels.get('PhysicalSizeYUnit')
                size_z_unit = isr_pixels.get('PhysicalSizeZUnit')
                timestamps = sorted(
                    np.unique([p.get('DeltaT') for p in isr_pixels.findall('ome:Plane', self.ome_ns) if
                               p.get('DeltaT') is not None]).astype(np.float64))
                series_info.append({
                    'filename':                          os.path.basename(self.image_path),
                    'image_id':                          imageseries.get('ID'),
                    'image_name':                        imageseries.get('Name'),
                    'instrument_id':                     instrument.get('ID'),
                    'pixels_id':                         isr_pixels.get('ID'),
                    'channels':                          int(isr_pixels.get('SizeC')),
                    'z-stacks':                          int(isr_pixels.get('SizeZ')),
                    'frames':                            int(isr_pixels.get('SizeT')),
                    'delta_t':                           float(np.nanmean(np.diff(timestamps))),
                    # 'timestamps': timestamps,
                    'width':                             self.width,
                    'height':                            self.height,
                    'data_type':                         isr_pixels.get('Type'),
                    'objective_id':                      obj_id,
                    'magnification':                     int(float(objective.get('NominalMagnification'))),
                    'pixel_size':                        (size_x, size_y, size_z),
                    'pixel_size_unit':                   (size_x_unit, size_y_unit, size_z_unit),
                    'pix_per_um':                        (1 / size_x, 1 / size_y, 1 / size_z),
                    'change (Unix), creation (Windows)': fcreated,
                    'most recent modification':          fmodified,
                })
        out = pd.DataFrame(series_info)
        return out

    @property
    def series(self):
        return self.all_series[self._series]

    @series.setter
    def series(self, s):
        if type(s) is int:
            self._series = s
        elif type(s) is str:
            for k, imser in enumerate(self.all_series):
                if imser.attrib['Name'] == s:
                    self._series = k
                    break
        elif type(s) is xml.etree.ElementTree.Element:
            for k, imser in enumerate(self.all_series):
                if imser.attrib == s.attrib:
                    self._series = k
                    break
        else:
            raise ValueError("Unexpected type of variable to load series.")

        super().__init__(s)

    def _load_imageseries(self):
        if not self.all_series:
            return
        self.images_md = self.all_series[self._series]
        self.planes_md = self.md_description.find('Pixels')
        self.all_planes = self.md_description.find_all('Plane')

        self.channels = set(int(p.get('TheC')) for p in self.all_planes)
        self.zstacks = sorted(np.unique([p.get('TheZ') for p in self.all_planes]).astype(int))
        self.z_position = np.array([p.get('PositionZ') for p in self.all_planes]).astype(float)
        self.frames = sorted(np.unique([p.get('TheT') for p in self.all_planes]).astype(int))
        self.n_channels = len(self.channels)
        self.n_zstacks = len(self.zstacks)
        self.n_frames = len(self.frames)
        self._md_n_zstacks = self.n_zstacks
        self._md_n_frames = self.n_frames
        self._md_n_channels = self.n_channels
        if self.planes_md.get('PhysicalSizeX') and \
                self.planes_md.get('PhysicalSizeX') == self.planes_md.get('PhysicalSizeY'):
            self.um_per_pix = float(self.planes_md.get('PhysicalSizeX'))
        else:
            self.um_per_pix = 1
        self.pix_per_um = 1. / self.um_per_pix
        self.width = int(self.planes_md.get('SizeX'))
        self.height = int(self.planes_md.get('SizeY'))
        if self.md_description:
            px_md = self.md_description.find('Pixels')
            self.um_per_z = float(px_md.get('PhysicalSizeZ'))
        elif self.planes_md.get('PhysicalSizeZ'):
            self.um_per_z = float(self.planes_md.get('PhysicalSizeZ'))
        elif len(self.z_position) > 0:
            z_diff = np.diff(self.z_position)
            self.um_per_z = statistics.mode(z_diff[z_diff > 0])

        # obj = self.images_md.find('ObjectiveSettings', self.ome_ns)
        # obj_id = obj.get('ID') if obj else None
        # objective = self.md.find(f'Instrument/Objective[@ID="{obj_id}"]', self.ome_ns) if obj else None
        # self.magnification = int(float(objective.get('NominalMagnification'))) if objective else None

        self.timestamps = sorted(
            np.array([p.get('DeltaT') for p in self.all_planes if p.get('DeltaT') is not None]).astype(np.float64))
        ts_diff = np.diff(self.timestamps)
        self.time_interval = statistics.mode(ts_diff)
        # # values higher than 2s likely to be waiting times
        # self.time_interval = statistics.mode(ts_diff[(0<ts_diff) & (ts_diff<2000)])
        # # plot ticks
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(figsize=(8, 4))
        # ax.plot(self.timestamps, [0.01] * len(self.timestamps), '|', color='k')

        # build dictionary where the keys are combinations of c z t and values are the index
        self.all_planes_md_dict = {f"c{int(plane.get('TheC')):0{len(str(self._md_n_channels))}d}"
                                   f"z{int(plane.get('TheZ')):0{len(str(self._md_n_zstacks))}d}"
                                   f"t{int(plane.get('TheT')):0{len(str(self._md_n_frames))}d}": plane
                                   for i, plane in enumerate(self.all_planes)}

        self.log.info(f"Image series {self._series} loaded. "
                      f"Image size (WxH)=({self.width:d}x{self.height:d}); "
                      f"calibration is {self.pix_per_um:0.3f} pix/um and {self.um_per_z:0.3f} um/z-step; "
                      f"movie has {len(self.frames)} frames, {self.n_channels} channels, {self.n_zstacks} z-stacks and "
                      f"{len(self.all_planes)} image planes in total.")

    def _lazy_load_jvm(self):
        # if not self._jvm:
        #     self._jvm = create_jvm()
        if not self._rdr:
            self._rdr = BioformatsReader(self.image_path.as_posix())

    def _image(self, plane_ix, row=0, col=0, fid=0) -> MetadataImage:  # PLANE HAS METADATA INFO OF THE IMAGE PLANE
        plane = self.all_planes_md_dict[plane_ix]
        c, z, t = int(plane.get('TheC')), int(plane.get('TheZ')), int(plane.get('TheT'))
        # logger.debug('retrieving image id=%d row=%d col=%d fid=%d' % (_id, row, col, fid))
        self._lazy_load_jvm()

        # image = self._rdr.read(c=c, z=z, t=t, series=self._series, rescale=False)
        # returns 5D TCZYX xarray data array backed by dask array
        image = self._rdr.get_image_data("TCZYX", c=c, z=z, t=t)

        w = int(self.planes_md.get('SizeX'))
        h = int(self.planes_md.get('SizeY'))

        return MetadataImage(reader='OME',
                             image=image,
                             pix_per_um=1. / self.um_per_pix, um_per_pix=self.um_per_pix,
                             time_interval=None,
                             timestamp=float(plane.get('DeltaT')) if plane.get('DeltaT') is not None else 0.0,
                             frame=int(t), channel=int(c), z=int(z), width=w, height=h,
                             intensity_range=[np.min(image), np.max(image)])

    def _get_metadata(self) -> Tuple[OME, str]:
        self._lazy_load_jvm()

        biofile_kwargs = {'options': {}, 'original_meta': False, 'memoize': 0, 'dask_tiles': False, 'tile_size': None}
        with BioFile(self.image_path.as_posix(), **biofile_kwargs) as rdr:
            md_xml = rdr.ome_xml
        md = self._rdr.ome_metadata

        return md, md_xml
