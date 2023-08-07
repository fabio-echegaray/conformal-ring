import json
import re
from pathlib import Path

import numpy as np
import tifffile as tf

from fileops.image._base import ImageFileBase


class MetadataVersion10Mixin(ImageFileBase):

    def __init__(self, **kwargs):
        self.image_path = Path(self.image_path)
        base_name = self.image_path.name.split(".ome")[0]

        self._meta_name = f"{base_name}_metadata.txt"
        self.metadata_path = self.image_path.parent / self._meta_name
        self._load_metadata()

        super().__init__(**kwargs)

    def _load_metadata(self):
        with open(self.metadata_path) as f:
            self.md = json.load(f)

        with tf.TiffFile(self.image_path) as tif:
            imagej_metadata = tif.imagej_metadata
            micromanager_metadata = tif.micromanager_metadata
            keyframe = tif.pages.keyframe
        summary = self.md['Summary']
        self.all_positions = summary['StagePositions']

        self.channel_names = summary["ChNames"]
        self.channels = set(range(summary["Channels"]))

        mmf_size_x = int(summary["Width"])
        mmf_size_y = int(summary["Height"])
        mmf_size_z = int(summary["Slices"])
        mmf_size_t = int(summary["Frames"])
        mmf_size_c = int(summary["Channels"])
        mmf_physical_size_z = float(summary["z-step_um"])

        mm_sum = micromanager_metadata["Summary"]
        mm_size_x = int(mm_sum["Width"])
        mm_size_y = int(mm_sum["Height"])
        mm_size_z = int(mm_sum["Slices"])
        mm_size_t = int(mm_sum["Frames"])
        mm_size_c = int(mm_sum["Channels"])
        mm_physical_size_z = float(mm_sum["z-step_um"])

        kf_size_x = int(keyframe.shape[keyframe.axes.find('X')])
        kf_size_y = int(keyframe.shape[keyframe.axes.find('Y')])

        # calculate pixel size assuming square pixels
        if 'XResolution' in keyframe.tags:
            xr = keyframe.tags['XResolution'].value
            res = float(xr[0]) / float(xr[1])  # pixels per um
            if keyframe.tags['ResolutionUnit'].value == tf.TIFF.RESUNIT.CENTIMETER:
                res = res / 1e4
        else:
            res = 1

        # magnification = None
        # size_x_unit = size_y_unit = size_z_unit = "um"

        self.pix_per_um = 1. / res
        self.um_per_pix = res
        self.um_per_z = max(mmf_physical_size_z, mm_physical_size_z)
        self.width = max(mmf_size_x, mm_size_x, kf_size_x, keyframe.imagewidth)
        self.height = max(mmf_size_y, mm_size_y, kf_size_y, keyframe.imagelength)
        self.n_zstacks = max(mmf_size_z, mm_size_z, len(self.zstacks))
        self.n_frames = max(mmf_size_t, mm_size_t)
        self.n_channels = max(mmf_size_c, mm_size_c, len(self.channels))

        # build a list of the images stored in sequence
        for counter, fkey in enumerate(list(self.md.keys())[1:]):
            if fkey[0:8] == "FrameKey":
                t, c, z = re.search(r'^FrameKey-([0-9]*)-([0-9]*)-([0-9]*)$', fkey).groups()
                t, c, z = int(t), int(c), int(z)

                fname = self.md[fkey]["FileName"] if "FileName" in self.md[fkey] else ""
                fname = fname.split("/")[1] if "/" in fname else fname
                self.files.append(fname)
                self.timestamps.append(self.md[fkey]["ElapsedTime-ms"] / 1000)
                self.zstacks.append(z)
                self.zstacks_um.append(self.md[fkey]["ZPositionUm"])
                self.frames.append(int(t))
                self.all_planes.append(fkey)
                # build dictionary where the keys are combinations of c z t and values are the index
                self.all_planes_md_dict[f"c{int(c):0{len(str(self.n_channels))}d}"
                                        f"z{int(z):0{len(str(self.n_zstacks))}d}"
                                        f"t{int(t):0{len(str(self.n_frames))}d}"] = counter

        self.timestamps = sorted(np.unique(self.timestamps))
        self.frames = sorted(np.unique(self.frames))
        self.zstacks = sorted(np.unique(self.zstacks))
        self.zstacks_um = sorted(np.unique(self.zstacks_um))