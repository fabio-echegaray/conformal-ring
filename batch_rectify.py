import logging
import os
import warnings
import enlighten

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import skimage.exposure as exposure
import imageio

from filters import nucleus
from gui._ring_label import RingImageQLabel
from gui.measure import Measure
from rectification import SplineApproximation, FunctionRectification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('batch')

# reduce console output while using batch tool
warnings.simplefilter(action='ignore', category=FutureWarning)

base_dir = "/Volumes/Kidbeat/data/"
lab_dir = os.path.join(base_dir, "lab")
compiled_data_dir = os.path.join(lab_dir, "compiled")
out_dir = os.path.join(lab_dir, "scripts_output")
size_square = (3, 3)
size_A4 = (8.268, 11.693)


def ensure_dir(file_path):
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        # print(f"Creating dir {file_path}")
        os.makedirs(file_path, exist_ok=True)
    return file_path


def rectify(path):
    manager = enlighten.get_manager()
    # extent = (0, 2 * np.pi, -1, 1)
    extent = (0, 16, -1, 1)

    for root, directories, filenames in os.walk(os.path.join(path)):
        bar = manager.counter(total=len(filenames), desc='Progress', unit='files')
        for k, filename in enumerate(filenames):
            ext = filename.split('.')[-1]
            if ext == 'czi':
                logger.info(f"Processing {filename}")
                fname = os.path.join(root, filename)
                me = Measure()
                me.file = fname
                me.dnaChannel = 0
                me.measure_all_nuclei()

                p = Path(fname)
                sub_path = ensure_dir(os.path.join(out_dir, *p.parts[-3:], "nucleus"))
                me.measurements.to_csv(os.path.join(sub_path, "measurements.csv"))

                nuclei = me.measurements \
                    .query("type ==  'nucleus'") \
                    .pipe(nucleus,
                          nucleus_col='value',
                          radius_min=5 * me.pix_per_um,
                          radius_max=13.5 * me.pix_per_um)

                fig = plt.figure(1, size_A4, dpi=300)
                for zst, zdf in nuclei.groupby(["z"]):
                    zst = int(zst)
                    folder = ensure_dir(os.path.join(sub_path, f"z_{zst}"))
                    fig.clf()
                    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                                     nrows_ncols=(len(nuclei), 3),
                                     share_all=True,
                                     aspect=False,
                                     axes_pad=0.05,  # pad between axes in inch.
                                     )
                    for g in grid:
                        g.set_aspect(0.01, adjustable='box')
                        g.set_axis_off()

                    it_grid = iter(grid)
                    for nid, df in nuclei.groupby(["id"]):
                        nid = int(nid)
                        logger.debug(f"Processing nucleus={nid} in zstack={zst}.")
                        poly = me.nucleus(nid)["value"].iloc[0]
                        spl = SplineApproximation(poly, fname)
                        rect = FunctionRectification(spl)

                        me.zstack = zst
                        dna_rect = exposure.rescale_intensity(rect.rectify(me.dnaimage))
                        imageio.imwrite(os.path.join(folder, f"nuc_{nid}_dna_rectified.png"), dna_rect)

                        me.rngChannel = 1
                        tub_rect = exposure.rescale_intensity(rect.rectify(me.rngimage))
                        imageio.imwrite(os.path.join(folder, f"nuc_{nid}_tub_rectified.png"), tub_rect)

                        me.rngChannel = 2
                        act_rect = exposure.rescale_intensity(rect.rectify(me.rngimage))
                        imageio.imwrite(os.path.join(folder, f"nuc_{nid}_act_rectified.png"), act_rect)

                        next(it_grid).imshow(dna_rect, cmap='gray', interpolation="nearest", extent=extent)
                        next(it_grid).imshow(act_rect, cmap='gray', interpolation="nearest", extent=extent)
                        next(it_grid).imshow(tub_rect, cmap='gray', interpolation="nearest", extent=extent)
                    fig.savefig(os.path.join(folder, f"rectification_summary.pdf"))
            bar.update()
    manager.stop()


if __name__ == '__main__':
    p = os.path.join('/Volumes/Kidbeat/data/lab/airy-ring')
    rectify(p)
