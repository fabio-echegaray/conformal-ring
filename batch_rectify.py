import logging
import os
import warnings
from itertools import cycle

import enlighten

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import skimage.exposure as exposure

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


def image(**kwargs):
    extent = (0, 16, -1, 1)
    ax = plt.gca()
    data = kwargs.pop("data")
    me = kwargs.pop("measurements")
    folder = kwargs.pop("folder")
    fname = kwargs.pop("filename")

    nid, ch, zst = data[['id', 'ch', 'z']].astype(int).values[0]
    poly = data["value"].iloc[0]
    spl = SplineApproximation(poly, fname)
    rect = FunctionRectification(spl)

    me.zstack = zst
    me.rngChannel = ch
    img_rect = exposure.rescale_intensity(rect.rectify(me.rngimage))
    ax.imshow(img_rect, cmap='gray', interpolation="nearest", extent=extent)
    ax.set_axis_off()

    # channel = "dna" if ch == 0 else "tub" if ch == 1 else "act" if ch == 2 else "nan"
    # imageio.imwrite(os.path.join(folder, f"nuc_{nid}_{channel}_rectified.png"), img_rect)


def rectify(path):
    manager = enlighten.get_manager()

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
                nuclei = nuclei.copy().assign(ch=0) \
                    .append(nuclei.copy().assign(ch=1), ignore_index=True) \
                    .append(nuclei.copy().assign(ch=2), ignore_index=True)

                for zst, zdf in nuclei.groupby(["z"]):
                    zst = int(zst)
                    logger.debug(f"Processing zstack={zst}.")
                    folder = ensure_dir(os.path.join(sub_path, f"z_{zst}"))
                    g = sns.FacetGrid(data=zdf,
                                      row='id', col='ch', col_order=[0, 2, 1],
                                      height=0.9, aspect=2,
                                      despine=True, margin_titles=True,
                                      gridspec_kws={"wspace": 0.1}
                                      )
                    g.map_dataframe(image, measurements=me, folder=folder, filename=fname)
                    g.fig.suptitle('\r\n    '.join(p.parts[-2:]))
                    g.savefig(os.path.join(folder, f"rectification_summary.pdf"))
            bar.update()
    manager.stop()


if __name__ == '__main__':
    p = os.path.join('/Volumes/Kidbeat/data/lab/airy-ring')
    rectify(p)
