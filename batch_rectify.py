import logging
import os
import warnings
import enlighten

import shapely.wkt
from pathlib import Path
import imageio

from gui.measure import Measure
from rectification import SplineApproximation, FunctionRectification

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('batch')
logger.setLevel(logging.DEBUG)

# reduce console output while using batch tool
warnings.simplefilter(action='ignore', category=FutureWarning)

base_dir = "/Volumes/Kidbeat/data/"
lab_dir = os.path.join(base_dir, "lab")
compiled_data_dir = os.path.join(lab_dir, "compiled")
out_dir = os.path.join(lab_dir, "scripts_output")


def ensure_dir(file_path):
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        # print(f"Creating dir {file_path}")
        os.makedirs(file_path, exist_ok=True)
    return file_path


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

                for ix, df in me.measurements.query("type ==  'nucleus'").groupby(["z", "id"]):
                    zst, nid = [int(i) for i in ix]
                    folder = ensure_dir(os.path.join(sub_path, f"z_{zst}"))
                    logger.debug(f"Processing nucleus={nid} in zstack={zst}.")
                    poly = shapely.wkt.loads(me.nucleus(nid)["value"].iloc[0])
                    spl = SplineApproximation(poly, fname)
                    rect = FunctionRectification(spl)

                    me.zstack = zst
                    dna_rect = rect.rectify(me.dnaimage)
                    imageio.imwrite(os.path.join(folder, f"nuc_{nid}_dna_rectified.png"), dna_rect)

                    me.rngChannel = 1
                    tub_rect = rect.rectify(me.rngimage)
                    imageio.imwrite(os.path.join(folder, f"nuc_{nid}_tub_rectified.png"), tub_rect)

                    me.rngChannel = 2
                    act_rect = rect.rectify(me.rngimage)
                    imageio.imwrite(os.path.join(folder, f"nuc_{nid}_act_rectified.png"), act_rect)

            bar.update()
    manager.stop()


if __name__ == '__main__':
    p = os.path.join('/Volumes/Kidbeat/data/lab/airy-ring')
    rectify(p)
