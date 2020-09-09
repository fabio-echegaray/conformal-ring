import logging
import os
import warnings
import enlighten

import pandas as pd
from pandas.errors import EmptyDataError

logger = logging.getLogger('batch')
logger.setLevel(logging.DEBUG)

# reduce console output while using batch tool
warnings.simplefilter(action='ignore', category=FutureWarning)


def collect(path):
    df = pd.DataFrame()
    manager = enlighten.get_manager()

    for root, directories, filenames in os.walk(os.path.join(path)):
        bar = manager.counter(total=len(filenames), desc='Progress', unit='files')
        for k, filename in enumerate(filenames):
            ext = filename.split('.')[-1]
            if ext == 'csv':
                logger.info("Adding %s." % filename)
                try:
                    fname = os.path.join(root, filename)
                    csv = pd.read_csv(fname)
                    csv['folder'] = os.path.basename(os.path.dirname(fname))
                    df = df.append(csv, sort=False, ignore_index=True)
                except EmptyDataError:
                    logger.warning('Found empty csv file: %s.' % filename)
                    # traceback.print_stack()
            bar.update()
    manager.stop()
    return df


if __name__ == '__main__':
    path = os.path.join('/Volumes/Kidbeat/data/lab/airy-ring')
    csv_file = os.path.join(path, 'ringlines.csv')
    if os.path.isfile(csv_file):
        logger.warning("Csv file already exists in the folder. Please delete the file in order to proceed.")
        exit(1)

    df = collect(path)
    df['condition'] = df['folder'].replace({'20200128 - actrng - mcak 2 ctrl 1':  'arrest',
                                            '20200128 - actrng - mcak 2 ctrl 2':  'arrest',
                                            '20200129 - actrng - dhc sirna 1':    'dhc',
                                            '20200129 - actrng - dhc sirna 2':    'dhc',
                                            '20200129 - actrng - mcak 2 sirna 1': 'mcak',
                                            '20200129 - actrng - mcak 2 sirna 2': 'mcak'
                                            })

    df.to_csv(csv_file, index=False)
