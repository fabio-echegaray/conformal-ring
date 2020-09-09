import logging

import pandas as pd
from PyQt5 import QtWidgets

from ring.gui.ring import RingWindow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ring').setLevel(logging.INFO)
logging.getLogger('hhlab').setLevel(logging.INFO)
logging.getLogger('gui').setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PyQt5').setLevel(logging.ERROR)

pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)


if __name__ == '__main__':
    import sys
    import os

    from PyQt5.QtCore import QT_VERSION_STR
    from PyQt5.Qt import PYQT_VERSION_STR

    base_path = os.path.abspath('%s' % os.getcwd())
    logging.info('Qt version:' + QT_VERSION_STR)
    logging.info('PyQt version:' + PYQT_VERSION_STR)
    logging.info('Working dir:' + os.getcwd())
    logging.info('Base dir:' + base_path)
    os.chdir(base_path)

    app = QtWidgets.QApplication(sys.argv)

    gui = RingWindow()
    gui.show()

    sys.exit(app.exec_())
