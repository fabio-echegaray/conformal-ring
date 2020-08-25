import os
import sys
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.ticker import EngFormatter
from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (QFileDialog, QMainWindow, QStatusBar, QWidget)

from gui._ring_label import RingImageQLabel
from gui._widget_graph import GraphWidget
from gui.stack_ring import StkRingWidget
from rectification import TestSplineApproximation, TestPiecewiseLinearRectification, TestFunctionRectification
import measurements as m


# noinspection PyPep8Naming
class RingWindow(QMainWindow):
    log = logging.getLogger('gui.RingWindow')
    image: RingImageQLabel
    statusbar: QStatusBar

    def __init__(self):
        super(RingWindow, self).__init__()
        path = os.path.join(sys.path[0], __package__)

        uic.loadUi(os.path.join(path, 'gui_ring.ui'), self)

        self.ctrl = QWidget()
        uic.loadUi(os.path.join(path, 'gui_ring_controls.ui'), self.ctrl)

        self.ctrl.zSpin.valueChanged.connect(self.onZValueChange)
        self.ctrl.openButton.pressed.connect(self.onOpenButton)
        self.ctrl.addButton.pressed.connect(self.onAddButton)
        self.ctrl.plotButton.pressed.connect(self.onPlotButton)
        self.ctrl.measureButton.pressed.connect(self.onMeasureButton)
        self.ctrl.dnaSpin.valueChanged.connect(self.onDnaValChange)
        self.ctrl.actSpin.valueChanged.connect(self.onActValChange)
        self.ctrl.dnaChk.toggled.connect(self.onImgToggle)
        self.ctrl.actChk.toggled.connect(self.onImgToggle)
        self.ctrl.renderChk.stateChanged.connect(self.onRenderChk)

        self.image.clicked.connect(self.onImgUpdate)
        self.image.lineUpdated.connect(self.onImgUpdate)
        self.image.linePicked.connect(self.onLinePickedFromImage)
        self.image.nucleusPicked.connect(self.onNucleusPickedFromImage)
        self.image.dnaChannel = self.ctrl.dnaSpin.value()
        self.image.rngChannel = self.ctrl.actSpin.value()

        self.grph = GraphWidget()
        self.grphtimer = QTimer()
        self.grphtimer.setSingleShot(True)

        self.stk = StkRingWidget(self.image, linePicked=self.onLinePickedFromStackGraph)

        self.grph.linePicked.connect(self.onLinePickedFromGraph)
        # self.stk.linePicked.connect(self.onLinePickedFromStackGraph)
        self.grphtimer.timeout.connect(self._graph)

        self.image.dnaChannel = self.ctrl.dnaSpin.value()
        self.image.rngChannel = self.ctrl.actSpin.value()

        self.measure_n = 0
        self.selectedLine = None
        self.line_length = 4

        self.currMeasurement = None
        self.currN = None
        self.currZ = None

        self.df = pd.DataFrame()
        self.file = "/Users/Fabio/data/lab/airyscan/nil.czi"

        self.show()
        self.grph.show()
        self.stk.show()
        self.ctrl.show()
        self.move(0, 0)
        self.resizeEvent(None)
        self.moveEvent(None)

    def resizeEvent(self, event):
        # this is a hack to resize everything when the user resizes the main window
        self.grph.setFixedWidth(self.width())
        self.image.setFixedWidth(self.width())
        self.image.setFixedHeight(self.height())
        self.image.resizeEvent(None)
        self.moveEvent(None)

    def moveEvent(self, QMoveEvent):
        self.ctrl.move(self.frameGeometry().topRight())
        self.grph.move(self.geometry().bottomLeft())
        self.stk.move(self.ctrl.frameGeometry().topRight())

    def closeEvent(self, event):
        self._saveCurrentFileMeasurements()
        # if not self.df.empty:
        #     self.df.loc[:, "condition"] = self.ctrl.experimentLineEdit.text()
        #     self.df.loc[:, "l"] = self.df.loc[:, "l"].apply(lambda v: np.array2string(v, separator=','))
        #     self.df.to_csv(os.path.join(os.path.dirname(self.image.file), "ringlines.csv"))
        self.grph.close()
        self.ctrl.close()
        self.stk.close()

    def focusInEvent(self, QFocusEvent):
        self.log.debug('focusInEvent')
        self.ctrl.activateWindow()
        self.grph.activateWindow()
        self.stk.focusInEvent(None)

    def showEvent(self, event):
        self.setFocus()

    def _saveCurrentFileMeasurements(self):
        if not self.df.empty:
            fname = os.path.basename(self.image.file)
            df = self.df[self.df["file"] == fname]
            df.loc[:, "condition"] = self.ctrl.experimentLineEdit.text()
            df.loc[:, "l"] = self.df.loc[:, "l"].apply(lambda v: np.array2string(v, separator=','))
            df.to_csv(os.path.join(os.path.dirname(self.image.file), f"{fname}.csv"))

    def _graphTendency(self):
        df = pd.DataFrame(self.image.measurements).drop(['x', 'y', 'c', 'ls0', 'ls1', 'd', 'sum'], axis=1)
        df.loc[:, "xx"] = df.loc[:, "l"].apply(
            lambda v: np.arange(start=0, stop=len(v) * self.image.dl, step=self.image.dl))
        df = m.vector_column_to_long_fmt(df, val_col="l", ix_col="xx")
        sns.lineplot(x="xx", y="l", data=df, ax=self.grph.ax, color='k', ci="sd", zorder=20)
        self.grph.ax.set_ylabel('')
        self.grph.ax.set_xlabel('')
        self.grph.canvas.draw()

    def _graph(self, alpha=1.0):
        self.grph.clear()
        lines = self.image.lines(self.image.currNucleusId)
        if lines.empty:
            return

        for ix, me in lines.iterrows():
            x = np.arange(start=0, stop=len(me['value']) * self.image.dl, step=self.image.dl)
            lw = 0.1 if me['li'] != self.image.selectedLine else 0.5
            self.grph.ax.plot(x, me['value'], linewidth=lw, linestyle='-', color=me['c'], alpha=alpha, zorder=10,
                              picker=5, label=int(me['li']))  # , marker='o', markersize=1)
        self.grph.format_ax()
        # self.statusbar.showMessage("ptp: %s" % ["%d " % me['d'] for me in self.image.lines().iterrows()])
        self.grph.canvas.draw()

    @QtCore.pyqtSlot()
    def onImgToggle(self):
        self.log.debug('onImgToggle')
        if self.ctrl.dnaChk.isChecked():
            self.image.activeCh = "dna"
        if self.ctrl.actChk.isChecked():
            self.image.activeCh = "act"

    @QtCore.pyqtSlot()
    def onRenderChk(self):
        self.log.debug('onRenderChk')
        self.image.renderMeasurements = self.ctrl.renderChk.isChecked()
        self.stk.renderMeasurements = self.ctrl.renderChk.isChecked()

    @QtCore.pyqtSlot()
    def onOpenButton(self):
        self.log.debug('onOpenButton')

        # save current file measurements as a backup
        self._saveCurrentFileMeasurements()

        qfd = QFileDialog()
        path = os.path.dirname(self.file)
        if self.image.file is not None:
            self.statusbar.showMessage("current file: %s" % os.path.basename(self.image.file))
        flt = "zeiss(*.czi)"
        f = QFileDialog.getOpenFileName(qfd, "Open File", path, flt)
        if len(f) > 0:
            self._open(f[0])

    def _open(self, fname):
        assert type(fname) is str and len(fname) > 0, "No filename given!"
        self.file = fname
        self.image.file = fname
        self.image.zstack = self.ctrl.zSpin.value()
        self.image.dnaChannel = self.ctrl.dnaSpin.value()
        self.ctrl.nchLbl.setText("%d channels" % self.image.nChannels)
        self.ctrl.nzsLbl.setText("%d z-stacks" % self.image.nZstack)
        self.ctrl.nfrLbl.setText("%d %s" % (self.image.nFrames, "frames" if self.image.nFrames > 1 else "frame"))
        self.currMeasurement = None
        self.currN = None
        self.currZ = None

        self.stk.close()
        self.stk = StkRingWidget(self.image,
                                 nucleus_id=self.image.currNucleusId,
                                 linePicked=self.onLinePickedFromStackGraph,
                                 line_length=self.line_length,
                                 dl=self.image.dl,
                                 lines_to_measure=self.image._nlin
                                 )
        # self.stk.linePicked.connect(self.onLinePickedFromStackGraph)
        self.stk.loadImages(self.image.images, xy=(100, 100), wh=(200, 200))
        self.stk.hide()
        self.stk.show()
        self.moveEvent(None)

    @QtCore.pyqtSlot()
    def onImgUpdate(self):
        # self.log.debug(f"onImgUpdate")
        self.ctrl.renderChk.setChecked(True)
        self.stk.selectedLineId = self.image.selectedLine if self.image.selectedLine is not None else 0
        self.log.debug(f"onImgUpdate. Selected line is {self.stk.selectedLineId}")
        self.stk.drawMeasurements(erase_bkg=True)
        self.grphtimer.start(1000)

    @QtCore.pyqtSlot()
    def onNucleusPickedFromImage(self):
        self.log.debug('onNucleusPickedFromImage')
        self.stk.dnaChannel = self.image.dnaChannel
        self.stk.rngChannel = self.image.rngChannel
        self.stk.selectedLineId = self.image.selectedLine if self.image.selectedLine is not None else 0
        self.stk.selectedNucId = self.image.currNucleusId if self.image.currNucleusId is not None else 0

        # test rectification code
        dl = 4
        ndl = 10
        nth = 100
        ppdl = 1
        ppth = 1

        tsplaprx = TestSplineApproximation(self.image.currNucleus, self.image)
        tsplaprx.test_fit()
        tsplaprx.plot_grid()

        trct = TestPiecewiseLinearRectification(tsplaprx,
                                                dl=dl, n_dl=ndl, n_theta=nth, pix_per_dl=ppdl, pix_per_theta=ppth)
        trct.plot_rectification()

        tfnrect = TestFunctionRectification(tsplaprx, dl=dl, n_dl=ndl, n_theta=nth, pix_per_dl=ppdl, pix_per_theta=ppth)
        tfnrect.plot_rectification()

        minx, miny, maxx, maxy = self.image.currNucleus.bounds
        r = int(max(maxx - minx, maxy - miny) / 2)
        self.stk.loadImages(self.image.images, xy=[n[0] for n in self.image.currNucleus.centroid.xy],
                            wh=(r * self.image.pix_per_um, r * self.image.pix_per_um))
        self.stk.measure()
        self.stk.drawMeasurements(erase_bkg=True)

    @QtCore.pyqtSlot()
    def onMeasureButton(self):
        self.log.debug('onMeasureButton')
        self.image.paint_measures()
        self._graph(alpha=0.2)
        self._graphTendency()

    @QtCore.pyqtSlot()
    def onZValueChange(self):
        self.log.debug('onZValueChange')
        self.image.zstack = self.ctrl.zSpin.value() % self.image.nZstack
        self.ctrl.zSpin.setValue(self.image.zstack)
        self._graph()

    @QtCore.pyqtSlot()
    def onDnaValChange(self):
        self.log.debug('onDnaValChange')
        val = self.ctrl.dnaSpin.value() % self.image.nChannels
        self.ctrl.dnaSpin.setValue(val)
        self.image.dnaChannel = val
        if self.ctrl.dnaChk.isChecked():
            self.image.activeCh = "dna"
        self.ctrl.dnaChk.setChecked(True)

    @QtCore.pyqtSlot()
    def onActValChange(self):
        self.log.debug('onActValChange')
        val = self.ctrl.actSpin.value() % self.image.nChannels
        self.ctrl.actSpin.setValue(val)
        self.image.rngChannel = val
        if self.ctrl.actChk.isChecked():
            self.image.activeCh = "act"
        self.ctrl.actChk.setChecked(True)

    @QtCore.pyqtSlot()
    def onAddButton(self):
        self.log.debug('onAddButton')
        if self.currMeasurement is not None and self.currN is not None and self.currZ is not None:
            new = pd.DataFrame(self.currMeasurement)
            new = new.loc[(new["n"] == self.currN) & (new["z"] == self.currZ)]
            new.loc[:, "m"] = self.measure_n
            new.loc[:, "file"] = os.path.basename(self.image.file)
            # new.loc[:, "x"] = new.loc[:, "l"].apply(lambda v: np.arange(start=0, stop=len(v), step=self.image.dl))
            self.df = self.df.append(new, ignore_index=True, sort=False)
            self.measure_n += 1
            self.currMeasurement = None
            self.currN = None
            self.currZ = None

            print(self.df)

    @QtCore.pyqtSlot()
    def onPlotButton(self):
        self.log.debug('onPlotButton')
        if self.image.measurements is None: return
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.gridspec import GridSpec

        plt.style.use('bmh')
        pal = sns.color_palette("Blues", n_colors=len(self.image.measurements))
        fig = plt.figure(figsize=(2, 2 * 4), dpi=300)
        gs = GridSpec(nrows=2, ncols=1, height_ratios=[4, 0.5])
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[1, 0])
        self.image.drawMeasurements(ax1, pal)

        lw = 1
        for me, c in zip(self.image.measurements, pal):
            x = np.arange(start=0, stop=len(me['l']) * self.image.dl, step=self.image.dl)
            ax2.plot(x, me['l'], linewidth=lw, linestyle='-', color=c, alpha=1, zorder=10)

        ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax1.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax1.yaxis.set_minor_locator(ticker.MultipleLocator(10))

        ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(1e4))
        ax2.yaxis.set_minor_locator(ticker.MultipleLocator(5e3))
        ax2.yaxis.set_major_formatter(EngFormatter(unit=''))

        fig.savefig(os.path.basename(self.image.file) + ".pdf")

    @QtCore.pyqtSlot()
    def onLinePickedFromGraph(self):
        self.log.debug('onLinePickedFromGraph')
        self.selectedLine = self.grph.selectedLine if self.grph.selectedLine is not None else None
        if self.selectedLine is not None:
            self.image.selectedLine = self.selectedLine

            self.currMeasurement = self.image.measurements
            self.currN = self.selectedLine
            self.currZ = self.image.zstack
            self.stk.selectedLineId = self.image.selectedLine if self.image.selectedLine is not None else 0
            self.stk.selectedNucId = self.image.currNucleusId if self.image.currNucleusId is not None else 0
            self.stk.selectedZ = self.currZ

            try:
                self.stk.drawMeasurements(erase_bkg=True)
            except Exception as e:
                self.log.error(e)

            self.statusbar.showMessage("line %d selected" % self.selectedLine)

    @QtCore.pyqtSlot()
    def onLinePickedFromStackGraph(self):
        self.log.debug('onLinePickedFromStackGraph')
        self.selectedLine = self.stk.selectedLineId if self.stk.selectedLineId is not None else None
        if self.selectedLine is not None:
            self.currN = self.stk.selectedLineId
            self.stk.selectedLineId = self.image.selectedLine if self.image.selectedLine is not None else 0
            self.stk.selectedNucId = self.image.currNucleusId if self.image.currNucleusId is not None else 0
            self.currZ = self.stk.selectedZ

            self.statusbar.showMessage(f"Line {self.currN} of z-stack {self.currZ} selected.")
            self.log.info(f"Line {self.currN} of z-stack {self.currZ} selected.")

    @QtCore.pyqtSlot()
    def onLinePickedFromImage(self):
        self.log.debug('onLinePickedFromImage')
        self.selectedLine = self.image.selectedLine if self.image.selectedLine is not None else None
        if self.selectedLine is not None:
            self.currN = self.selectedLine
            self.currZ = self.image.zstack
            self.stk.selectedLineId = self.image.selectedLine if self.image.selectedLine is not None else 0
            self.stk.selectedNucId = self.image.currNucleusId if self.image.currNucleusId is not None else 0
            self.stk.selectedZ = self.currZ

            self.statusbar.showMessage("Line %d selected" % self.selectedLine)


if __name__ == '__main__':
    from PyQt5.QtCore import QT_VERSION_STR
    from PyQt5.Qt import PYQT_VERSION_STR

    base_path = os.path.abspath('%s' % os.getcwd())
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.info('Qt version:' + QT_VERSION_STR)
    logging.info('PyQt version:' + PYQT_VERSION_STR)
    logging.info('Working dir:' + os.getcwd())
    logging.info('Base dir:' + base_path)
    os.chdir(base_path)

    app = QtGui.QApplication(sys.argv)

    gui = RingWindow()
    gui.show()

    sys.exit(app.exec_())
