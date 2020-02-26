import logging
from typing import List

import numpy as np
import seaborn as sns
from PyQt5 import Qt, QtCore, QtWidgets
from PyQt5.QtCore import QRect, QTimer
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen
from PyQt5.QtWidgets import QLabel, QWidget
from shapely import affinity
import shapely.wkt

from gui._widget_graph import GraphWidget
from gui.measure import Measure
import measurements as m

logger = logging.getLogger('ring.stk.gui')


# noinspection PyPep8Naming
class StkRingWidget(QWidget):
    images: List[QLabel]
    linePicked = Qt.pyqtSignal()

    def __init__(self, measure: Measure, parent=None, nucleus_id=None,
                 line_length=4, dl=0.05, lines_to_measure=1, **kwargs):
        super().__init__(parent, **kwargs)
        # path = os.path.join(sys.path[0], __package__)

        # layout for images
        self.vLayout = QtWidgets.QHBoxLayout()
        self.setLayout(self.vLayout)

        self.images = list()
        for i in range(measure.nZstack):
            img = QtWidgets.QLabel()
            img.width = 100
            img.height = 100
            self.images.append(img)
            self.vLayout.addWidget(img)

        self.xy = (0, 0)
        self.wh = (0, 0)
        self.nlines = lines_to_measure
        self.dl = dl
        self.line_length = line_length

        self._meas = measure
        self.selectedNucId = nucleus_id
        self.selectedLineId = None
        self.selectedZ = None
        self._render = True

        self._images = None
        self._pixmaps = None
        self._nucboundaries = None
        self._colors = sns.husl_palette(lines_to_measure, h=.5).as_hex()

        # graph widget
        self.grph = GraphWidget()
        self.grph.setWindowTitle('Selected line across stack')
        self.grphtimer = QTimer()
        self.grphtimer.setSingleShot(True)

        self.grph.linePicked.connect(self.onLinePickedFromGraph)
        self.grph.linePicked.connect(self.linePicked)
        self.grphtimer.timeout.connect(self._graph)

        self.setWindowTitle('Stack images')
        self.grph.show()
        self.show()
        self.moveEvent(None)

    def moveEvent(self, QMoveEvent):
        logger.debug("moveEvent")
        px = self.geometry().x()
        py = self.geometry().y()
        pw = self.geometry().width()
        ph = self.geometry().height()

        # dw = self.grph.width()
        dh = self.grph.height()
        self.grph.setGeometry(px, py + ph + 20, pw, dh)

    def closeEvent(self, event):
        self.grph.close()

    def focusInEvent(self, QFocusEvent):
        logger.debug('focusInEvent')
        # self.activateWindow()
        self.grph.activateWindow()

    def showEvent(self, event):
        self.setFocus()

    def mouseReleaseEvent(self, ev):
        for k, im in enumerate(self.images):
            if im.underMouse():
                logger.info(f"image {k} clicked.")
                self.selectedZ = k
                self.linePicked.emit()

        self._repaintImages()
        self._graph()
        self.drawMeasurements()

    def _repaintImages(self):
        for i in range(len(self._pixmaps)):
            self.images[i].setPixmap(self._pixmaps[i])
            # self.images[i].setPixmap(
            #     self._pixmaps[i].scaled(self.images[i].width, self.images[i].height, QtCore.Qt.KeepAspectRatio))

    def loadImages(self, images, xy=(0, 0), wh=(1, 1)):
        assert (self._meas.dnaChannel is not None and
                self._meas.rngChannel is not None and
                self._meas.nChannels is not None), "Some parameters were not correctly set."
        # assert (len(boundaries) == self.zstacks and len(
        #     lines) == self.zstacks), "boundaries and lines should be the same length as zstacks in the image."

        # crop the image
        self.xy = np.array(xy).astype(np.int32)
        self.wh = np.array(wh).astype(np.int32)
        x1, x2 = int(xy[0] - wh[0] / 2), int(xy[0] + wh[0] / 2)
        y1, y2 = int(xy[1] - wh[1] / 2), int(xy[1] + wh[1] / 2)
        self._images = images[:, y1:y2, x1:x2]
        if self._images.size == 0:
            logger.warning(f"Empty cropped image! x1,x2,y1,y2 ({x1},{x2},{y1},{y2})")

        self._pixmaps = list()

        _old_zstk = self._meas.zstack
        for i in range(self._meas.nZstack):
            self._meas.zstack = i
            imagePixmap = self._meas.rngpixmap
            rect = QRect(x1, y1, wh[0], wh[1])
            cropped = imagePixmap.copy(rect)
            self._pixmaps.append(cropped)
        self._meas.zstack = _old_zstk

        self._repaintImages()
        self.update()
        return

    def measure(self):
        if self._images.size == 0:
            logger.warning("Can't measure an empty image!")
            return

        _old_zstk = self._meas.zstack
        self._nucboundaries = list()
        for i in range(self._meas.nZstack):
            self._meas.zstack = i
            x, y = self.xy
            w, h = self.wh
            nuc = self._meas.nucleus(x, y)
            if not nuc.empty:
                nucbnd = affinity.translate(shapely.wkt.loads(nuc["value"].iloc[0]), -x + h / 2, -y + w / 2)
                self._nucboundaries.append(nucbnd)
            else:
                self._nucboundaries.append(None)
        self._meas.zstack = _old_zstk

    # @profile
    def drawMeasurements(self, erase_bkg=False):
        if not self.renderMeasurements or not self._nucboundaries:
            return
        if erase_bkg:
            self._repaintImages()
        angle_delta = 2 * np.pi / self.nlines
        nim, width, height = self._images.shape

        for i, n in enumerate(self._nucboundaries):
            if not n:
                continue

            painter = QPainter()
            painter.begin(self.images[i].pixmap())
            painter.setRenderHint(QPainter.Antialiasing)

            nuc_pen = QPen(QBrush(QColor('white')), 1.1)
            nuc_pen.setStyle(QtCore.Qt.DotLine)
            painter.setPen(nuc_pen)
            dl2 = self.line_length * self._meas.pix_per_um / 2

            try:
                # get nuclei external and internal boundaries as a polygons
                for d in [dl2, -dl2]:
                    nucb_qpoints = [Qt.QPoint(x, y) for x, y in n.buffer(d).exterior.coords]
                    painter.drawPolygon(Qt.QPolygon(nucb_qpoints))
            except Exception as e:
                logger.error(e)

            if self.selectedLineId is not None:
                alpha = angle_delta * self.selectedLineId
                x, y = int(width / 2), int(height / 2)
                a = int(width / 2)
                pt1 = Qt.QPoint(x, y)
                pt2 = Qt.QPoint(a * np.cos(alpha) + x, a * np.sin(alpha) + y)
                painter.drawLine(pt1, pt2)

                for ix, me in self._meas.lines().iterrows():
                    if self.selectedLineId is not None and me['li'] == self.selectedLineId:
                        painter.setPen(QPen(QBrush(QColor(me['c'])), 1 * self._meas.pix_per_um))
                    else:
                        painter.setPen(QPen(QBrush(QColor('gray')), 0.1 * self._meas.pix_per_um))

                    pts = [Qt.QPoint(x, y) for x, y in [me['ls0'], me['ls1']]]
                    painter.drawLine(pts[0], pts[1])

            painter.end()
        self.grphtimer.start(1000)
        self.update()

    def _graph(self, alpha=1.0):
        self.grph.clear()
        _old_zstk = self._meas.zstack
        for i in range(self._meas.nZstack):
            self._meas.zstack = i
            if not self._meas.lines().empty:
                for ix, me in self._meas.lines().iterrows():
                    if me['li'] == self.selectedLineId:
                        x = np.arange(start=0, stop=len(me['value']) * self.dl, step=self.dl)
                        lw = 0.1 if self.selectedZ is not None and me['z'] != self.selectedZ else 0.5
                        self.grph.ax.plot(x, me['value'], linewidth=lw, linestyle='-', color=me['c'], alpha=alpha,
                                          zorder=10, picker=5, label=me['z'])
        self._meas.zstack = _old_zstk
        self.grph.format_ax()
        self.grph.canvas.draw()

    @property
    def selectedLine(self):
        if self.selectedLineId is not None and self.selectedZ is not None:
            for ix, me in self._meas.lines().iterrows():
                if me['li'] == self.selectedLineId and me['z'] == self.selectedZ:
                    return me
        return None

    @QtCore.pyqtSlot()
    def onLinePickedFromGraph(self):
        logger.debug('onLinePickedFromGraph')
        self.selectedZ = self.grph.selectedLine if self.grph.selectedLine is not None else None
        if self.selectedZ is not None:
            logger.debug(f"Z {self.selectedZ} selected")
            self._repaintImages()
            self.drawMeasurements()

            # self.emit(QtCore.SIGNAL('linePicked()'))
            # self.linePicked.emit()

    @property
    def renderMeasurements(self):
        return self._render

    @renderMeasurements.setter
    def renderMeasurements(self, value):
        if value is not None:
            self._render = value
            self._repaintImages()
            self.drawMeasurements()
