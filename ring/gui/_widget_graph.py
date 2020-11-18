import os
import sys
import logging

from PyQt5 import Qt, uic
from PyQt5.QtWidgets import QWidget
import matplotlib.ticker as ticker
from matplotlib.ticker import EngFormatter

from ring.gui.gui_mplwidget import MplWidget

logger = logging.getLogger('ring.gui')


class GraphWidget(QWidget):
    graphWidget: MplWidget
    linePicked = Qt.pyqtSignal()

    def __init__(self):
        super(GraphWidget, self).__init__()
        path = os.path.join(sys.path[0], *__package__.split('.'))
        uic.loadUi(os.path.join(path, 'gui_ring_graph.ui'), self)
        self.canvas.callbacks.connect('pick_event', self.on_pick)

        self.selectedLine = None

        self.graphWidget.clear()
        self.format_ax()

    @property
    def canvas(self):
        return self.graphWidget.canvas

    @property
    def ax(self):
        return self.canvas.ax

    def clear(self):
        self.graphWidget.clear()
        self.selectedLine = None

    def on_pick(self, event):
        logger.info('on_pick')
        for l in self.ax.lines:
            l.set_linewidth(0.1)
        event.artist.set_linewidth(0.5)
        # logger.debug([l.get_label() for l in self.ax.lines])
        self.selectedLine = int(event.artist.get_label())
        self.linePicked.emit()
        self.canvas.draw()

    def format_ax(self):
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        self.ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        self.ax.yaxis.set_major_locator(ticker.MultipleLocator(1e4))
        self.ax.yaxis.set_minor_locator(ticker.MultipleLocator(5e3))
        self.ax.yaxis.set_major_formatter(EngFormatter(unit=''))
        # self.ax.set_ylim((0, 3e4))

    def resizeEvent(self, event):
        self.graphWidget.setFixedWidth(self.width())
