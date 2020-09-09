import io
import os
import logging
import xml.etree.ElementTree
from collections import namedtuple

import numpy as np
import tifffile as tf
from czifile import CziFile
from PyQt5.QtGui import QImage, QPixmap

logger = logging.getLogger(__name__)

MetadataImage = namedtuple('MetadataImage', ['image', 'pix_per_um', 'um_per_pix',
                                             'time_interval', 'frames', 'channels',
                                             'width', 'height', 'series'])


def load_tiff(file_or_path):
    if type(file_or_path) == str:
        _, img_name = os.path.split(file_or_path)
    if issubclass(type(file_or_path), io.BufferedIOBase):
        _, img_name = os.path.split(file_or_path.name)

    res = None
    with tf.TiffFile(file_or_path) as tif:
        assert len(tif.series) == 1, "Not currently handled."
        idx = tif.series[0].axes
        width = tif.series[0].shape[idx.find('X')]
        height = tif.series[0].shape[idx.find('Y')]

        if tif.is_imagej is not None:
            metadata = tif.imagej_metadata
            dt = metadata['finterval'] if 'finterval' in metadata else 1

            # asuming square pixels
            if 'XResolution' in tif.pages[0].tags:
                xr = tif.pages[0].tags['XResolution'].value
                res = float(xr[0]) / float(xr[1])  # pixels per um
                if metadata['unit'] == 'centimeter':
                    res = res / 1e4

            images = None
            if len(tif.pages) == 1:
                if ('slices' in metadata and metadata['slices'] > 1) or (
                        'frames' in metadata and metadata['frames'] > 1):
                    images = tif.pages[0].asarray()
                else:
                    images = [tif.pages[0].asarray()]
            elif len(tif.pages) > 1:
                images = list()
                for i, page in enumerate(tif.pages):
                    images.append(page.asarray())

            return MetadataImage(image=np.asarray(images), pix_per_um=res, um_per_pix=1. / res, time_interval=dt,
                                 frames=metadata['frames'] if 'frames' in metadata else 1,
                                 channels=metadata['channels'] if 'channels' in metadata else 1,
                                 width=width, height=height, series=tif.series[0])


def load_zeiss(path):
    _, img_name = os.path.split(path)
    with CziFile(path) as czi:
        xmltxt = czi.metadata()
        meta = xml.etree.ElementTree.fromstring(xmltxt)

        # next line is somewhat cryptic, but just extracts um/pix (calibration) of X and Y into res
        res = [float(i[0].text) for i in meta.findall('.//Scaling/Items/*') if
               i.attrib['Id'] == 'X' or i.attrib['Id'] == 'Y']
        assert np.isclose(res[0], res[1]), "Pixels are not square."

        # get first calibration value and convert it from meters to um
        res = res[0] * 1e6

        ts_ix = [k for k, a1 in enumerate(czi.attachment_directory) if a1.filename[:10] == 'TimeStamps'][0]
        timestamps = list(czi.attachments())[ts_ix].data()
        dt = np.median(np.diff(timestamps))

        ax_dct = {n: k for k, n in enumerate(czi.axes)}
        n_frames = czi.shape[ax_dct['T']]
        n_channels = czi.shape[ax_dct['C']]
        n_X = czi.shape[ax_dct['X']]
        n_Y = czi.shape[ax_dct['Y']]

        images = list()
        for sb in czi.subblock_directory:
            images.append(sb.data_segment().data().reshape((n_X, n_Y)))

        return MetadataImage(image=np.array(images), pix_per_um=1. / res, um_per_pix=res, time_interval=dt,
                             frames=n_frames, channels=n_channels,
                             width=n_X, height=n_Y, series=None)


def find_image(img_name, folder=None):
    if folder is None:
        folder = os.path.dirname(img_name)
        img_name = os.path.basename(img_name)

    for root, directories, filenames in os.walk(folder):
        for file in filenames:
            joinf = os.path.abspath(os.path.join(root, file))
            if os.path.isfile(joinf) and joinf[-4:] == '.tif' and file == img_name:
                return load_tiff(joinf)
            if os.path.isfile(joinf) and joinf[-4:] == '.czi' and file == img_name:
                return load_zeiss(joinf)


def retrieve_image(image_arr, frame=0, zstack=0, channel=0, number_of_channels=1, number_of_zstacks=1, to_8bit=True):
    if image_arr is not None:
        ix = frame * (number_of_channels * number_of_zstacks) + zstack * number_of_channels + channel
        logger.debug("Retrieving frame %d of channel %d at z-stack=%d (index=%d)" % (frame, channel, zstack, ix))

        data = image_arr[ix]
        # normalize and convert image to unsigned 8 bit
        if to_8bit:
            info = np.iinfo(data.dtype)  # Get the information of the incoming image type
            data = data.astype(np.float64) / info.max  # normalize the data to 0 - 1
            data = 255 * data  # Now scale by 255
            data = data.astype(np.uint8)

        return data


def image_iterator(image_arr, channel=0, number_of_frames=1):
    nimgs = image_arr.shape[0]
    n_channels = int(nimgs / number_of_frames)
    for f in range(number_of_frames):
        ix = f * n_channels + channel
        logger.debug("Retrieving frame %d of channel %d (index=%d)" % (f, channel, ix))
        if ix < nimgs: yield image_arr[ix]


def qpixmap_from(image: np.array):
    dwidth, dheight = image.shape

    # map the data range to 0 - 255
    img_8bit = ((image - image.min()) / (image.ptp() / 255.0)).astype(np.uint8)
    # qimg = QImage(img_8bit.repeat(4), dwidth, dheight, QImage.Format_RGB32)
    qimg = QImage(img_8bit.repeat(3), dwidth, dheight, QImage.Format_RGB888)
    return QPixmap(qimg)
