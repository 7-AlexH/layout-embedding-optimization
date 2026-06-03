#pragma once

#include <iostream>
#include <typed-geometry/tg-std.hh>

#define DEBUG_OUT(msg)                               \
    {                                                \
        std::cout << "[DEBUG] " << msg << std::endl; \
    }

#define WARNING(exp, msg)                                  \
    {                                                      \
        if (!(exp))                                        \
            std::cout << "[WARNING] " << msg << std::endl; \
    }

#define DEBUG_VAR(var) DEBUG_OUT(#var " = " << (var))
