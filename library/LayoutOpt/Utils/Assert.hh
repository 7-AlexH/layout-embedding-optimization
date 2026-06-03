#pragma once

#include <cmath>
#include <iostream>
#include "StackTrace.hh"

#define ERROR(str) std::cout << "[ERROR] " << str << " (in function " << __FUNCTION__ << ", " << __FILE__ << ":" << __LINE__ << ")" << std::endl

#define ERROR_THROW(str)                   \
    {                                      \
        ERROR(str);                        \
        LayoutOpt::print_stack_trace();    \
        throw std::runtime_error("ERROR"); \
    }

#define ASSERT(exp)                                 \
    {                                               \
        if (!(exp))                                 \
            ERROR_THROW("Assertion failed: " #exp); \
    }

#define ASSERT_MSG(exp, msg)                                    \
    {                                                           \
        if (!(exp))                                             \
            ERROR_THROW("Assertion failed: " #exp ". " << msg); \
    }

#define ASSERT_EQ(a, b)                                        \
    {                                                          \
        if (!((a) == (b)))                                     \
            ERROR_THROW("Assertion failed: " #a " == " #b "\n" \
                        "    " #a " = "                        \
                        << (a)                                 \
                        << "\n"                                \
                           "    " #b " = "                     \
                        << (b) << "\n");                       \
    }

#define ASSERT_NEQ(a, b)                                       \
    {                                                          \
        if (!((a) != (b)))                                     \
            ERROR_THROW("Assertion failed: " #a " != " #b "\n" \
                        "    " #a " = "                        \
                        << (a)                                 \
                        << "\n"                                \
                           "    " #b " = "                     \
                        << (b) << "\n");                       \
    }

#define ASSERT_G(a, b)                                        \
    {                                                         \
        if (!((a) > (b)))                                     \
            ERROR_THROW("Assertion failed: " #a " > " #b "\n" \
                        "    " #a " = "                       \
                        << (a)                                \
                        << "\n"                               \
                           "    " #b " = "                    \
                        << (b) << "\n");                      \
    }

#define ASSERT_GEQ(a, b)                                       \
    {                                                          \
        if (!((a) >= (b)))                                     \
            ERROR_THROW("Assertion failed: " #a " >= " #b "\n" \
                        "    " #a " = "                        \
                        << (a)                                 \
                        << "\n"                                \
                           "    " #b " = "                     \
                        << (b) << "\n");                       \
    }

#define ASSERT_L(a, b)                                        \
    {                                                         \
        if (!((a) < (b)))                                     \
            ERROR_THROW("Assertion failed: " #a " < " #b "\n" \
                        "    " #a " = "                       \
                        << (a)                                \
                        << "\n"                               \
                           "    " #b " = "                    \
                        << (b) << "\n");                      \
    }

#define ASSERT_LEQ(a, b)                                       \
    {                                                          \
        if (!((a) <= (b)))                                     \
            ERROR_THROW("Assertion failed: " #a " <= " #b "\n" \
                        "    " #a " = "                        \
                        << (a)                                 \
                        << "\n"                                \
                           "    " #b " = "                     \
                        << (b) << "\n");                       \
    }

#define ASSERT_EPS(a, b, eps)                                              \
    {                                                                      \
        if (std::abs((a) - (b)) >= (eps))                                  \
            ERROR_THROW("Assertion failed: |" #a " - " #b "| < " #eps "\n" \
                        "    " #a " = "                                    \
                        << (a)                                             \
                        << "\n"                                            \
                           "    " #b " = "                                 \
                        << (b)                                             \
                        << "\n"                                            \
                           "    |" #a " - " #b "| = "                      \
                        << std::abs((a) - (b)) << "\n");                   \
    }
