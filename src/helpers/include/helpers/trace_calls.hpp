#ifndef TRACE_CALLS_HPP_
#define TRACE_CALLS_HPP_

#include <source_location>

#include <spdlog/spdlog.h>

namespace hlp {

inline void trace_call([[maybe_unused]] std::source_location loc = std::source_location::current()) {
#if !defined(NDEBUG)
    // spdlog::debug("Function call: {}, {}:{}", loc.function_name(), loc.file_name(), loc.line());
    SPDLOG_TRACE("Function call: {}", loc.function_name());
#endif // NDEBUG
}

} // namespace hlp

#endif // TRACE_CALLS_HPP_
