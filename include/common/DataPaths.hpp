#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>

namespace dlob {

inline std::string resolve_data_file(const std::string& filename) {
    namespace fs = std::filesystem;
    const fs::path supplied(filename);
    if (supplied.is_absolute() || fs::exists(supplied)) return supplied.string();

    if (const char* env = std::getenv("LOB_DATA_DIR"); env != nullptr && *env != '\0') {
        const fs::path candidate = fs::path(env) / supplied.filename();
        if (fs::exists(candidate)) return candidate.string();
    }

    const fs::path basename = supplied.filename();
    if (fs::exists(basename)) return basename.string();

    const fs::path under_data = fs::path("data") / basename;
    if (fs::exists(under_data)) return under_data.string();

    return supplied.string();
}

} // namespace dlob
