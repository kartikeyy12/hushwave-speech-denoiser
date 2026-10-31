#!/usr/bin/env bash
# Print the path of the best Python interpreter for the project venv.
# Preference: 3.11, 3.12, 3.13, a uv-managed 3.11, then whatever python3 is.
# Override at any time with: make setup PYTHON=/path/to/python
set -u

is_ok() {
    # $1 = interpreter path; succeed if it is Python >= 3.10
    "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null
}

for name in python3.11 python3.12 python3.13; do
    if path="$(command -v "$name" 2>/dev/null)" && is_ok "$path"; then
        echo "$path"
        exit 0
    fi
done

if command -v uv >/dev/null 2>&1; then
    if path="$(uv python find 3.11 2>/dev/null)" && [ -n "$path" ] && is_ok "$path"; then
        echo "$path"
        exit 0
    fi
fi

if path="$(command -v python3 2>/dev/null)" && is_ok "$path"; then
    version="$("$path" -c 'import platform; print(platform.python_version())')"
    echo "warning: using python3 ($version); 3.11 is preferred, some metric packages may lack wheels" >&2
    echo "$path"
    exit 0
fi

echo "error: no Python >= 3.10 found" >&2
exit 1
