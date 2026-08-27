#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 MODEL.gguf" >&2
    exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="$1"
shift
trace_path="${HAR_TRACE_OUT:-$(mktemp /tmp/har-native-runtime.XXXXXX.trace)}"
result_path="${HAR_TRACE_RESULT:-$(mktemp /tmp/har-native-runtime.XXXXXX.txt)}"

cargo build --release --manifest-path "$project_root/Cargo.toml" -p har-serve --bin har-server

binary="$project_root/target/release/har-server"
if command -v strace >/dev/null 2>&1; then
    strace -f -qq -e trace=process,file -o "$trace_path" \
        "$binary" --model "$model" --rows 16 --max-new 1 --prompt-ids 1,2,3 >"$result_path"
    exec_count="$(grep -c 'execve(' "$trace_path" || true)"
    if [[ "$exec_count" -ne 1 ]]; then
        echo "runtime trace rejected: expected one execve (the HAR binary), got $exec_count" >&2
        exit 1
    fi
    trace_method="strace process/file syscalls"
elif command -v gdb >/dev/null 2>&1; then
    # GDB is the bounded fallback when strace is unavailable. Catchpoints
    # cover helper-process execution; LD_DEBUG records the inferior's dynamic
    # loader dependencies. The inferior still runs directly, without a shell.
    model_arg="$(printf '%q' "$model")"
    gdb -q -nx -batch "$binary" \
        -ex 'set pagination off' \
        -ex 'set confirm off' \
        -ex 'set startup-with-shell off' \
        -ex 'set env LD_DEBUG libs,files' \
        -ex 'catch syscall execve' \
        -ex 'catch syscall execveat' \
        -ex "set args --model $model_arg --rows 16 --max-new 1 --prompt-ids 1,2,3" \
        -ex run >"$result_path" 2>"$trace_path"
    if grep -Eq '^Catchpoint [0-9]+$|^Catchpoint [0-9]+.*hit' "$result_path"; then
        echo "runtime trace rejected: helper-process execution catchpoint fired" >&2
        exit 1
    fi
    trace_method="gdb execve catchpoints + dynamic-loader trace"
else
    echo "runtime trace unavailable: install strace or gdb" >&2
    exit 1
fi

if grep -Eiq 'python|cmake|llama|ggml|libstdc\+\+|libc\+\+' "$trace_path" "$result_path"; then
    echo "runtime trace rejected: forbidden runtime dependency/process observed" >&2
    exit 1
fi

echo "NATIVE_RUNTIME_TRACE PASS"
echo "method=$trace_method"
echo "model=$model"
echo "result=$result_path"
echo "trace=$trace_path"
