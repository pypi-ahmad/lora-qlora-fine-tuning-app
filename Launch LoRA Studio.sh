#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

for required_file in pyproject.toml uv.lock streamlit_app.py; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required project file not found: $required_file" >&2
        exit 1
    fi
done

if command -v uv >/dev/null 2>&1; then
    uv_exe="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
    uv_exe="$HOME/.local/bin/uv"
else
    echo "Installing uv for this Linux user..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
    else
        echo "Install curl or wget, then run this launcher again." >&2
        exit 1
    fi
    uv_exe="$HOME/.local/bin/uv"
fi

if [[ ! -x "$uv_exe" ]]; then
    echo "uv could not be installed." >&2
    exit 1
fi

echo "Preparing Python 3.14, .venv, and project dependencies..."
"$uv_exe" sync --locked --no-dev --python 3.14

mkdir -p .runs
pid_file=".runs/streamlit.pid"
if [[ -f "$pid_file" ]]; then
    existing_pid="$(<"$pid_file")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
        existing_command="$(tr '\0' ' ' <"/proc/$existing_pid/cmdline" 2>/dev/null || true)"
        if [[ "$existing_command" == *"streamlit_app.py"* ]]; then
            kill "$existing_pid"
            for _ in {1..20}; do
                kill -0 "$existing_pid" 2>/dev/null || break
                sleep 0.25
            done
            if kill -0 "$existing_pid" 2>/dev/null; then
                kill -9 "$existing_pid"
            fi
        fi
    fi
    rm -f "$pid_file"
fi

echo "Starting LoRA Fine-tune Studio on http://localhost:8504 ..."
nohup .venv/bin/python -m streamlit run streamlit_app.py \
    --server.headless=true --server.port=8504 \
    >.runs/streamlit.out.log 2>.runs/streamlit.err.log &
streamlit_pid=$!
printf '%s\n' "$streamlit_pid" >"$pid_file"

for _ in {1..180}; do
    if ! kill -0 "$streamlit_pid" 2>/dev/null; then
        rm -f "$pid_file"
        echo "Streamlit stopped during startup. Read .runs/streamlit.err.log" >&2
        exit 1
    fi
    if .venv/bin/python -c "from urllib.request import urlopen; urlopen('http://localhost:8504/_stcore/health', timeout=2)" 2>/dev/null; then
        if command -v xdg-open >/dev/null 2>&1; then
            xdg-open http://localhost:8504 >/dev/null 2>&1 || true
        fi
        echo "LoRA Fine-tune Studio is ready at http://localhost:8504"
        exit 0
    fi
    sleep 0.5
done

kill "$streamlit_pid" 2>/dev/null || true
rm -f "$pid_file"
echo "Streamlit did not become ready within 90 seconds. Read .runs/streamlit.err.log" >&2
exit 1
