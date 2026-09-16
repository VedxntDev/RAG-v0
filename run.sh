#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python3"

if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

export PYTHONPATH="${SCRIPT_DIR}"

MODE="${1:-all}"

case "$MODE" in
    test|--test)
        echo "=== Running Automated Test Suite ==="
        "$PYTHON_BIN" -m pytest -v "${SCRIPT_DIR}/tests"
        ;;
    bench|--bench)
        echo "=== Running Benchmarks ==="
        "$PYTHON_BIN" "${SCRIPT_DIR}/benchmark/run_benchmarks.py"
        ;;
    ablate|--ablate)
        echo "=== Running Ablation Studies ==="
        "$PYTHON_BIN" "${SCRIPT_DIR}/benchmark/run_ablations.py"
        ;;
    serve|--serve)
        echo "=== Launching FastAPI Server on Port 8085 ==="
        "$PYTHON_BIN" -m uvicorn src.api.server:app --host 0.0.0.0 --port 8085
        ;;
    all|--all)
        echo "===================================================="
        echo "   STREAMING LIVE RAG ENGINE — FULL SUITE EXECUTION "
        echo "===================================================="
        echo "[1/3] Running Automated Tests..."
        "$PYTHON_BIN" -m pytest -v "${SCRIPT_DIR}/tests"
        echo ""
        echo "[2/3] Running Benchmark Suite..."
        "$PYTHON_BIN" "${SCRIPT_DIR}/benchmark/run_benchmarks.py"
        echo ""
        echo "[3/3] Running Ablation Studies..."
        "$PYTHON_BIN" "${SCRIPT_DIR}/benchmark/run_ablations.py"
        echo ""
        echo "===================================================="
        echo "   ALL TECHNICAL EVALUATION GATES (G1-G6) PASSED!   "
        echo "===================================================="
        ;;
    *)
        echo "Usage: ./run.sh [test|bench|ablate|serve|all]"
        exit 1
        ;;
esac
