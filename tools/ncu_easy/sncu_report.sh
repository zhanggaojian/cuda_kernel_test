#!/usr/bin/env bash
# Run Nsight Compute, save the .ncu-rep report, and extract key summary metrics.
#
# Default:
#   ./tools/ncu_easy/sncu_report.sh
#
# Equivalent profile command:
#   sncu --set full -f -o ./rep/sum_v1 ./output/sum_v1
#
# Custom:
#   ./tools/ncu_easy/sncu_report.sh ./output/histogram_v2 ./rep/histogram_v2
#
# Optional environment variables:
#   KERNEL_NAME=regex:histogram_v2
#   LAUNCH_SKIP=4
#   LAUNCH_COUNT=1
#   NCU_BIN=ncu
#   USE_SUDO=1
#
# Note:
#   Shell aliases are not inherited by scripts. If you have
#   alias sncu="sudo ncu" in ~/.bashrc, this script will invoke it through
#   an interactive bash shell when no real sncu executable is found.

set -euo pipefail

EXECUTABLE="${1:-./output/sum_v1}"
OUTPUT_PREFIX="${2:-./rep/sum_v1}"

if [[ ! -x "$EXECUTABLE" ]]; then
    echo "error: executable not found or not executable: $EXECUTABLE" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PREFIX")"

RUN_MODE="direct"

if [[ "${USE_SUDO:-0}" == "1" ]]; then
    if ! command -v sudo >/dev/null 2>&1; then
        echo "error: USE_SUDO=1 was set, but sudo was not found in PATH" >&2
        exit 1
    fi

    NCU_COMMAND="${NCU_BIN:-ncu}"
    if ! command -v "$NCU_COMMAND" >/dev/null 2>&1; then
        echo "error: ncu command not found: $NCU_COMMAND" >&2
        exit 1
    fi

    echo "== sudo authentication =="
    echo "Profiling requires sudo on this system. Enter your sudo password if prompted."
    sudo -v
    PROFILER_CMD=(sudo "$NCU_COMMAND")
elif [[ -n "${NCU_BIN:-}" ]]; then
    PROFILER_CMD=("$NCU_BIN")
elif command -v sncu >/dev/null 2>&1; then
    PROFILER_CMD=(sncu)
elif bash -ic 'type sncu >/dev/null 2>&1' >/dev/null 2>&1; then
    RUN_MODE="sncu_alias"
    PROFILER_CMD=(sncu)
elif command -v ncu >/dev/null 2>&1; then
    PROFILER_CMD=(ncu)
else
    echo "error: neither sncu nor ncu was found in PATH" >&2
    exit 1
fi

REPORT="${OUTPUT_PREFIX}.ncu-rep"
SUMMARY_TXT="${OUTPUT_PREFIX}_summary.txt"
KEY_TXT="${OUTPUT_PREFIX}_key_metrics.txt"

PROFILE_ARGS=(--set full -f -o "$OUTPUT_PREFIX")

if [[ -n "${KERNEL_NAME:-}" ]]; then
    PROFILE_ARGS+=(--kernel-name "$KERNEL_NAME")
fi

if [[ -n "${LAUNCH_SKIP:-}" ]]; then
    PROFILE_ARGS+=(--launch-skip "$LAUNCH_SKIP")
fi

if [[ -n "${LAUNCH_COUNT:-}" ]]; then
    PROFILE_ARGS+=(--launch-count "$LAUNCH_COUNT")
fi

echo "== Profile command =="
if [[ "$RUN_MODE" == "sncu_alias" ]]; then
    printf 'bash -ic %q bash ' 'sncu "$@"'
    printf '%q ' "${PROFILE_ARGS[@]}" "$EXECUTABLE"
else
    printf '%q ' "${PROFILER_CMD[@]}" "${PROFILE_ARGS[@]}" "$EXECUTABLE"
fi
printf '\n\n'

if [[ "$RUN_MODE" == "sncu_alias" ]]; then
    bash -ic 'sncu "$@"' bash "${PROFILE_ARGS[@]}" "$EXECUTABLE"
else
    "${PROFILER_CMD[@]}" "${PROFILE_ARGS[@]}" "$EXECUTABLE"
fi

if [[ ! -f "$REPORT" ]]; then
    echo "error: report was not generated: $REPORT" >&2
    exit 1
fi

echo
echo "== Export per-kernel summary =="
ncu --import "$REPORT" --print-summary per-kernel | tee "$SUMMARY_TXT"

echo
echo "== Extract key metrics =="
{
    echo "Report: $REPORT"
    echo "Summary: $SUMMARY_TXT"
    echo
    echo "Key metrics to inspect:"
    echo "- Duration"
    echo "- Memory Throughput"
    echo "- DRAM Throughput"
    echo "- Compute (SM) Throughput"
    echo "- Issue Slots Busy / SM Busy"
    echo "- Achieved Occupancy"
    echo "- One or More Eligible / No Eligible"
    echo "- Warp Cycles Per Issued Instruction"
    echo "- Registers Per Thread / Shared Memory Per Block / Grid Size"
    echo
    awk '
        /Invocations/ ||
        /Duration[[:space:]]/ ||
        /Memory Throughput/ ||
        /DRAM Throughput/ ||
        /Compute \(SM\) Throughput/ ||
        /Issue Slots Busy/ ||
        /SM Busy/ ||
        /Achieved Occupancy/ ||
        /Achieved Active Warps Per SM/ ||
        /One or More Eligible/ ||
        /No Eligible/ ||
        /Eligible Warps Per Scheduler/ ||
        /Issued Warp Per Scheduler/ ||
        /Warp Cycles Per Issued Instruction/ ||
        /Registers Per Thread/ ||
        /Static Shared Memory Per Block/ ||
        /Dynamic Shared Memory Per Block/ ||
        /Grid Size/ ||
        /Block Size/ ||
        /Waves Per SM/ {
            print
        }
    ' "$SUMMARY_TXT"
} | tee "$KEY_TXT"

echo
echo "Saved:"
echo "  report:  $REPORT"
echo "  summary: $SUMMARY_TXT"
echo "  key:     $KEY_TXT"
