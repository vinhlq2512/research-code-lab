#!/bin/bash
# ==============================================================================
# TAPTA Data Scalability Study: Multi-Setting Runner
# Protocol: FewRel 8 tasks x 10 relations, Nested Sampling, M=0 Rehearsal-Free
# ==============================================================================
set -e

DEVICE=${1:-"cuda"}
EXPERIMENTS=("ds05" "ds10" "ds50" "ds100" "ds_full")

echo "=================================================================="
echo "🚀 TAPTA Data Scalability Study — Automated Suite Runner"
echo "Target Device: $DEVICE"
echo "Suite: ${EXPERIMENTS[*]}"
echo "=================================================================="

for EXP in "${EXPERIMENTS[@]}"; do
    CONFIG="configs/scalability/p4_fewrel_${EXP}_seed2021.yaml"
    echo ""
    echo "------------------------------------------------------------------"
    echo "▶ Running Experiment: $EXP with config: $CONFIG"
    echo "------------------------------------------------------------------"
    python experiments/run_scalability_study.py \
        --config "$CONFIG" \
        --device "$DEVICE" \
        --oracle_diagnostic
    echo "✅ Finished: $EXP"
done

echo ""
echo "=================================================================="
echo "🎉 ALL 5 SCALABILITY EXPERIMENTS COMPLETED!"
echo "Results stored in: runs/tapta_scalability/"
echo "=================================================================="
