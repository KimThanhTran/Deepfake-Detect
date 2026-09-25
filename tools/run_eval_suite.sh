#!/usr/bin/env bash
# Resumable 5-benchmark evaluation: score-level fusion of NPR + AdaptiveNPR+Aug(strong),
# then the adaptive_only ablation. Finished runs (metrics.csv / scores.csv present) are skipped,
# so re-running after an interruption continues where it stopped.
# Run inside the train container:  bash tools/run_eval_suite.sh
set -u

declare -A ROOTS=(
  [ForenSynths-test]=dataset/ForenSynths/ForenSynths/test
  [GANGen-Detection]=dataset/GANGen-Detection/GANGen-Detection
  [UniversalFakeDetect]=dataset/UniversalFakeDetect/UniversalFakeDetect
  [DiffusionForensics]=dataset/DiffusionForensics/DiffusionForensics
  [Diffusion1kStep]=dataset/Diffusion1kStep/Diffusion1kStep
)
BENCHES=(GANGen-Detection ForenSynths-test UniversalFakeDetect DiffusionForensics Diffusion1kStep)
PROGRESS=results/scores/progress.log
mkdir -p results/scores

log() { echo "$(date '+%F %T') $*" | tee -a "$PROGRESS"; }

run() {  # run <model> <out_dir> <log> <dataroot> [--save_scores]
  local model=$1 out=$2 logf=$3 root=$4; shift 4
  if [ -f "$out/metrics.csv" ] && { [ $# -eq 0 ] || [ -f "$out/scores.csv" ]; }; then
    log "skip $out"; return 0
  fi
  log "start $out"
  if python tools/eval_report.py --model_path "weights/$model.pth" --dataroot "$root" \
       --out_dir "$out" --label "$model - $(basename "$out")" \
       --batch_size 32 --num_workers 4 "$@" > "$logf" 2>&1; then
    log "done $out  $(grep MEAN "$logf")"
  else
    log "FAILED $out (see $logf)"
  fi
}

for ds in "${BENCHES[@]}"; do
  root=${ROOTS[$ds]}
  run NPR "results/scores/NPR_$ds" "results/scores/eval_NPR_$ds.log" "$root" --save_scores
  run npr_adaptive_aug "results/scores/npr_adaptive_aug_$ds" "results/scores/eval_npr_adaptive_aug_$ds.log" "$root" --save_scores
  if [ -f "results/scores/NPR_$ds/scores.csv" ] && [ -f "results/scores/npr_adaptive_aug_$ds/scores.csv" ]; then
    python tools/fuse_scores.py --a "results/scores/NPR_$ds" --b "results/scores/npr_adaptive_aug_$ds" \
      --out_dir "results/fusion_$ds" --label "fusion(NPR, AdaptiveNPR+Aug) - $ds" \
      > "results/scores/fuse_$ds.log" 2>&1
    log "fused $ds  $(grep MEAN "results/scores/fuse_$ds.log")"
  fi
done

for ds in "${BENCHES[@]}"; do
  run adaptive_only "results/adaptive_only_$ds" "results/eval_adaptive_only_$ds.log" "${ROOTS[$ds]}"
done

python tools/augment_metrics.py results > results/scores/augment.log 2>&1
log ALL_DONE
