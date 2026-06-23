#!/bin/bash
#
# Run the `tum` SLAM binary over each TUM RGB-D scene and evaluate the estimated
# trajectory against ground truth with evo (APE).
#
# Usage:
#   ./eval.sh                                  # auto-discover every scene with a groundtruth
#   ./eval.sh rgbd_dataset_freiburg1_xyz ...   # run only the named scenes
#   ./eval.sh --live [scenes...]               # visualize the Rerun stream live
#                                              # instead of recording it to .rrd
#
# Visualization (how the binary's Rerun stream is handled):
#   default        record to "$RESULTS_DIR/<scene>/<scene>.rrd" — headless, no
#                  display needed. Open later with: rerun <file>.rrd
#   --live         live view. Connects to an already-running viewer if
#                  RERUN_ADDRESS is set (e.g. RERUN_ADDRESS=127.0.0.1:9876, one
#                  viewer reused across scenes); otherwise spawns a native
#                  viewer per scene. Requires a display.
#
# Contract with the binary (see examples/tum.cpp):
#   ./<BIN> <scene>  ->  writes results/<scene>.txt  (TUM format)  then exits.
#
set -uo pipefail

# --- configuration ----------------------------------------------------------
BIN="${BIN:-./release_build/tum}"
DATASETS_DIR="${DATASETS_DIR:-datasets}"
RESULTS_DIR="${RESULTS_DIR:-results}"
# Extra flags forwarded to evo_ape (alignment + Sim(3) scale correction).
EVO_FLAGS="${EVO_FLAGS:---align --correct_scale}"
# Visualization mode: "save" (record .rrd, default) or "live" (--live).
VIZ_MODE="save"

# --- pre-flight checks ------------------------------------------------------
if [[ ! -x "$BIN" ]]; then
    echo "ERROR: binary '$BIN' not found or not executable." >&2
    echo "       Build it first (e.g. ./build.sh) or set BIN=..." >&2
    exit 1
fi
if ! command -v evo_ape >/dev/null 2>&1; then
    echo "ERROR: evo_ape not on PATH. Activate the venv (source /opt/taey/bin/activate)" >&2
    echo "       or 'pip install evo'." >&2
    exit 1
fi

# Force evo's matplotlib backend to the non-interactive Agg. With --save_plot,
# evo's plot.py calls mpl.use(settings.plot_backend) using its *own* configured
# backend (tkagg) — which overrides $MPLBACKEND and dies with "No module named
# 'tkinter'" on this headless host. Agg renders the PDFs without a display.
evo_config set plot_backend Agg >/dev/null 2>&1 || true

# --- argument parsing -------------------------------------------------------
# Flags are recognized anywhere; remaining (non-flag) args are scene names.
scene_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --live|-l) VIZ_MODE="live" ;;
        --save)    VIZ_MODE="save" ;;
        -h|--help)
            sed -n '2,/^set /p' "$0" | sed 's/^# \?//; /^set /d'
            exit 0 ;;
        --) shift; scene_args+=("$@"); break ;;
        -*) echo "ERROR: unknown option '$1' (try --help)." >&2; exit 2 ;;
        *)  scene_args+=("$1") ;;
    esac
    shift
done

# --- scene list -------------------------------------------------------------
# Explicit args win; otherwise discover every dataset dir that has a groundtruth.
scenes=()
if [[ ${#scene_args[@]} -gt 0 ]]; then
    scenes=("${scene_args[@]}")
else
    for d in "$DATASETS_DIR"/*/; do
        [[ -f "${d}groundtruth.txt" && -f "${d}calibration.yaml" ]] || continue
        scenes+=("$(basename "$d")")
    done
fi

if [[ ${#scenes[@]} -eq 0 ]]; then
    echo "ERROR: no scenes to evaluate (none found under '$DATASETS_DIR' with groundtruth.txt)." >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# --- run + evaluate each scene ---------------------------------------------
declare -A status_of   # scene -> OK / FAIL:<reason>
declare -A rmse_of     # scene -> APE rmse

for scene in "${scenes[@]}"; do
    echo
    echo "=================================================================="
    echo "=== $scene"
    echo "=================================================================="

    gt="$DATASETS_DIR/$scene/groundtruth.txt"
    est="$RESULTS_DIR/$scene.txt"
    out_dir="$RESULTS_DIR/$scene"
    zip="$RESULTS_DIR/$scene.zip"

    if [[ ! -f "$gt" ]]; then
        echo "  SKIP: groundtruth not found ($gt)"
        status_of["$scene"]="FAIL:no-groundtruth"
        continue
    fi

    # Clean only this scene's prior results.
    rm -rf "$out_dir" "$zip" "$est"
    mkdir -p "$out_dir"

    # --- run the SLAM binary (writes results/<scene>.txt) ---
    # The Rerun sink is selected via env vars read by Visualizer (priority:
    # RERUN_SAVE > RERUN_ADDRESS > spawn). See src/Visualizer.cpp.
    if [[ "$VIZ_MODE" == "live" ]]; then
        # Live view. Connect to a running viewer if RERUN_ADDRESS is set (one
        # viewer reused across scenes); otherwise let the binary spawn its own.
        # Pass through only RERUN_ADDRESS so RERUN_SAVE never wins the priority.
        if [[ -n "${RERUN_ADDRESS:-}" ]]; then
            echo "  Running $BIN $scene (live -> $RERUN_ADDRESS) ..."
            run_cmd=(env -u RERUN_SAVE RERUN_ADDRESS="$RERUN_ADDRESS" "$BIN" "$scene")
        else
            echo "  Running $BIN $scene (live -> spawn viewer) ..."
            run_cmd=(env -u RERUN_SAVE -u RERUN_ADDRESS "$BIN" "$scene")
        fi
    else
        # Record the Rerun stream to an .rrd file instead of spawning the native
        # viewer — the eval host may be headless (no GPU/display), where spawn()
        # aborts. Open a recording later with: rerun "$out_dir/$scene.rrd".
        echo "  Running $BIN $scene (recording -> $out_dir/$scene.rrd) ..."
        run_cmd=(env RERUN_SAVE="$out_dir/$scene.rrd" "$BIN" "$scene")
    fi
    if ! "${run_cmd[@]}"; then
        echo "  FAIL: '$BIN $scene' exited non-zero."
        status_of["$scene"]="FAIL:run"
        continue
    fi
    if [[ ! -s "$est" ]]; then
        echo "  FAIL: expected trajectory '$est' was not produced (or is empty)."
        status_of["$scene"]="FAIL:no-trajectory"
        continue
    fi

    # --- evaluate with evo (APE) ---
    echo "  Evaluating with evo_ape ..."
    eval_log="$out_dir/evo_ape.log"
    # shellcheck disable=SC2086  # EVO_FLAGS is intentionally word-split
    if ! evo_ape tum "$gt" "$est" \
            $EVO_FLAGS \
            -va --plot_mode xz \
            --save_results "$zip" \
            --save_plot "$out_dir/$scene.pdf" \
            2>&1 | tee "$eval_log"; then
        echo "  FAIL: evo_ape returned an error (see $eval_log)."
        status_of["$scene"]="FAIL:evo"
        continue
    fi

    if [[ -f "$zip" ]]; then
        unzip -o "$zip" -d "$out_dir/" >/dev/null
        rm -f "$zip"
    fi

    # Pull the APE RMSE out of the evo log for the summary table.
    rmse=$(awk '/^[[:space:]]*rmse/ {print $2; exit}' "$eval_log")
    rmse_of["$scene"]="${rmse:-?}"
    status_of["$scene"]="OK"
    echo "  APE RMSE: ${rmse:-?}"
done

# --- summary ----------------------------------------------------------------
echo
echo "=================================================================="
echo "=== Summary"
echo "=================================================================="
printf "%-55s %-18s %s\n" "SCENE" "STATUS" "APE_RMSE"
fails=0
for scene in "${scenes[@]}"; do
    st="${status_of[$scene]:-FAIL:unknown}"
    [[ "$st" == OK ]] || fails=$((fails + 1))
    printf "%-55s %-18s %s\n" "$scene" "$st" "${rmse_of[$scene]:-}"
done

echo
if [[ $fails -gt 0 ]]; then
    echo "$fails scene(s) failed."
    exit 1
fi
echo "All ${#scenes[@]} scene(s) evaluated. Results in '$RESULTS_DIR/'."
