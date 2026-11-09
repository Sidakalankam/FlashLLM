
#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo "=== FlashLLM Worker Starting ==========="
echo "========================================"

#
# Required configuration
#

: "${MODEL_NAME:?MODEL_NAME is required}"

if [[ ! "$MODEL_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] ||
   [[ "$MODEL_NAME" == "." ]] ||
   [[ "$MODEL_NAME" == ".." ]]; then
    echo "ERROR: Invalid MODEL_NAME: $MODEL_NAME"
    exit 1
fi

#
# Model configuration
#

MODEL_SOURCE="${MODEL_SOURCE:-local}"
MODEL_ROOT="${MODEL_ROOT:-/models}"
MODEL_DIR="${MODEL_ROOT%/}/$MODEL_NAME"

#
# vLLM configuration
#

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DTYPE="${DTYPE:-auto}"
QUANTIZATION="${QUANTIZATION:-}"
VLLM_EXTRA_ARGS_JSON="${VLLM_EXTRA_ARGS_JSON:-[]}"

#
# Server configuration
#

PORT="${PORT:-8000}"

echo ""
echo "Configuration:"
echo "  Model:          $MODEL_NAME"
echo "  Model source:   $MODEL_SOURCE"
echo "  Model path:     $MODEL_DIR"
echo "  Port:           $PORT"
echo "  Max context:    $MAX_MODEL_LEN"
echo "  GPU memory:     $GPU_MEMORY_UTILIZATION"
echo "  Dtype:          $DTYPE"
echo ""

#
# Verify model storage exists
#

if [[ ! -d "$MODEL_ROOT" ]]; then
    echo "ERROR: Model storage root does not exist: $MODEL_ROOT"
    exit 1
fi

#
# Validate model files
#

validate_model() {
    local directory="$1"

    if [[ ! -f "$directory/config.json" ]]; then
        echo "ERROR: Missing config.json in $directory"
        return 1
    fi

    python3 - "$directory" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
index_path = model_dir / "model.safetensors.index.json"

if index_path.exists():
    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)

    weight_map = index.get("weight_map")

    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Invalid safetensors weight_map")

    expected_files = set(weight_map.values())

    for filename in expected_files:
        path = model_dir / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing safetensors shard: {filename}"
            )

    print(f"Validated {len(expected_files)} safetensors shards.")

else:
    weights = (
        list(model_dir.glob("*.safetensors"))
        + list(model_dir.glob("*.bin"))
    )

    if not weights:
        raise FileNotFoundError("No model weight files found")

    print(f"Found {len(weights)} model weight files.")
PY
}

#
# Handle model source
#

case "$MODEL_SOURCE" in

    local)
        echo "Using locally cached model."

        if [[ ! -d "$MODEL_DIR" ]]; then
            echo "ERROR: Local model directory not found."
            exit 1
        fi

        validate_model "$MODEL_DIR"
        ;;

    s3)
        : "${MODEL_BUCKET:?MODEL_BUCKET is required for S3 models}"

        MODEL_PREFIX="${MODEL_PREFIX:-models/$MODEL_NAME}"
        MODEL_PREFIX="${MODEL_PREFIX%/}"

        if [[ ! "$MODEL_BUCKET" =~ ^[a-zA-Z0-9][a-zA-Z0-9.-]*$ ]] ||
           [[ "$MODEL_PREFIX" == /* ]] ||
           [[ "$MODEL_PREFIX" == *".."* ]] ||
           [[ -z "$MODEL_PREFIX" ]]; then
            echo "ERROR: Invalid S3 model location."
            exit 1
        fi

        CACHE_MARKER="$MODEL_DIR/.flashllm-complete"

        #
        # Reuse a completed cache when valid.
        #

        if [[ -f "$CACHE_MARKER" ]]; then
            echo "Found completed model cache."

            if validate_model "$MODEL_DIR"; then
                echo "Using cached model."
            else
                echo "Cached model validation failed."
                rm -f "$CACHE_MARKER"
            fi
        fi

        #
        # Download into a staging directory.
        #

        if [[ ! -f "$CACHE_MARKER" ]]; then
            echo "Downloading model from S3..."
            echo "  s3://$MODEL_BUCKET/$MODEL_PREFIX"

            STAGING_DIR=$(mktemp -d \
                "${MODEL_ROOT%/}/.${MODEL_NAME}.staging.XXXXXX")

            BACKUP_DIR=""

            cleanup() {
                if [[ -n "${STAGING_DIR:-}" && -d "$STAGING_DIR" ]]; then
                    rm -rf -- "$STAGING_DIR"
                fi

                if [[ -n "${BACKUP_DIR:-}" && -d "$BACKUP_DIR" ]]; then
                    if [[ ! -e "$MODEL_DIR" ]]; then
                        mv -- "$BACKUP_DIR" "$MODEL_DIR"
                    fi
                fi
            }

            trap cleanup EXIT

            aws s3 sync \
                "s3://$MODEL_BUCKET/$MODEL_PREFIX/" \
                "$STAGING_DIR/" \
                --only-show-errors

            validate_model "$STAGING_DIR"

            #
            # Preserve the old cache until the new
            # model has been installed successfully.
            #

            if [[ -e "$MODEL_DIR" ]]; then
                BACKUP_DIR=$(mktemp -d \
                    "${MODEL_ROOT%/}/.${MODEL_NAME}.backup.XXXXXX")

                rmdir "$BACKUP_DIR"

                mv -- "$MODEL_DIR" "$BACKUP_DIR"
            fi

            if ! mv -- "$STAGING_DIR" "$MODEL_DIR"; then
                echo "ERROR: Failed to install downloaded model."

                if [[ -n "$BACKUP_DIR" && -d "$BACKUP_DIR" ]]; then
                    mv -- "$BACKUP_DIR" "$MODEL_DIR"
                    BACKUP_DIR=""
                fi

                exit 1
            fi

            STAGING_DIR=""

            touch "$CACHE_MARKER"

            #
            # Remove the old cache only after the
            # replacement is marked complete.
            #

            if [[ -n "$BACKUP_DIR" && -d "$BACKUP_DIR" ]]; then
                rm -rf -- "$BACKUP_DIR"
                BACKUP_DIR=""
            fi

            trap - EXIT

            echo "Model downloaded and cached successfully."
        fi
        ;;

    *)
        echo "ERROR: Unsupported MODEL_SOURCE: $MODEL_SOURCE"
        exit 1
        ;;

esac

#
# Construct vLLM arguments
#

VLLM_ARGS=(
    "$MODEL_DIR"
    --served-model-name "$MODEL_NAME"
    --dtype "$DTYPE"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --host 0.0.0.0
    --port "$PORT"
)

if [[ -n "$QUANTIZATION" ]]; then
    VLLM_ARGS+=(--quantization "$QUANTIZATION")
fi

#
# Validate optional vLLM arguments.
# Must be a JSON array of strings.
#

ARGS_FILE=$(mktemp)
trap 'rm -f -- "$ARGS_FILE"' EXIT

if ! python3 - "$VLLM_EXTRA_ARGS_JSON" > "$ARGS_FILE" <<'PY'
import json
import sys

try:
    args = json.loads(sys.argv[1])

    if not isinstance(args, list) or not all(
        isinstance(arg, str) for arg in args
    ):
        raise ValueError(
            "VLLM_EXTRA_ARGS_JSON must be a JSON array of strings"
        )

    if any("\x00" in arg for arg in args):
        raise ValueError("Arguments cannot contain null bytes")

    for arg in args:
        sys.stdout.buffer.write(arg.encode("utf-8") + b"\0")

except (json.JSONDecodeError, ValueError) as exc:
    print(f"ERROR: Invalid vLLM arguments: {exc}", file=sys.stderr)
    sys.exit(1)
PY
then
    exit 1
fi

mapfile -d '' -t EXTRA_ARGS < "$ARGS_FILE"
rm -f -- "$ARGS_FILE"
trap - EXIT

VLLM_ARGS+=("${EXTRA_ARGS[@]}")

echo ""
echo "Starting vLLM..."
echo "Model: $MODEL_NAME"
echo "Port:  $PORT"
echo ""

exec vllm serve "${VLLM_ARGS[@]}"
