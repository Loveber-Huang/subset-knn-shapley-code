#!/bin/bash

# ============================================
# Run data point selection experiment script
# Usage: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 bash scripts/run_data_selection.sh
# ============================================

# Set Python path to ensure algs module can be imported
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"


# Basic configuration parameters
rep=1  # Number of repetitions
seed=${1:-36}
k=5  # k value in KNN-Shapley
sleep_time=10
timestamp=$(date +%Y%m%d-%H%M%S)  # Timestamp for log files

# Dataset and model configuration
dataset="imdb-embedding"  # Optional:  'cifar10-embedding',  'covertype',  'imdb-embedding'
model="svm"  # Optional: 'logistic-regression', 'random-forest', 'gradient-boosting', 'svm', 'decision-tree' 'cnn'
max_points=100  # Maximum number of data points to select
dataset_path="$PROJECT_ROOT/data_files"  # Use absolute path to avoid relative path issues


# Get train, test sizes and max_points for a given dataset
get_dataset_config() {
    local dataset="$1"

    case "$dataset" in
        cifar10-embedding)
            echo "15000 1500 800"
            ;;
        imdb-embedding)
            echo "3000 1000 400"
            ;;
        covertype)
            echo "30000 5000 1000"
            ;;
        # Add more datasets here
        *)
            # Default values
            echo "12000 500 100"
            ;;
    esac
}

# Log directory
log_dir="../exps/logs"
mkdir -p "$log_dir"

# Experiment configuration arrays for batch experiments
# Define datasets and models to run
datasets=('imdb-embedding')
models=( 'logistic-regression')

# Define methods to run
methods=('all')

# Timeout configuration (in seconds)
timelimit=60000  

# Single experiment run function
run_experiment() {
    local ver="$1"
    local seed="$2"
    local dataset="$3"
    local model="$4"
    local k="$5"
    local max_points="$6"
    local dataset_path="$7"
    local train_size="$8"
    local test_size="$9"
    local method="${10}"
    local F="${11}"

    # Output information to stderr instead of stdout
    echo "============================================" >&2
    echo "Starting experiment: ver=$ver, dataset=$dataset, model=$model, method=$method" >&2
    echo "Random seed: $seed" >&2
    echo "============================================" >&2

    # Run Python experiment script (switch to project root directory to ensure correct paths)
    cd "$PROJECT_ROOT"

    # Create log directory structure: log_dir/dataset/model
    local dataset_model_log_dir="$log_dir/$dataset/$model"
    mkdir -p "$dataset_model_log_dir"

    # Run the command and capture only the PID
    local pid
    local log_file="$dataset_model_log_dir/$ver-$method-$timestamp.log"

    pid=$(nohup timeout $timelimit python3 exps/data_selection.py \
        --ver "$ver" \
        --seed "$seed" \
        --dataset "$dataset" \
        --model "$model" \
        --k "$k" \
        --max_points "$current_max_points" \
        --dataset_path "$dataset_path" \
        --train_size "$current_train_size" \
        --test_size "$current_test_size" \
        --method "$method" \
        --F $F \
        >> "$log_file" 2>&1 & echo $!)


    # Return to script directory
    cd - > /dev/null

    # Return only the PID (to stdout)
    echo "$pid"
}

# Main execution flow
main() {
    # Use automatically generated version number (based on timestamp)
    auto_ver="point_add_10"

    # Array to store process IDs
    pids=()

    # Run multiple experiments in batch
    for dataset in "${datasets[@]}"; do
        # Get train, test sizes and max_points for current dataset
        local current_train_size
        local current_test_size
        local current_max_points
        read current_train_size current_test_size current_max_points <<< $(get_dataset_config "$dataset")

        # If this dataset requires embedding extraction, precompute once to avoid
        # multiple parallel jobs racing and OOM'ing GPU memory.
        if [[ "$dataset" == *"-embedding" ]]; then
            echo "Precomputing embeddings for dataset=$dataset (train_size=$current_train_size, test_size=$current_test_size)..." >&2
            cd "$PROJECT_ROOT"
            python3 exps/precompute_embeddings.py \
                --dataset_path "$dataset_path" \
                --datasets "$dataset:$current_train_size:$current_test_size" \
                --seed "$seed" \
                || exit 1
            cd - > /dev/null
        fi

        for model in "${models[@]}"; do
            for method in "${methods[@]}"; do
                # Define F parameter (empty list by default)
                local F=""

                echo "Scheduling experiment: ver=$auto_ver, dataset=$dataset, model=$model, method=$method, F=$F" >&2
                echo "Using train_size=$current_train_size, test_size=$current_test_size, max_points=$current_max_points" >&2

                # Run experiment and get PID
                pid=$(run_experiment "$auto_ver" "$seed" "$dataset" "$model" "$k" "$current_max_points" "$dataset_path" "$current_train_size" "$current_test_size" "$method" "$F")
                pids+=($pid)

                # Wait for a short time before starting the next experiment
                sleep $sleep_time
            done
	    sleep $sleep_time
        done
    done

    # Wait for all experiments to complete
    echo "============================================"
    echo "Waiting for all experiments to complete..."
    echo "============================================"

    for pid in "${pids[@]}"; do
        wait "$pid"
        echo "Experiment with PID $pid completed"
    done

    echo "============================================"
    echo "All experiments completed!"
    echo "Log files saved at: $log_dir/"
    echo "============================================"
}

# Execute main function
main

