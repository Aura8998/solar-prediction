import os
import shutil
import uuid
import gc
from turtle import pd

import psutil
import cv2
import h5py
import numpy as np
import tensorflow as tf
from flask import Blueprint, render_template, request, jsonify
from pathlib import Path
import traceback
from datetime import datetime, date
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from functools import lru_cache

# Configuration constants
WEIGHTS_PATH = "static/DLFile/final_model_weights.h5"
ARCHITECTURE_PATH = "static/DLFile/model_architecture.json"
TRAIN_INDICES_PATH = "data/selected_indices.npy"
SHARED_DATA_DIR = "static/shared_data"
os.makedirs(SHARED_DATA_DIR, exist_ok=True)

bp = Blueprint("functionsPage", __name__, url_prefix="/functions")

# Memory management configuration
MAX_MEMORY_MB = 2048

SAMPLE_RATIO = 0.2

# Global components
model = None
process = psutil.Process()


def get_memory_usage():
    """Get current memory usage (MB)"""
    return process.memory_info().rss / 1024 / 1024


def memory_safe(func):
    """Memory safety decorator"""

    def wrapper(*args, **kwargs):
        try:
            if get_memory_usage() > MAX_MEMORY_MB * 0.8:
                gc.collect()
                if get_memory_usage() > MAX_MEMORY_MB:
                    raise MemoryError(f"Memory usage exceeds threshold: {get_memory_usage():.1f}MB")
            return func(*args, **kwargs)
        except Exception as e:
            raise

    return wrapper



@memory_safe
def load_inference_model():
    """Model loading with memory caching"""
    global model
    if model is None:
        try:
            from GUI_CODE.projectGUI.static.DLFile.diffModel.model import create_sunset_model
            model = create_sunset_model()
            model.load_weights(WEIGHTS_PATH)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss='mse',
                metrics=['mae']
            )
            print(f"Model loaded, memory usage: {get_memory_usage():.1f}MB")
        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {str(e)}")
    return model


def validate_hdf5_structure(filepath):
    """Enhanced HDF5 structure validation"""
    required = {
        'test': {
            'images_log': (64, 64, 3),
            'pv_log': ()
        }
    }
    try:
        with h5py.File(filepath, 'r') as f:
            for group, datasets in required.items():
                if group not in f:
                    return False, f"Missing required group: {group}"
                for ds, shape in datasets.items():
                    if ds not in f[group]:
                        return False, f"Group '{group}' missing dataset: {ds}"
                    if ds == 'images_log' and f[group][ds].shape[1:] != shape:
                        return False, f"Image shape mismatch: expected {shape}, got {f[group][ds].shape[1:]}"
        return True, ""
    except Exception as e:
        return False, str(e)


def safe_load_timestamps(npy_path):
    """Safely load and convert time data"""
    data = np.load(npy_path, allow_pickle=True)

    # Automatic type detection
    if data.dtype == object:
        # Handle datetime objects
        return np.array([dt.timestamp() if isinstance(dt, datetime) else dt for dt in data])
    elif np.issubdtype(data.dtype, np.integer):
        # Handle Unix timestamps (seconds)
        return data.astype('float64')
    elif np.issubdtype(data.dtype, np.floating):
        # Handle floating point timestamps
        return data
    else:
        raise ValueError(f"Unknown time data type: {data.dtype}")



def convert_to_dates(timestamps):
    """Date conversion with type checking"""
    dates = []
    for ts in timestamps:
        if isinstance(ts, (int, float)):
            dates.append(datetime.fromtimestamp(ts).date())
        elif isinstance(ts, datetime):
            dates.append(ts.date())
        else:
            raise TypeError(f"Invalid time data type: {type(ts)}")
    return np.array(dates)


def generate_predict_indices(total_samples):
    """Optimized systematic sampling method, keeping indices ordered"""
    sample_ratio = 0.2
    step = max(1, int(round(1 / sample_ratio)))  # Keep step calculation method

    # Generate equidistant indices and enforce sorting
    start = np.random.randint(0, step)
    indices = np.arange(start, total_samples, step, dtype=np.int64)
    indices = np.sort(indices[indices < total_samples])  # Added sorting step

    # Enhanced validation logic
    assert np.all(np.diff(indices) > 0), "Indices must be strictly increasing"
    assert indices[-1] < total_samples, f"Max index out of bounds: {indices[-1]} >= {total_samples}"
    assert indices.dtype == np.int64, f"Index type error: {indices.dtype}"

    return indices


@bp.route('/predict', methods=['POST'])
@memory_safe
def predict():
    mem_log = []
    temp_dir = None

    # Configuration constants
    BATCH_SIZE = 32  # Further reduce batch size
    MEMORY_SAFE_FACTOR = 0.6  # Memory safety threshold

    try:
        # === File reception and validation ===
        if 'file' not in request.files or 'times_test' not in request.files:
            return jsonify({"error": "Both dataset and time files must be uploaded"}), 400

        # === Create temporary session directory ===
        session_id = str(uuid.uuid4())[:8]
        temp_dir = os.path.join("static/temp", session_id)
        os.makedirs(temp_dir, exist_ok=True)

        # Save uploaded files
        h5_path = os.path.join(temp_dir, "data.h5")
        npy_path = os.path.join(temp_dir, "times.npy")
        request.files['file'].save(h5_path)
        request.files['times_test'].save(npy_path)

        # === File validation ===
        is_valid, msg = validate_hdf5_structure(h5_path)
        if not is_valid:
            return jsonify({"error": "Invalid HDF5 file", "detail": msg}), 400

        # === Load time data ===
        try:
            timestamps = safe_load_timestamps(npy_path)
            dates_test = np.array([datetime.fromtimestamp(ts).date() for ts in timestamps])
        except Exception as e:
            return jsonify({"error": "Time data error", "detail": str(e)}), 400

        # === Memory-optimized data loading ===
        model = load_inference_model()
        predictions = []
        ground_truth = []

        with h5py.File(h5_path, 'r') as f:
            test_group = f['test']
            total_samples = test_group['images_log'].shape[0]
            selected_indices = generate_predict_indices(total_samples)
            selected_indices = np.random.choice(
                selected_indices,
                int(len(selected_indices) * SAMPLE_RATIO),
                replace=False
            )
            selected_indices.sort()

            # Batch processing loop
            for i in range(0, len(selected_indices), BATCH_SIZE):
                batch_indices = selected_indices[i:i + BATCH_SIZE]
                batch_indices.sort()

                # Memory safety check
                current_mem = get_memory_usage()
                if current_mem > MAX_MEMORY_MB * MEMORY_SAFE_FACTOR:
                    raise MemoryError(f"Memory warning: {current_mem:.1f}MB > {MAX_MEMORY_MB * MEMORY_SAFE_FACTOR:.1f}MB")

                # Load and convert data
                batch_images = test_group['images_log'][batch_indices].astype('float16') / 255.0  # Use float16
                batch_pv = test_group['pv_log'][batch_indices].astype('float16')

                # Predict and collect results
                with tf.device('/cpu:0'):  # Force CPU usage
                    batch_pred = model.predict(batch_images, verbose=0, batch_size=BATCH_SIZE).flatten()

                predictions.extend(batch_pred.astype('float32').tolist())  # Convert back to float32 for precision
                ground_truth.extend(batch_pv.astype('float32').tolist())

                # Force memory release
                del batch_images, batch_pv, batch_pred
                tf.keras.backend.clear_session()  # Critical! Clean TensorFlow computation graph
                gc.collect()

                # Record memory usage
                mem_log.append(get_memory_usage())

        # === Metric calculation ===
        metrics = {
            "processed_samples": len(predictions),
            "rmse": np.sqrt(mean_squared_error(ground_truth, predictions)),
            "mae": mean_absolute_error(ground_truth, predictions),
            "r2": r2_score(ground_truth, predictions),
            "evs": explained_variance_score(ground_truth, predictions),
            "prediction_range": {
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions))
            }
        }

        # === Visualization generation ===
        vis_urls = generate_gradcam_visualization(
            model=model,
            h5_path=h5_path,
            dates_test=dates_test,
            session_id=session_id
        )

        return jsonify({
            "status": "success",
            "metrics": metrics,
            "visualization": vis_urls,
            "memory_usage": {
                "max": max(mem_log),
                "average": sum(mem_log) / len(mem_log)
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Processing failed",
            "detail": str(e),
            "memory_log": mem_log[-10:] if mem_log else []
        }), 500

    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        tf.keras.backend.clear_session()  # Ensure TensorFlow resources are cleaned


# Predefined sunny/cloudy date lists
SUNNY_DATES = [
    date(2017,9,15), date(2017,10,6), date(2017,10,22),
    date(2018,2,16), date(2018,6,12), date(2018,6,23),
    date(2019,1,25), date(2019,6,23), date(2019,7,14), date(2019,10,14)
]
CLOUDY_DATES = [
    date(2017,6,24), date(2017,9,20), date(2017,10,11),
    date(2018,1,25), date(2018,3,9), date(2018,10,4),
    date(2019,5,27), date(2019,6,28), date(2019,8,10), date(2019,10,19)
]


def safe_gradcam_heatmap(model, img_array, layer_name):
    """Robust heatmap generation function"""
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = np.maximum(heatmap, 0)

        max_val = np.max(heatmap)
        return heatmap / max_val if max_val != 0 else heatmap

    except Exception as e:
        print(f"GradCAM generation failed: {str(e)}")
        return np.zeros(img_array.shape[1:-1])


def generate_gradcam_visualization(model, h5_path, dates_test, session_id):
    """Generate heatmaps to fix blank image issues"""
    vis_dir = os.path.join("static/visualizations", session_id)
    os.makedirs(vis_dir, exist_ok=True)
    vis_path = os.path.join(vis_dir, "gradcam_summary.png")

    try:
        # === Force non-interactive backend ===
        plt.switch_backend('Agg')
        plt.ioff()
        tf.keras.backend.clear_session()

        # === Data loading and preprocessing ===
        with h5py.File(h5_path, 'r') as f:
            # Load all data (no sampling for small-scale tests)
            images = f['test/images_log'][:].astype('float32') / 255.0  # Ensure float32
            pv = f['test/pv_log'][:]
            sunny_mask = np.array([d in SUNNY_DATES for d in dates_test], dtype=bool)

            # Split dataset
            sunny_images = images[sunny_mask]
            sunny_pv = pv[sunny_mask]
            cloudy_images = images[~sunny_mask]
            cloudy_pv = pv[~sunny_mask]

        # === Model prediction ===
        sunny_pred = model.predict(sunny_images, batch_size=32).flatten()
        cloudy_pred = model.predict(cloudy_images, batch_size=32).flatten()

        # === Sample selection ===
        def select_samples(images, true, pred):
            errors = np.abs(pred - true)
            return {
                'best': images[np.argmin(errors)],
                'median': images[np.argsort(errors)[len(errors) // 2]],
                'worst': images[np.argmax(errors)],
                'random': images[np.random.choice(len(images))]
            }, {
                'best': (true[np.argmin(errors)], pred[np.argmin(errors)]),
                'median': (true[len(errors) // 2], pred[len(errors) // 2]),
                'worst': (true[np.argmax(errors)], pred[np.argmax(errors)]),
                'random': (true[-1], pred[-1])
            }

        sunny_samples, sunny_labels = select_samples(sunny_images, sunny_pv, sunny_pred)
        cloudy_samples, cloudy_labels = select_samples(cloudy_images, cloudy_pv, cloudy_pred)

        # === Visualization generation ===
        fig = plt.figure(figsize=(18, 22), dpi=100)

        # Color mapping configuration
        norm = plt.Normalize(vmin=0, vmax=1)
        cmap = plt.get_cmap("jet")

        for row, stype in enumerate(['best', 'median', 'worst', 'random']):
            # Sunny samples
            img = sunny_samples[stype]
            true, pred = sunny_labels[stype]

            # Generate heatmap (add debug output)
            heatmap = safe_gradcam_heatmap(model, np.expand_dims(img, 0), "conv2d")
            print(f"Sunny sample {stype} heatmap range: {np.min(heatmap):.2f}-{np.max(heatmap):.2f}")  # Debug

            ax = fig.add_subplot(4, 2, 2 * row + 1)
            ax.imshow(img)  # Original image
            ax.imshow(heatmap, alpha=0.5, cmap=cmap, norm=norm)
            ax.set_title(f"Sunny {stype}\nTrue: {true:.2f} | Pred: {pred:.2f}")
            ax.axis('off')

            # Cloudy samples
            img = cloudy_samples[stype]
            true, pred = cloudy_labels[stype]

            heatmap = safe_gradcam_heatmap(model, np.expand_dims(img, 0), "conv2d")
            print(f"Cloudy sample {stype} heatmap range: {np.min(heatmap):.2f}-{np.max(heatmap):.2f}")  # Debug

            ax = fig.add_subplot(4, 2, 2 * row + 2)
            ax.imshow(img)
            ax.imshow(heatmap, alpha=0.5, cmap=cmap, norm=norm)
            ax.set_title(f"Cloudy {stype}\nTrue: {true:.2f} | Pred: {pred:.2f}")
            ax.axis('off')

        # === Save and validate ===
        plt.tight_layout()
        plt.savefig(vis_path, bbox_inches='tight', dpi=100)

        # Verify image validity
        if os.path.getsize(vis_path) < 1024:  # File too small indicates save failure
            raise RuntimeError("Generated image file abnormal")

        plt.close()
        return [f"/static/visualizations/{session_id}/gradcam_summary.png"]

    except Exception as e:
        print(f"Visualization generation failed: {str(e)}")
        traceback.print_exc()  # Print full stack trace
        return []
    finally:
        plt.close('all')
        tf.keras.backend.clear_session()
        gc.collect()


# Supporting utility functions
def date_converter(timestamp):
    """Unified date format conversion"""
    if isinstance(timestamp, np.datetime64):
        return timestamp.astype('datetime64[D]').item().date()
    elif isinstance(timestamp, str):
        return datetime.strptime(timestamp, "%Y-%m-%d").date()
    return timestamp.date()


@bp.route('/dataDisplay')
def dataDisplay():
    return render_template("dataDisplay.html")


@bp.route('/predictModel')
def predictModel():
    return render_template("predictModel.html")


@bp.route('/model-layers')
def get_model_layers():
    model = load_inference_model()
    return jsonify([layer.name for layer in model.layers if 'conv' in layer.name])


