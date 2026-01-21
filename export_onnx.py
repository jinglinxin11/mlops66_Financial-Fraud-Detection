"""
ONNX Model Export Script
Converts TabNet model to ONNX format for faster inference

Usage:
    python export_onnx.py
    python export_onnx.py --model-path tabnet_fraud_model.zip --output model.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.config.settings import Config
from src.features.preprocessor import FraudPreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def export_to_onnx(
    model_path: str = None, output_path: str = "tabnet_fraud_model.onnx", verify: bool = True
):
    """
    Export TabNet model to ONNX format.

    Args:
        model_path: Path to trained TabNet model (.zip)
        output_path: Output path for ONNX model
        verify: Whether to verify the exported model
    """
    logger.info("=" * 60)
    logger.info("     ONNX Model Export")
    logger.info("=" * 60)

    config = Config()

    # Default model path
    if model_path is None:
        model_path = str(config.MODEL_PATH) + ".zip"

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info(f"Loading model from: {model_path}")

    # Load TabNet model
    from pytorch_tabnet.tab_model import TabNetClassifier

    model = TabNetClassifier()
    model.load_model(str(model_path))

    # Get input dimension from the model
    # TabNet stores network in model.network
    network = model.network
    input_dim = model.input_dim

    logger.info(f"Model input dimension: {input_dim}")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load preprocessor to identify categorical features
    preprocessor = FraudPreprocessor(config)
    preprocessor.load()

    # Construct dummy input with valid categorical indices
    cat_idxs = model.cat_idxs
    cat_dims = model.cat_dims

    # Start with random float features
    dummy_data = torch.randn(1, input_dim, dtype=torch.float32).to(device)

    # Replace categorical columns with valid integers
    for _, (idx, dim) in enumerate(zip(cat_idxs, cat_dims)):
        # Generate valid index (0 to dim-1)
        # Note: TabNet expects float input even for cats (will be cast inside)
        dummy_data[0, idx] = torch.randint(0, dim, (1,), dtype=torch.float32).item()

    dummy_input = dummy_data

    # Ensure network is on the correct device
    network.to(device)
    network.eval()

    # Export to ONNX
    logger.info(f"Exporting to ONNX: {output_path}")

    torch.onnx.export(
        network,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output", "attention"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
            "attention": {0: "batch_size"},
        },
    )

    logger.info(f"✅ Model exported to: {output_path}")

    # Verify the model
    if verify:
        logger.info("\nVerifying ONNX model...")

        # Check model is well-formed
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✅ ONNX model is valid")

        # Test inference with ONNX Runtime
        ort_session = ort.InferenceSession(output_path)

        # Run inference
        test_input = np.random.randn(5, input_dim).astype(np.float32)
        ort_inputs = {ort_session.get_inputs()[0].name: test_input}
        ort_outputs = ort_session.run(None, ort_inputs)

        logger.info("✅ ONNX Runtime inference successful")
        logger.info(f"   Input shape: {test_input.shape}")
        logger.info(f"   Output shape: {ort_outputs[0].shape}")

    # Print model info
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"\nModel file size: {file_size:.2f} MB")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export TabNet model to ONNX")
    parser.add_argument("--model-path", type=str, default=None, help="Path to TabNet model (.zip)")
    parser.add_argument(
        "--output", type=str, default="tabnet_fraud_model.onnx", help="Output ONNX file path"
    )
    parser.add_argument("--no-verify", action="store_true", help="Skip verification")
    args = parser.parse_args()

    export_to_onnx(model_path=args.model_path, output_path=args.output, verify=not args.no_verify)


if __name__ == "__main__":
    main()
