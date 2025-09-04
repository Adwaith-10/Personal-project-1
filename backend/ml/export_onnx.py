#!/usr/bin/env python3
"""
Food Vision Pro - ONNX Model Export Script

This script converts trained PyTorch models to ONNX format for deployment.
It handles different model architectures and ensures compatibility with ONNX runtime.

Usage:
    python export_onnx.py --model_path /path/to/model.pth --output_dir ./onnx_models
"""

import os
import argparse
import json
import torch
import torch.nn as nn
from torchvision import models
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_model_for_export(num_classes, model_name='efficientnet_b0'):
    """Create model architecture for export (without pretrained weights)"""
    
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=False)
        
        # Modify classifier for our number of classes
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(512, num_classes)
        )
        
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=False)
        
        # Modify classifier
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model


def load_trained_model(checkpoint_path, model_name='efficientnet_b0'):
    """Load trained model from checkpoint"""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get number of classes from class index if available
    class_index_path = os.path.join(os.path.dirname(checkpoint_path), 'class_index.json')
    if os.path.exists(class_index_path):
        with open(class_index_path, 'r') as f:
            class_data = json.load(f)
            num_classes = class_data.get('num_classes', 101)  # Default to Food-101
    else:
        # Try to infer from checkpoint
        if 'model_state_dict' in checkpoint:
            # Count the number of output classes from the last layer
            state_dict = checkpoint['model_state_dict']
            if model_name == 'efficientnet_b0':
                last_layer_key = 'classifier.3.weight'
            else:  # resnet50
                last_layer_key = 'fc.3.weight'
            
            if last_layer_key in state_dict:
                num_classes = state_dict[last_layer_key].shape[0]
            else:
                num_classes = 101  # Default
        else:
            num_classes = 101  # Default
    
    # Create model
    model = create_model_for_export(num_classes, model_name)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model state dict from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        # Assume the checkpoint is just the state dict
        model.load_state_dict(checkpoint)
        logger.info("Loaded model state dict directly")
    
    return model


def export_to_onnx(model, output_path, input_shape=(1, 3, 224, 224), opset_version=11):
    """Export PyTorch model to ONNX format"""
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    logger.info(f"Model exported to ONNX: {output_path}")


def validate_onnx_model(onnx_path, pytorch_model, test_input):
    """Validate ONNX model against PyTorch model"""
    
    try:
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validation passed")
        
        # Test inference with ONNX Runtime
        ort_session = ort.InferenceSession(onnx_path)
        
        # Get input/output names
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        # Prepare input
        ort_inputs = {input_name: test_input.numpy()}
        
        # Run inference
        ort_outputs = ort_session.run([output_name], ort_inputs)
        onnx_output = ort_outputs[0]
        
        # Run inference with PyTorch
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input)
            pytorch_output = pytorch_output.numpy()
        
        # Compare outputs
        diff = np.abs(onnx_output - pytorch_output).max()
        logger.info(f"Max difference between ONNX and PyTorch outputs: {diff:.6f}")
        
        if diff < 1e-5:
            logger.info("✅ ONNX model validation successful - outputs match PyTorch")
            return True
        else:
            logger.warning("⚠️ ONNX model validation failed - outputs differ significantly")
            return False
            
    except Exception as e:
        logger.error(f"❌ ONNX model validation failed: {e}")
        return False


def optimize_onnx_model(onnx_path, optimized_path):
    """Optimize ONNX model for inference"""
    
    try:
        import onnxoptimizer
        
        # Load model
        model = onnx.load(onnx_path)
        
        # Apply optimizations
        optimized_model = onnxoptimizer.optimize(model)
        
        # Save optimized model
        onnx.save(optimized_model, optimized_path)
        
        logger.info(f"ONNX model optimized and saved to: {optimized_path}")
        
        # Compare model sizes
        original_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
        optimized_size = os.path.getsize(optimized_path) / (1024 * 1024)  # MB
        
        logger.info(f"Original model size: {original_size:.2f} MB")
        logger.info(f"Optimized model size: {optimized_size:.2f} MB")
        logger.info(f"Size reduction: {((original_size - optimized_size) / original_size * 100):.1f}%")
        
    except ImportError:
        logger.warning("onnxoptimizer not available, skipping optimization")
        # Copy original file
        import shutil
        shutil.copy2(onnx_path, optimized_path)
        logger.info(f"Copied original model to: {optimized_path}")


def main():
    parser = argparse.ArgumentParser(description='Export Food Vision Pro Model to ONNX')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained PyTorch model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./onnx_models',
                       help='Output directory for ONNX models')
    parser.add_argument('--model_name', type=str, default='efficientnet_b0',
                       choices=['efficientnet_b0', 'resnet50'],
                       help='Model architecture name')
    parser.add_argument('--input_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for export')
    parser.add_argument('--opset_version', type=int, default=11,
                       help='ONNX opset version')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize ONNX model after export')
    parser.add_argument('--validate', action='store_true',
                       help='Validate ONNX model after export')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load trained model
    logger.info(f"Loading trained model from: {args.model_path}")
    model = load_trained_model(args.model_path, args.model_name)
    
    # Set model to evaluation mode
    model.eval()
    
    # Export to ONNX
    onnx_path = os.path.join(args.output_dir, f'{args.model_name}.onnx')
    input_shape = (args.batch_size, 3, args.input_size, args.input_size)
    
    logger.info("Exporting model to ONNX...")
    export_to_onnx(model, onnx_path, input_shape, args.opset_version)
    
    # Validate ONNX model
    if args.validate:
        logger.info("Validating ONNX model...")
        test_input = torch.randn(input_shape)
        validation_passed = validate_onnx_model(onnx_path, model, test_input)
        
        if not validation_passed:
            logger.error("ONNX model validation failed!")
            return
    
    # Optimize ONNX model
    if args.optimize:
        logger.info("Optimizing ONNX model...")
        optimized_path = os.path.join(args.output_dir, f'{args.model_name}_optimized.onnx')
        optimize_onnx_model(onnx_path, optimized_path)
    
    # Copy class index if available
    class_index_path = os.path.join(os.path.dirname(args.model_path), 'class_index.json')
    if os.path.exists(class_index_path):
        output_class_index = os.path.join(args.output_dir, 'class_index.json')
        import shutil
        shutil.copy2(class_index_path, output_class_index)
        logger.info(f"Copied class index to: {output_class_index}")
    
    # Create model info file
    model_info = {
        "model_name": args.model_name,
        "input_shape": input_shape,
        "opset_version": args.opset_version,
        "export_date": str(torch.datetime.now()),
        "framework": "PyTorch -> ONNX",
        "notes": "Exported for Food Vision Pro deployment"
    }
    
    info_path = os.path.join(args.output_dir, 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info(f"✅ ONNX export completed successfully!")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"ONNX model: {onnx_path}")
    
    if args.optimize:
        logger.info(f"Optimized model: {optimized_path}")


if __name__ == '__main__':
    main()
