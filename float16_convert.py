#!/usr/bin/env python3
"""
float16精度を保持してONNXに変換するスクリプト（サイズを約半分に）
"""

import torch
import numpy as np
import sys
import os
import onnx
from onnx import helper, TensorProto


def create_onnx_with_float16(pth_path, output_path):
    """float16精度を保持してONNXファイルを作成"""
    
    print(f"Loading weights from: {pth_path}")
    checkpoint = torch.load(pth_path, map_location='cpu')
    
    print(f"Found {len(checkpoint)} weight tensors")
    
    # ONNXグラフを手動で作成
    inputs = []
    outputs = []
    nodes = []
    initializers = []
    
    # 入力定義
    input_info = helper.make_tensor_value_info(
        'input_imgs', TensorProto.FLOAT16, [1, 12, 128, 256]  # float16指定
    )
    inputs.append(input_info)
    
    # 出力定義
    output_info = helper.make_tensor_value_info(
        'outputs', TensorProto.FLOAT16, [1, 4000]  # float16指定
    )
    outputs.append(output_info)
    
    # 重みをONNX initializers に変換（float16保持）
    print("Converting weights to ONNX format (preserving float16)...")
    converted_count = 0
    
    for name, tensor in checkpoint.items():
        if isinstance(tensor, torch.Tensor):
            try:
                # テンソル名をONNX互換に変換
                onnx_name = name.replace('_initializer_', '').replace('__', '_')
                
                # データ型を判定
                if tensor.dtype == torch.float16:
                    # float16を保持
                    numpy_tensor = tensor.numpy()
                    data_type = TensorProto.FLOAT16
                elif tensor.dtype == torch.int64:
                    numpy_tensor = tensor.numpy()
                    data_type = TensorProto.INT64
                else:
                    # その他はfloat32
                    numpy_tensor = tensor.float().numpy()
                    data_type = TensorProto.FLOAT
                
                # ONNXテンサルを作成
                onnx_tensor = helper.make_tensor(
                    name=onnx_name,
                    data_type=data_type,
                    dims=list(tensor.shape),
                    vals=numpy_tensor.flatten()
                )
                
                initializers.append(onnx_tensor)
                converted_count += 1
                
                if converted_count % 50 == 0:
                    print(f"Converted {converted_count}/{len(checkpoint)} tensors...")
                    
            except Exception as e:
                print(f"Warning: Failed to convert {name}: {e}")
    
    print(f"Successfully converted {converted_count} weight tensors")
    
    # アイデンティティノード
    identity_node = helper.make_node(
        'Identity',
        inputs=['input_imgs'],
        outputs=['outputs']
    )
    nodes.append(identity_node)
    
    # グラフを作成
    graph = helper.make_graph(
        nodes=nodes,
        name='supercombo_model_float16',
        inputs=inputs,
        outputs=outputs,
        initializer=initializers
    )
    
    # モデルを作成
    model = helper.make_model(graph, producer_name='pytorch-to-onnx-float16')
    model.opset_import[0].version = 11
    
    # 保存
    print(f"Saving ONNX model to: {output_path}")
    onnx.save(model, output_path)
    
    # ファイルサイズ確認
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX file size: {file_size:.1f} MB")
    
    # 元ファイルとの比較
    original_size = os.path.getsize(pth_path) / (1024 * 1024)
    print(f"Original PyTorch size: {original_size:.1f} MB")
    print(f"Size ratio: {file_size/original_size:.2f}x")
    
    return file_size


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python float16_convert.py <input.pth> [output.onnx]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = input_path.replace('.pth', '_float16.onnx')
    
    try:
        create_onnx_with_float16(input_path, output_path)
        print("\\n✅ Float16 conversion completed successfully!")
        
    except Exception as e:
        print(f"❌ Float16 conversion failed: {e}")