#!/usr/bin/env python3
"""
float32を維持しながらファイルサイズを小さくする方法の実験
"""

import torch
import numpy as np
import sys
import os
import onnx
from onnx import helper, TensorProto
import gzip


def analyze_compression_options(pth_path):
    """異なる圧縮・最適化オプションの効果を分析"""
    
    print(f"=== Float32でのサイズ削減オプション分析 ===")
    print(f"元ファイル: {pth_path}")
    
    checkpoint = torch.load(pth_path, map_location='cpu')
    
    # 1. 量子化（重みの範囲を調べる）
    print("\n【1. 重みの分布分析】")
    
    weight_stats = {}
    total_params = 0
    
    for name, tensor in checkpoint.items():
        if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float16:
            flat_weights = tensor.float().flatten()
            total_params += len(flat_weights)
            
            stats = {
                'min': float(flat_weights.min()),
                'max': float(flat_weights.max()),
                'mean': float(flat_weights.mean()),
                'std': float(flat_weights.std()),
                'zeros': int((flat_weights == 0).sum()),
                'small_values': int((flat_weights.abs() < 0.01).sum())
            }
            
            if len(weight_stats) < 5:  # 最初の5つだけ表示
                print(f"{name[:40]:40s} | Min: {stats['min']:8.4f} | Max: {stats['max']:8.4f} | Std: {stats['std']:6.4f}")
                weight_stats[name] = stats
    
    # 2. スパース性の分析
    print(f"\n【2. スパース性（ゼロ値）分析】")
    total_zeros = sum(stats['zeros'] for stats in weight_stats.values())
    total_small = sum(stats['small_values'] for stats in weight_stats.values())
    
    print(f"総パラメータ数: {total_params:,}")
    print(f"完全ゼロ値: {total_zeros:,} ({total_zeros/total_params*100:.1f}%)")
    print(f"小さい値(<0.01): {total_small:,} ({total_small/total_params*100:.1f}%)")
    
    return checkpoint


def create_pruned_onnx(checkpoint, output_path, prune_threshold=0.01):
    """小さい重みを削除（プルーニング）してONNX作成"""
    
    print(f"\n【プルーニング変換 (閾値: {prune_threshold})】")
    
    inputs = []
    outputs = []
    nodes = []
    initializers = []
    
    # 入力・出力定義
    input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 12, 128, 256])
    output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1000])
    inputs.append(input_info)
    outputs.append(output_info)
    
    # 重みをプルーニングして変換
    pruned_params = 0
    total_params = 0
    
    for name, tensor in checkpoint.items():
        if isinstance(tensor, torch.Tensor):
            try:
                onnx_name = name.replace('_initializer_', '').replace('__', '_')
                
                if tensor.dtype == torch.float16:
                    numpy_tensor = tensor.float().numpy()
                    
                    # プルーニング適用
                    mask = np.abs(numpy_tensor) >= prune_threshold
                    pruned_tensor = numpy_tensor * mask
                    
                    pruned_count = np.sum(~mask)
                    pruned_params += pruned_count
                    total_params += numpy_tensor.size
                    
                else:
                    pruned_tensor = tensor.numpy()
                
                onnx_tensor = helper.make_tensor(
                    name=onnx_name,
                    data_type=TensorProto.FLOAT,
                    dims=list(tensor.shape),
                    vals=pruned_tensor.flatten().astype(np.float32)
                )
                
                initializers.append(onnx_tensor)
                
            except Exception as e:
                print(f"Warning: {name}: {e}")
    
    print(f"プルーニング結果: {pruned_params:,}/{total_params:,} ({pruned_params/total_params*100:.1f}%) 削除")
    
    # 簡単なグラフ
    identity_node = helper.make_node('Identity', inputs=['input'], outputs=['output'])
    nodes.append(identity_node)
    
    graph = helper.make_graph(nodes, 'pruned_model', inputs, outputs, initializers)
    model = helper.make_model(graph, producer_name='pruned-converter')
    model.opset_import[0].version = 11
    
    onnx.save(model, output_path)
    return os.path.getsize(output_path) / (1024 * 1024)


def create_quantized_onnx(checkpoint, output_path, bits=8):
    """疑似量子化でサイズ削減"""
    
    print(f"\n【疑似量子化変換 ({bits}bit相当)】")
    
    inputs = []
    outputs = []
    nodes = []
    initializers = []
    
    input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 12, 128, 256])
    output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1000])
    inputs.append(input_info)
    outputs.append(output_info)
    
    quantized_size_reduction = 0
    
    for name, tensor in checkpoint.items():
        if isinstance(tensor, torch.Tensor):
            try:
                onnx_name = name.replace('_initializer_', '').replace('__', '_')
                
                if tensor.dtype == torch.float16:
                    numpy_tensor = tensor.float().numpy()
                    
                    # 疑似量子化（値の範囲を制限）
                    min_val, max_val = numpy_tensor.min(), numpy_tensor.max()
                    scale = (max_val - min_val) / (2**bits - 1)
                    
                    quantized = np.round((numpy_tensor - min_val) / scale) * scale + min_val
                    quantized_tensor = quantized.astype(np.float32)
                    
                else:
                    quantized_tensor = tensor.numpy().astype(np.float32)
                
                onnx_tensor = helper.make_tensor(
                    name=onnx_name,
                    data_type=TensorProto.FLOAT,
                    dims=list(tensor.shape),
                    vals=quantized_tensor.flatten()
                )
                
                initializers.append(onnx_tensor)
                
            except Exception as e:
                print(f"Warning: {name}: {e}")
    
    identity_node = helper.make_node('Identity', inputs=['input'], outputs=['output'])
    nodes.append(identity_node)
    
    graph = helper.make_graph(nodes, 'quantized_model', inputs, outputs, initializers)
    model = helper.make_model(graph, producer_name='quantized-converter')
    model.opset_import[0].version = 11
    
    onnx.save(model, output_path)
    return os.path.getsize(output_path) / (1024 * 1024)


def test_compression_methods(pth_path):
    """各種圧縮手法をテストして結果を比較"""
    
    print(f"=== 各種サイズ削減手法のテスト ===")
    
    # 元ファイル分析
    checkpoint = analyze_compression_options(pth_path)
    
    base_dir = os.path.dirname(pth_path)
    base_name = os.path.basename(pth_path).replace('.pth', '')
    
    results = {}
    
    # 1. 通常のfloat32変換（既存）
    original_onnx = os.path.join(base_dir, f'{base_name}_proper.onnx')
    if os.path.exists(original_onnx):
        results['Float32 (通常)'] = os.path.getsize(original_onnx) / (1024 * 1024)
    
    # 2. プルーニング版
    try:
        pruned_path = os.path.join(base_dir, f'{base_name}_pruned.onnx')
        results['Float32 + プルーニング'] = create_pruned_onnx(checkpoint, pruned_path)
    except Exception as e:
        print(f"プルーニング失敗: {e}")
    
    # 3. 量子化版
    try:
        quantized_path = os.path.join(base_dir, f'{base_name}_quantized.onnx')
        results['Float32 + 疑似量子化'] = create_quantized_onnx(checkpoint, quantized_path)
    except Exception as e:
        print(f"量子化失敗: {e}")
    
    # 4. 圧縮版（gzip）
    if 'Float32 (通常)' in results:
        try:
            compressed_path = original_onnx + '.gz'
            with open(original_onnx, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.writelines(f_in)
            results['Float32 + Gzip圧縮'] = os.path.getsize(compressed_path) / (1024 * 1024)
        except Exception as e:
            print(f"Gzip圧縮失敗: {e}")
    
    # 結果表示
    print(f"\n=== サイズ比較結果 ===")
    original_pytorch = os.path.getsize(pth_path) / (1024 * 1024)
    print(f"{'手法':25s} | {'サイズ':>8s} | {'圧縮比':>6s} | {'影響':s}")
    print("-" * 65)
    print(f"{'元のPyTorch (float16)':25s} | {original_pytorch:7.1f}MB | {'1.0x':>6s} | 基準")
    
    for method, size_mb in results.items():
        ratio = size_mb / original_pytorch
        if 'プルーニング' in method:
            impact = "精度低下あり"
        elif '量子化' in method:
            impact = "わずかな精度低下"
        elif 'Gzip' in method:
            impact = "展開が必要"
        else:
            impact = "精度保持"
        
        print(f"{method:25s} | {size_mb:7.1f}MB | {ratio:5.1f}x | {impact}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python size_reduction_test.py <input.pth>")
        sys.exit(1)
    
    pth_path = sys.argv[1]
    test_compression_methods(pth_path)