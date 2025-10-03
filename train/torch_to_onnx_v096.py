"""
学習済みPyTorchモデルをONNX形式に変換するスクリプト
"""

import torch
import torch.onnx
import numpy as np
import argparse
import os
from distillation_models import DistillationStudent

def parse_args():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
    parser.add_argument('--checkpoint', type=str, 
                      default='./checkpoints/best_student_model.pth',
                      help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, 
                      default='./checkpoints/distilled_supercombo.onnx',
                      help='Output ONNX file path')
    parser.add_argument('--student_hidden_dim', type=int, default=512,
                      help='Hidden dimension for student model')
    return parser.parse_args()

def create_example_inputs():
    """ONNXエクスポート用のサンプル入力を作成"""
    
    batch_size = 1
    
    # 実際の学習時の形状に合わせる（distillation_models.pyの処理を参照）
    example_inputs = {
        'input_imgs': torch.randn(batch_size, 12, 128, 256, dtype=torch.float32),
        'big_input_imgs': torch.randn(batch_size, 12, 128, 256, dtype=torch.float32), 
        'desire': torch.randn(batch_size, 100, 8, dtype=torch.float32),  # (batch, 100, 8) -> flatten -> 800
        'traffic_convention': torch.randn(batch_size, 2, dtype=torch.float32),  # (batch, 2)
        'lateral_control_params': torch.randn(batch_size, 2, dtype=torch.float32),  # (batch, 2) 
        'prev_desired_curv': torch.randn(batch_size, 100, 1, dtype=torch.float32),  # (batch, 100, 1) -> flatten -> 100
        'nav_features': torch.randn(batch_size, 256, dtype=torch.float32),  # (batch, 256)
        'nav_instructions': torch.randn(batch_size, 150, dtype=torch.float32),  # (batch, 150)
        'features_buffer': torch.randn(batch_size, 99, 512, dtype=torch.float32)  # (batch, 99, 512) -> flatten -> 50688
    }
    
    return example_inputs

def load_student_model(checkpoint_path, student_hidden_dim):
    """学習済みStudentモデルを読み込み"""
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # モデルを作成
    student = DistillationStudent(hidden_dim=student_hidden_dim)
    
    # チェックポイント読み込み
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    student.load_state_dict(checkpoint['model_state_dict'])
    
    # 評価モードに設定
    student.eval()
    
    print(f"Model loaded successfully!")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"Epoch: {checkpoint['epoch']}")
    
    return student

class ONNXWrapper(torch.nn.Module):
    """ONNX変換用のラッパーモデル"""
    
    def __init__(self, student_model):
        super().__init__()
        self.student = student_model
    
    def forward(self, input_imgs, big_input_imgs, desire, traffic_convention, 
                lateral_control_params, prev_desired_curv, nav_features, 
                nav_instructions, features_buffer):
        """位置引数を辞書形式に変換してStudentモデルに渡す"""
        
        inputs = {
            'input_imgs': input_imgs,
            'big_input_imgs': big_input_imgs,
            'desire': desire,
            'traffic_convention': traffic_convention,
            'lateral_control_params': lateral_control_params,
            'prev_desired_curv': prev_desired_curv,
            'nav_features': nav_features,
            'nav_instructions': nav_instructions,
            'features_buffer': features_buffer
        }
        
        return self.student(**inputs)

def convert_to_onnx(model, example_inputs, output_path):
    """PyTorchモデルをONNXに変換"""
    
    print(f"Converting model to ONNX format...")
    
    # ラッパーモデルを作成
    wrapper = ONNXWrapper(model)
    wrapper.eval()
    
    # 入力名と出力名を定義
    input_names = list(example_inputs.keys())
    output_names = ['outputs']
    
    # 動的軸を定義（バッチサイズを可変にする）
    dynamic_axes = {}
    for name in input_names:
        dynamic_axes[name] = {0: 'batch_size'}
    dynamic_axes['outputs'] = {0: 'batch_size'}
    
    # ONNXエクスポート
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                tuple(example_inputs.values()),  # 入力をタプルに変換
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=True
            )
        
        print(f"Successfully converted to ONNX: {output_path}")
        
        # ファイルサイズを確認
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"ONNX file size: {file_size:.1f} MB")
        
    except Exception as e:
        print(f"Error during ONNX conversion: {e}")
        raise

def verify_onnx_model(onnx_path, example_inputs):
    """ONNXモデルの動作確認"""
    
    try:
        import onnxruntime as ort
        
        print("Verifying ONNX model...")
        
        # ONNXランタイムセッションを作成
        session = ort.InferenceSession(onnx_path)
        
        # 入力データをnumpy配列に変換
        input_data = {}
        for name, tensor in example_inputs.items():
            input_data[name] = tensor.numpy()
        
        # 推論実行
        outputs = session.run(None, input_data)
        
        print(f"ONNX model verification successful!")
        print(f"Output shape: {outputs[0].shape}")
        print(f"Output dtype: {outputs[0].dtype}")
        
    except ImportError:
        print("ONNXRuntime not available for verification")
    except Exception as e:
        print(f"ONNX verification error: {e}")

def main():
    args = parse_args()
    
    # 出力ディレクトリを作成
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Studentモデルを読み込み
    student = load_student_model(args.checkpoint, args.student_hidden_dim)
    
    # サンプル入力を作成
    example_inputs = create_example_inputs()
    
    # PyTorchモデルでテスト推論
    print("Testing PyTorch model...")
    with torch.no_grad():
        pytorch_output = student(**example_inputs)
        print(f"PyTorch output shape: {pytorch_output.shape}")
    
    # ONNXに変換
    convert_to_onnx(student, example_inputs, args.output)
    
    # ONNX モデルを検証
    verify_onnx_model(args.output, example_inputs)
    
    print("\nConversion completed successfully!")
    print(f"Original PyTorch model: {args.checkpoint}")
    print(f"Converted ONNX model: {args.output}")

if __name__ == '__main__':
    main()