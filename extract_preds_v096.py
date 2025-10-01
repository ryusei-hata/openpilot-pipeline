#!/usr/bin/env python3

import numpy as np

def sigmoid(x):
    """シグモイド関数 (オーバーフロー対策付き)"""
    return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def extract_preds_v096(outputs, best_plan_only=True):
    """v0.9.6 supercombo model用の予測抽出関数"""
    # v0.9.6モデルの出力サイズ: 6504
    
    # インデックス定義 (v0.9.6用に調整)
    plan_start_idx = 0
    plan_end_idx = 4955

    lanes_start_idx = plan_end_idx
    lanes_end_idx = lanes_start_idx + 528

    lane_lines_prob_start_idx = lanes_end_idx
    lane_lines_prob_end_idx = lane_lines_prob_start_idx + 8

    road_start_idx = lane_lines_prob_end_idx
    road_end_idx = road_start_idx + 264
    
    # 新しいセクション（749要素）- おそらく追加機能
    additional_start_idx = road_end_idx
    additional_end_idx = additional_start_idx + 749  # 6504 - 5755
    
    # Debug info (uncomment if needed)
    # print(f"Output shape: {outputs.shape}")
    # print(f"Plan: {plan_start_idx}-{plan_end_idx} ({plan_end_idx-plan_start_idx} elements)")

    # plan
    plan = outputs[:, plan_start_idx:plan_end_idx]  # (N, 4955)
    plans = plan.reshape((-1, 5, 991))  # (N, 5, 991)
    plan_probs = plans[:, :, -1]  # (N, 5)
    plans = plans[:, :, :-1].reshape(-1, 5, 2, 33, 15)  # (N, 5, 2, 33, 15)
    
    # バッチサイズ取得
    batch_size = plans.shape[0]
    best_plan_idx = np.argmax(plan_probs, axis=1)  # (N,)
    
    # lane lines
    lane_lines = outputs[:, lanes_start_idx:lanes_end_idx]  # (N, 528)
    lane_lines_deflat = lane_lines.reshape((-1, 2, 264))  # (N, 2, 264)
    lane_lines_means = lane_lines_deflat[:, 0, :]  # (N, 264)
    lane_lines_means = lane_lines_means.reshape(-1, 4, 33, 2)  # (N, 4, 33, 2)

    outer_left_lane = lane_lines_means[:, 0, :, :]  # (N, 33, 2)
    inner_left_lane = lane_lines_means[:, 1, :, :]  # (N, 33, 2)
    inner_right_lane = lane_lines_means[:, 2, :, :]  # (N, 33, 2)
    outer_right_lane = lane_lines_means[:, 3, :, :]  # (N, 33, 2)

    # lane lines probs
    lane_lines_probs = outputs[:, lane_lines_prob_start_idx:lane_lines_prob_end_idx]  # (N, 8)
    lane_lines_probs = lane_lines_probs.reshape((-1, 4, 2))  # (N, 4, 2)
    lane_lines_probs = sigmoid(lane_lines_probs[:, :, 1])  # (N, 4), 0th is deprecated

    outer_left_prob = lane_lines_probs[:, 0]  # (N,)
    inner_left_prob = lane_lines_probs[:, 1]  # (N,)
    inner_right_prob = lane_lines_probs[:, 2]  # (N,)
    outer_right_prob = lane_lines_probs[:, 3]  # (N,)

    # road edges
    road_edges = outputs[:, road_start_idx:road_end_idx]
    road_edges_deflat = road_edges.reshape((-1, 2, 132))  # (N, 2, 132)
    road_edge_means = road_edges_deflat[:, 0, :].reshape(-1, 2, 33, 2)  # (N, 2, 33, 2)
    road_edge_stds = road_edges_deflat[:, 1, :].reshape(-1, 2, 33, 2)  # (N, 2, 33, 2)

    left_edge = road_edge_means[:, 0, :, :]  # (N, 33, 2)
    right_edge = road_edge_means[:, 1, :, :]
    left_edge_std = road_edge_stds[:, 0, :, :]  # (N, 33, 2)
    right_edge_std = road_edge_stds[:, 1, :, :]

    result_batch = []

    for i in range(batch_size):
        lanelines = [outer_left_lane[i], inner_left_lane[i], inner_right_lane[i], outer_right_lane[i]]
        lanelines_probs = [outer_left_prob[i], inner_left_prob[i], inner_right_prob[i], outer_right_prob[i]]
        road_edges = [left_edge[i], right_edge[i]]
        road_edges_probs = [left_edge_std[i], right_edge_std[i]]

        if best_plan_only:
            plan = plans[i, best_plan_idx[i], ...]  # (2, 33, 15)
        else:
            plan = (plans[i], plan_probs[i])

        result_batch.append(((lanelines, lanelines_probs), (road_edges, road_edges_probs), plan))

    return result_batch


# テスト関数
def test_extract_preds_v096():
    """新しい抽出関数をテスト"""
    # ダミーデータで形状を確認
    dummy_output = np.random.randn(1, 6504).astype(np.float16)
    
    print("Testing extract_preds_v096...")
    results = extract_preds_v096(dummy_output, best_plan_only=False)
    
    (lane_lines, lane_lines_probs), (road_edges, road_edges_stds), (plans, plans_prob) = results[0]
    
    print(f"Lane lines: {len(lane_lines)} lines, each shape: {lane_lines[0].shape}")
    print(f"Lane probs: {len(lane_lines_probs)} probs, values: {[p for p in lane_lines_probs]}")
    print(f"Road edges: {len(road_edges)} edges, each shape: {road_edges[0].shape}")
    print(f"Road stds: {len(road_edges_stds)} stds, each shape: {road_edges_stds[0].shape}")
    print(f"Plans: shape {plans.shape}")
    print(f"Plans prob: shape {plans_prob.shape}")
    
    return True

if __name__ == "__main__":
    test_extract_preds_v096()