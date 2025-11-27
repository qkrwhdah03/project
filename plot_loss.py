#!/usr/bin/env python3
"""
Loss 그래프 시각화 스크립트
train.log에서 loss 값을 추출하여 그래프로 그립니다.
"""
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def parse_loss_from_log(log_file, max_step=None):
    """train.log에서 loss 값 추출"""
    losses = []
    steps = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # "step [X] loss: Y" 형식 찾기
            match = re.search(r'step\s+\[(\d+)\]\s+loss:\s+([\d.e+-]+|nan|inf)', line, re.IGNORECASE)
            if match:
                step = int(match.group(1))
                
                # max_step이 지정되면 그 이전만 추출
                if max_step is not None and step >= max_step:
                    continue
                
                loss_str = match.group(2).lower()
                
                if loss_str in ['nan', 'inf']:
                    loss_val = np.nan if loss_str == 'nan' else np.inf
                else:
                    try:
                        loss_val = float(loss_str)
                    except:
                        continue
                
                steps.append(step)
                losses.append(loss_val)
    
    return steps, losses

def plot_loss(steps, losses, output_file='loss_plot.png'):
    """Loss 그래프 그리기"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 전체 그래프
    ax1 = axes[0]
    valid_mask = ~np.isnan(losses) & ~np.isinf(losses)
    if valid_mask.sum() > 0:
        ax1.plot(np.array(steps)[valid_mask], np.array(losses)[valid_mask], 
                'b-', linewidth=1, alpha=0.7, label='Loss')
        ax1.scatter(np.array(steps)[valid_mask], np.array(losses)[valid_mask], 
                   s=10, alpha=0.5)
    
    # NaN/Inf 표시
    nan_mask = np.isnan(losses)
    inf_mask = np.isinf(losses)
    if nan_mask.sum() > 0:
        nan_steps = np.array(steps)[nan_mask]
        ax1.scatter(nan_steps, [ax1.get_ylim()[1] * 0.9] * len(nan_steps), 
                   c='red', marker='x', s=100, label='NaN', zorder=10)
    if inf_mask.sum() > 0:
        inf_steps = np.array(steps)[inf_mask]
        ax1.scatter(inf_steps, [ax1.get_ylim()[1] * 0.95] * len(inf_steps), 
                   c='orange', marker='^', s=100, label='Inf', zorder=10)
    
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss (Full History)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 최근 부분 확대 (마지막 1000 step)
    ax2 = axes[1]
    if len(steps) > 0:
        recent_steps = steps[-1000:] if len(steps) > 1000 else steps
        recent_losses = losses[-1000:] if len(losses) > 1000 else losses
        
        recent_valid = [not (np.isnan(l) or np.isinf(l)) for l in recent_losses]
        if sum(recent_valid) > 0:
            recent_steps_valid = [s for s, v in zip(recent_steps, recent_valid) if v]
            recent_losses_valid = [l for l, v in zip(recent_losses, recent_valid) if v]
            
            ax2.plot(recent_steps_valid, recent_losses_valid, 
                    'b-', linewidth=1.5, alpha=0.8, label='Loss')
            ax2.scatter(recent_steps_valid, recent_losses_valid, s=15, alpha=0.6)
            
            # NaN/Inf 표시
            for i, (s, l) in enumerate(zip(recent_steps, recent_losses)):
                if np.isnan(l):
                    ax2.scatter([s], [ax2.get_ylim()[1] * 0.9] if ax2.get_ylim()[1] > 0 else [0.1], 
                              c='red', marker='x', s=150, zorder=10)
                elif np.isinf(l):
                    ax2.scatter([s], [ax2.get_ylim()[1] * 0.95] if ax2.get_ylim()[1] > 0 else [0.2], 
                              c='orange', marker='^', s=150, zorder=10)
    
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Loss (Recent 1000 Steps)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Loss plot saved to {output_file}")
    
    # 통계 출력
    valid_losses = [l for l in losses if not (np.isnan(l) or np.isinf(l))]
    if valid_losses:
        print(f"\n📊 Loss Statistics:")
        print(f"   Total steps: {len(steps)}")
        print(f"   Valid loss steps: {len(valid_losses)}")
        print(f"   NaN steps: {nan_mask.sum() if 'nan_mask' in locals() else 0}")
        print(f"   Inf steps: {inf_mask.sum() if 'inf_mask' in locals() else 0}")
        print(f"   Min loss: {min(valid_losses):.6f}")
        print(f"   Max loss: {max(valid_losses):.6f}")
        print(f"   Mean loss: {np.mean(valid_losses):.6f}")
        print(f"   Last valid loss: {valid_losses[-1]:.6f}")
        
        # NaN이 발생한 첫 번째 step 찾기
        nan_indices = [i for i, l in enumerate(losses) if np.isnan(l)]
        if nan_indices:
            first_nan_step = steps[nan_indices[0]]
            print(f"\n⚠️  First NaN at step: {first_nan_step}")
            if nan_indices[0] > 0:
                print(f"   Loss before NaN: {losses[nan_indices[0]-1]:.6f}")

if __name__ == "__main__":
    import sys
    
    log_file = sys.argv[1] if len(sys.argv) > 1 else "train.log"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "loss_plot.png"
    max_step = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    if max_step:
        print(f"📊 Parsing loss from {log_file} (up to step {max_step})...")
    else:
        print(f"📊 Parsing loss from {log_file}...")
    
    steps, losses = parse_loss_from_log(log_file, max_step=max_step)
    
    if len(steps) == 0:
        print("❌ No loss values found in log file")
        sys.exit(1)
    
    print(f"✅ Found {len(steps)} loss values")
    plot_loss(steps, losses, output_file)


