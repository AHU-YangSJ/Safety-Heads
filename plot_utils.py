import os
import numpy as np
import matplotlib.pyplot as plt

def plot_vector_line(vec, out_path, title=None, xlabel='Index', ylabel='Value', dpi=200, figsize=(8,4)):
    """
    将一维向量绘制为折线图并保存为圖片（PNG）。
    支持 numpy 数组、list 或可转为 numpy 的对象。
    """
    # 转为 numpy 一维数组
    try:
        import torch
        if isinstance(vec, torch.Tensor):
            vec = vec.cpu().numpy()
    except Exception:
        pass

    arr = np.asarray(vec).ravel()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    plt.figure(figsize=figsize)
    plt.plot(arr, marker='o', linewidth=1)
    plt.title(title or os.path.basename(out_path))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

def plot_attention_heads(data_list, save_path, title="Attention Heads", x_label="Heads", y_label="Value"):
    """
    绘制多个序列的折线图对比
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 6))
    
    labels = ['1-Malicious-Normal', '2-Normal-Normal', '3-Malicious-Malicious'] 
    
    for i, data in enumerate(data_list):
        try:
            import torch
            if isinstance(data, torch.Tensor):
                data = data.cpu().detach().numpy()
        except ImportError:
            pass
        except Exception:
            pass

        # Ensure 1D array
        data = np.array(data).flatten()
        
        label = labels[i] if i < len(labels) else f'Data {i}'
        plt.plot(data, label=label, alpha=0.8)
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_heatmap(matrix, save_path, title="Heatmap", x_label="X", y_label="Y"):
    """
    绘制矩阵的热力图
    """
    try:
        import torch
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.cpu().detach().numpy()
    except ImportError:
        pass
    except Exception:
        pass

    matrix = np.array(matrix)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 8))

    # 字体加大
    plt.imshow(matrix, aspect='auto', cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    # 图中涉及的文字都大一点
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    
    plt.title(title, fontsize=30)
    plt.xlabel(x_label, fontsize=24)
    plt.ylabel(y_label, fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_distribution_bar(data, save_path, title="Distribution", x_label="Range", y_label="Count", bins=10):
    """
    绘制数据的区间分布柱状图 (例如 0-0.1, 0.1-0.2 ...)
    并显示每个区间的计数。
    """
    try:
        import torch
        if isinstance(data, torch.Tensor):
            data = data.cpu().detach().numpy()
    except Exception:
        pass
        
    data = np.array(data).flatten()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 生成 0到1 的区间
    bin_edges = np.linspace(0, 1, bins + 1)
    counts, _ = np.histogram(data, bins=bin_edges)
    
    # 生成X轴标签
    labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(bins)]
    
    plt.figure(figsize=(10, 6))
    
    # 绘制柱状图
    bars = plt.bar(labels, counts, width=0.6, color='#0C76B1', edgecolor='black', alpha=0.8)
    
    # 在柱子上方显示具体数值
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}',
                     ha='center', va='bottom', fontsize=10)
    
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=30)  # 防止标签重叠
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_strip_distribution(data, save_path, title=None, x_label=None, model_name="Model"):
    """
    创建 1D 条形图（抖动图），带有颜色映射和垂直阈值线。
    """
    data_np = data.flatten().cpu().numpy() if hasattr(data, 'cpu') else np.array(data).flatten()
    
    plt.figure(figsize=(10, 2.5))
    
    # Jitter y coordinates to spread points vertically (strip plot)
    y = np.random.normal(0, 0.08, size=len(data_np))
    
    # Plot scatter with colors mapped to x values (Green to Purple style via PRGn_r or similar)
    plt.scatter(data_np, y, c=data_np, cmap='PRGn_r', alpha=0.8, s=70, edgecolors='black', linewidths=0.25)
    
    # Add vertical dashed line at top 5% threshold
    threshold = np.percentile(data_np, 95)
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.axvline(x=0.9, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    plt.yticks([0], [model_name], fontsize=16)
    plt.ylim(-0.3, 0.3)
    
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if title:
        plt.title(title, fontsize=20)
        
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
