import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib样式
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class TrainingLossVisualizer:
    def __init__(self, csv_file='training_loss_history.csv'):
        """初始化训练损失可视化器"""
        try:
            self.df = pd.read_csv(csv_file)
            print(f"✅ 成功加载训练损失数据: {csv_file}")
            print(f"📊 数据形状: {self.df.shape}")
            print(f"📈 训练轮数范围: {self.df['Epoch'].min()} - {self.df['Epoch'].max()}")
            print(f"📉 损失值范围: {self.df['Loss'].min():.6f} - {self.df['Loss'].max():.6f}")
            
        except FileNotFoundError:
            print(f"❌ 错误: 找不到文件 {csv_file}")
            print("请确保运行了MLP训练程序并生成了损失历史文件")
            raise
        except Exception as e:
            print(f"❌ 读取损失历史文件时发生错误: {e}")
            raise
    
    def create_loss_overview(self):
        """创建损失总览图 - 专注前2000轮"""
        # 只使用前2000轮数据
        first_2000 = self.df[self.df['Epoch'] <= 2000]
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('MLP Training Loss Analysis - First 2000 Epochs Focus', fontsize=18, fontweight='bold')
        
        # 1. 双轴损失曲线（线性和对数）
        ax1 = axes[0, 0]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(first_2000['Epoch'], first_2000['Loss'], 'b-', linewidth=2, alpha=0.8, label='Linear Scale')
        line2 = ax2.plot(first_2000['Epoch'], first_2000['Loss'], 'r--', linewidth=2, alpha=0.6, label='Log Scale')
        ax2.set_yscale('log')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (Linear)', color='b')
        ax2.set_ylabel('Loss (Log)', color='r')
        ax1.set_title('Dual-Scale Loss Curve')
        ax1.grid(True, alpha=0.3)
        
        # 2. 损失下降速度分析
        loss_diff = first_2000['Loss'].diff()
        loss_acceleration = loss_diff.diff()  # 二阶导数
        
        axes[0, 1].plot(first_2000['Epoch'][1:], loss_diff[1:], 'g-', linewidth=1.5, alpha=0.7, label='Velocity')
        axes[0, 1].plot(first_2000['Epoch'][2:], loss_acceleration[2:]*1000, 'orange', linewidth=1, alpha=0.6, label='Acceleration (×1000)')
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.8)
        axes[0, 1].set_title('Loss Velocity & Acceleration')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss Change')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 多窗口滑动平均
        windows = [10, 50, 100]
        colors = ['lightblue', 'blue', 'darkblue']
        alphas = [0.4, 0.6, 0.8]
        
        for window, color, alpha in zip(windows, colors, alphas):
            rolling_loss = pd.Series(first_2000['Loss']).rolling(window=window).mean()
            axes[0, 2].plot(first_2000['Epoch'], rolling_loss, color=color, linewidth=2, 
                           alpha=alpha, label=f'MA-{window}')
        
        axes[0, 2].plot(first_2000['Epoch'], first_2000['Loss'], 'gray', linewidth=0.5, alpha=0.3, label='Original')
        axes[0, 2].set_title('Multi-Window Moving Averages')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_yscale('log')
        
        # 4. 学习阶段热力图
        # 将训练过程分为多个阶段，显示损失变化率
        n_phases = 20
        phase_size = len(first_2000) // n_phases
        phase_losses = []
        phase_epochs = []
        
        for i in range(n_phases):
            start_idx = i * phase_size
            end_idx = min((i + 1) * phase_size, len(first_2000))
            if start_idx < len(first_2000):
                phase_data = first_2000.iloc[start_idx:end_idx]
                phase_losses.append(phase_data['Loss'].mean())
                phase_epochs.append(phase_data['Epoch'].mean())
        
        im = axes[1, 0].imshow([phase_losses], aspect='auto', cmap='RdYlBu_r', interpolation='bilinear')
        axes[1, 0].set_title('Training Phase Heatmap')
        axes[1, 0].set_xlabel('Training Phase')
        axes[1, 0].set_ylabel('Loss Intensity')
        axes[1, 0].set_xticks(range(0, len(phase_losses), 5))
        axes[1, 0].set_xticklabels([f'{int(phase_epochs[i])}' for i in range(0, len(phase_epochs), 5)])
        plt.colorbar(im, ax=axes[1, 0], shrink=0.6)
        
        # 5. 损失分布的演化
        # 分三个阶段显示损失分布
        early = first_2000[first_2000['Epoch'] <= 600]
        middle = first_2000[(first_2000['Epoch'] > 600) & (first_2000['Epoch'] <= 1200)]
        late = first_2000[first_2000['Epoch'] > 1200]
        
        axes[1, 1].hist(early['Loss'], bins=30, alpha=0.5, color='red', density=True, label='Early (1-600)')
        axes[1, 1].hist(middle['Loss'], bins=30, alpha=0.5, color='orange', density=True, label='Middle (601-1200)')
        axes[1, 1].hist(late['Loss'], bins=30, alpha=0.5, color='green', density=True, label='Late (1201-2000)')
        axes[1, 1].set_title('Loss Distribution Evolution')
        axes[1, 1].set_xlabel('Loss Value')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        # 6. 收敛质量评估
        # 计算收敛指标
        recent_window = 200
        recent_loss = first_2000['Loss'].iloc[-recent_window:]
        convergence_stability = recent_loss.std() / recent_loss.mean()
        
        # 绘制收敛质量指标
        window_stds = []
        window_means = []
        window_epochs = []
        
        for i in range(100, len(first_2000), 50):
            window_data = first_2000['Loss'].iloc[max(0, i-100):i]
            window_stds.append(window_data.std() / window_data.mean())
            window_means.append(window_data.mean())
            window_epochs.append(first_2000['Epoch'].iloc[i])
        
        ax6 = axes[1, 2]
        ax6_twin = ax6.twinx()
        
        line1 = ax6.plot(window_epochs, window_stds, 'purple', linewidth=2, label='Stability Index')
        line2 = ax6_twin.plot(window_epochs, window_means, 'green', linewidth=2, alpha=0.7, label='Mean Loss')
        
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Stability Index (CV)', color='purple')
        ax6_twin.set_ylabel('Mean Loss', color='green')
        ax6_twin.set_yscale('log')
        ax6.set_title('Convergence Quality Analysis')
        ax6.grid(True, alpha=0.3)
        
        # 添加收敛质量文本
        ax6.text(0.05, 0.95, f'Final Stability: {convergence_stability:.4f}', 
                transform=ax6.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        plt.savefig('training_loss_overview.png', dpi=300, bbox_inches='tight')
        print("📄 训练损失总览图已保存至: training_loss_overview.png")
        plt.show()

    def create_interactive_loss_plot(self):
        """创建高级交互式损失可视化 - 专注前2000轮"""
        first_2000 = self.df[self.df['Epoch'] <= 2000]
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Loss Curve with Annotations',
                'Loss Rate Analysis',
                'Multi-Scale Smoothing',
                'Phase Transition Detection',
                'Convergence Metrics',
                'Learning Efficiency Map'
            ],
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"type": "heatmap"}]]
        )
        
        # 1. 带注释的损失曲线
        fig.add_trace(
            go.Scatter(
                x=first_2000['Epoch'],
                y=first_2000['Loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color='blue', width=2),
                hovertemplate='Epoch: %{x}<br>Loss: %{y:.6f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 添加关键点注释
        loss_values = first_2000['Loss'].values
        epochs = first_2000['Epoch'].values
        
        # 找到损失下降最快的点
        loss_diff = np.diff(loss_values)
        max_drop_idx = np.argmin(loss_diff) + 1
        
        # 找到收敛开始的点（损失变化趋于平缓）
        rolling_std = pd.Series(loss_values).rolling(window=100).std()
        convergence_start = np.argmin(rolling_std[100:]) + 100
        
        fig.add_trace(
            go.Scatter(
                x=[epochs[max_drop_idx], epochs[convergence_start]],
                y=[loss_values[max_drop_idx], loss_values[convergence_start]],
                mode='markers+text',
                text=['Max Drop', 'Convergence Start'],
                textposition='top center',
                marker=dict(size=10, color='red'),
                name='Key Points'
            ),
            row=1, col=1
        )
        
        # 2. 损失变化率分析
        loss_pct_change = first_2000['Loss'].pct_change() * 100
        
        fig.add_trace(
            go.Scatter(
                x=first_2000['Epoch'][1:],
                y=loss_pct_change[1:],
                mode='lines',
                name='Loss % Change',
                line=dict(color='orange', width=1),
                fill='tonexty'
            ),
            row=1, col=2
        )
        
        # 3. 多尺度平滑
        for window, color in zip([10, 50, 100], ['lightblue', 'blue', 'darkblue']):
            rolling_loss = pd.Series(first_2000['Loss']).rolling(window=window).mean()
            fig.add_trace(
                go.Scatter(
                    x=first_2000['Epoch'],
                    y=rolling_loss,
                    mode='lines',
                    name=f'MA-{window}',
                    line=dict(color=color, width=2),
                    opacity=0.8
                ),
                row=2, col=1
            )
        
        # 4. 相位转换检测
        # 使用变化点检测算法
        loss_diff = np.abs(np.diff(loss_values))
        threshold = np.percentile(loss_diff, 95)
        phase_changes = np.where(loss_diff > threshold)[0]
        
        fig.add_trace(
            go.Scatter(
                x=first_2000['Epoch'],
                y=loss_diff,
                mode='lines',
                name='Loss Gradient',
                line=dict(color='green', width=1)
            ),
            row=2, col=2
        )
        
        # 标记相位变化点
        if len(phase_changes) > 0:
            fig.add_trace(
                go.Scatter(
                    x=epochs[phase_changes],
                    y=loss_diff[phase_changes],
                    mode='markers',
                    name='Phase Changes',
                    marker=dict(size=8, color='red', symbol='star')
                ),
                row=2, col=2
            )
        
        # 5. 收敛指标
        window_size = 100
        convergence_metrics = []
        metric_epochs = []
        
        for i in range(window_size, len(first_2000)):
            window_data = first_2000['Loss'].iloc[i-window_size:i]
            cv = window_data.std() / window_data.mean()  # 变异系数
            convergence_metrics.append(cv)
            metric_epochs.append(first_2000['Epoch'].iloc[i])
        
        fig.add_trace(
            go.Scatter(
                x=metric_epochs,
                y=convergence_metrics,
                mode='lines',
                name='Convergence Index',
                line=dict(color='purple', width=2)
            ),
            row=3, col=1
        )
        
        # 6. 学习效率热力图
        # 创建2D热力图：epoch vs loss level
        n_bins = 20
        epoch_bins = np.linspace(first_2000['Epoch'].min(), first_2000['Epoch'].max(), n_bins)
        loss_bins = np.linspace(first_2000['Loss'].min(), first_2000['Loss'].max(), n_bins)
        
        # 计算每个bin的学习效率
        efficiency_matrix = np.zeros((n_bins-1, n_bins-1))
        
        for i in range(len(first_2000)-1):
            epoch_idx = np.digitize(first_2000['Epoch'].iloc[i], epoch_bins) - 1
            loss_idx = np.digitize(first_2000['Loss'].iloc[i], loss_bins) - 1
            
            if 0 <= epoch_idx < n_bins-1 and 0 <= loss_idx < n_bins-1:
                loss_change = first_2000['Loss'].iloc[i+1] - first_2000['Loss'].iloc[i]
                efficiency_matrix[loss_idx, epoch_idx] = -loss_change  # 负值表示损失减少
        
        fig.add_trace(
            go.Heatmap(
                z=efficiency_matrix,
                x=epoch_bins[:-1],
                y=loss_bins[:-1],
                colorscale='RdBu',
                name='Learning Efficiency'
            ),
            row=3, col=2
        )
        
        # 更新布局
        fig.update_layout(
            height=1200,
            title_text="Advanced MLP Training Loss Analysis - First 2000 Epochs",
            showlegend=True
        )
        
        # 设置对数尺度
        fig.update_yaxes(type="log", row=1, col=1)
        fig.update_yaxes(type="log", row=2, col=1)
        
        fig.write_html("training_loss_interactive.html")
        print("📄 交互式训练损失分析已保存至: training_loss_interactive.html")

    def create_convergence_analysis(self):
        """创建深度收敛性分析 - 专注前2000轮"""
        first_2000 = self.df[self.df['Epoch'] <= 2000]
        
        fig, axes = plt.subplots(3, 3, figsize=(21, 15))
        fig.suptitle('Deep Convergence Analysis - First 2000 Epochs', fontsize=18, fontweight='bold')
        
        # 1. 损失景观分析
        axes[0, 0].plot(first_2000['Epoch'], first_2000['Loss'], 'b-', linewidth=1.5, alpha=0.7)
        axes[0, 0].fill_between(first_2000['Epoch'], first_2000['Loss'], alpha=0.3, color='lightblue')
        axes[0, 0].set_title('Loss Landscape')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 学习速度谱
        loss_diff = np.abs(np.diff(first_2000['Loss'].values))
        frequencies = np.fft.fftfreq(len(loss_diff))
        fft_values = np.abs(np.fft.fft(loss_diff))
        
        axes[0, 1].plot(frequencies[:len(frequencies)//2], fft_values[:len(fft_values)//2], 'r-', linewidth=1.5)
        axes[0, 1].set_title('Learning Speed Spectrum')
        axes[0, 1].set_xlabel('Frequency')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 动态稳定性分析
        window_sizes = [20, 50, 100, 200]
        colors = ['red', 'orange', 'green', 'blue']
        
        for window_size, color in zip(window_sizes, colors):
            rolling_std = pd.Series(first_2000['Loss']).rolling(window=window_size).std()
            rolling_mean = pd.Series(first_2000['Loss']).rolling(window=window_size).mean()
            stability = rolling_std / rolling_mean
            
            axes[0, 2].plot(first_2000['Epoch'], stability, color=color, linewidth=1.5, 
                           alpha=0.7, label=f'Window-{window_size}')
        
        axes[0, 2].set_title('Dynamic Stability Analysis')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Coefficient of Variation')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 损失下降分解
        # 使用经验模态分解模拟
        loss_values = first_2000['Loss'].values
        trend = pd.Series(loss_values).rolling(window=200, center=True).mean()
        residual = loss_values - trend.fillna(method='bfill').fillna(method='ffill')
        
        axes[1, 0].plot(first_2000['Epoch'], loss_values, 'b-', linewidth=1, alpha=0.7, label='Original')
        axes[1, 0].plot(first_2000['Epoch'], trend, 'r-', linewidth=2, label='Trend')
        axes[1, 0].fill_between(first_2000['Epoch'], trend-np.abs(residual), trend+np.abs(residual), 
                               alpha=0.3, color='gray', label='Residual Band')
        axes[1, 0].set_title('Loss Decomposition')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 收敛速度热力图
        n_segments = 10
        segment_size = len(first_2000) // n_segments
        convergence_matrix = []
        
        for i in range(n_segments):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, len(first_2000))
            segment_data = first_2000.iloc[start_idx:end_idx]
            
            if len(segment_data) > 1:
                # 计算不同指标
                loss_change = (segment_data['Loss'].iloc[-1] - segment_data['Loss'].iloc[0]) / segment_data['Loss'].iloc[0]
                volatility = segment_data['Loss'].std() / segment_data['Loss'].mean()
                efficiency = -loss_change / len(segment_data)  # 每epoch的改进
                
                convergence_matrix.append([loss_change, volatility, efficiency])
        
        convergence_matrix = np.array(convergence_matrix).T
        im = axes[1, 1].imshow(convergence_matrix, aspect='auto', cmap='RdYlBu', interpolation='bilinear')
        axes[1, 1].set_title('Convergence Metrics Heatmap')
        axes[1, 1].set_xlabel('Training Segment')
        axes[1, 1].set_ylabel('Metrics')
        axes[1, 1].set_yticks([0, 1, 2])
        axes[1, 1].set_yticklabels(['Loss Change', 'Volatility', 'Efficiency'])
        plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
        
        # 6. 学习阶段识别
        # 使用K-means对损失行为进行聚类
        from sklearn.cluster import KMeans
        
        # 准备特征
        features = []
        window_size = 100
        for i in range(window_size, len(first_2000)):
            window_data = first_2000['Loss'].iloc[i-window_size:i]
            mean_loss = window_data.mean()
            std_loss = window_data.std()
            trend_slope = np.polyfit(range(window_size), window_data.values, 1)[0]
            features.append([mean_loss, std_loss, trend_slope])
        
        features = np.array(features)
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        colors_cluster = ['red', 'green', 'blue']
        for cluster_id in range(3):
            mask = clusters == cluster_id
            cluster_epochs = first_2000['Epoch'].iloc[window_size:].iloc[mask]
            cluster_losses = first_2000['Loss'].iloc[window_size:].iloc[mask]
            axes[1, 2].scatter(cluster_epochs, cluster_losses, c=colors_cluster[cluster_id], 
                             alpha=0.6, s=20, label=f'Phase {cluster_id+1}')
        
        axes[1, 2].set_title('Learning Phase Identification')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].legend()
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7-9. 其他高级分析
        # 7. 损失梯度分析
        loss_gradient = np.gradient(first_2000['Loss'].values)
        axes[2, 0].plot(first_2000['Epoch'], np.abs(loss_gradient), 'purple', linewidth=1.5)
        axes[2, 0].set_title('Loss Gradient Magnitude')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('|∇Loss|')
        axes[2, 0].set_yscale('log')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 8. 收敛质量评分
        # 综合多个指标计算收敛质量
        final_loss = first_2000['Loss'].iloc[-1]
        loss_reduction = (first_2000['Loss'].iloc[0] - final_loss) / first_2000['Loss'].iloc[0]
        final_stability = first_2000['Loss'].iloc[-200:].std() / first_2000['Loss'].iloc[-200:].mean()
        
        quality_score = loss_reduction * (1 / (1 + final_stability)) * (1 / (1 + final_loss))
        
        # 绘制质量指标雷达图
        metrics = ['Loss Reduction', 'Stability', 'Final Loss', 'Overall Quality']
        values = [loss_reduction, 1/(1+final_stability), 1/(1+final_loss), quality_score]
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        values_plot = np.concatenate((values, [values[0]]))  # 闭合
        angles_plot = np.concatenate((angles, [angles[0]]))
        
        axes[2, 1] = plt.subplot(3, 3, 8, projection='polar')
        axes[2, 1].plot(angles_plot, values_plot, 'bo-', linewidth=2)
        axes[2, 1].fill(angles_plot, values_plot, alpha=0.25)
        axes[2, 1].set_xticks(angles)
        axes[2, 1].set_xticklabels(metrics)
        axes[2, 1].set_title('Convergence Quality Radar')
        
        # 9. 训练效率时间线
        efficiency_timeline = []
        epochs_timeline = []
        
        for i in range(50, len(first_2000), 50):
            window_data = first_2000['Loss'].iloc[max(0, i-50):i]
            if len(window_data) > 1:
                efficiency = -(window_data.iloc[-1] - window_data.iloc[0]) / len(window_data)
                efficiency_timeline.append(efficiency)
                epochs_timeline.append(first_2000['Epoch'].iloc[i])
        
        axes[2, 2].bar(epochs_timeline, efficiency_timeline, width=80, alpha=0.7, 
                      color=['red' if x < 0 else 'green' for x in efficiency_timeline])
        axes[2, 2].set_title('Training Efficiency Timeline')
        axes[2, 2].set_xlabel('Epoch')
        axes[2, 2].set_ylabel('Learning Efficiency')
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_convergence_analysis.png', dpi=300, bbox_inches='tight')
        print("📄 训练收敛性分析已保存至: training_convergence_analysis.png")
        plt.show()

    def print_training_statistics(self):
        """打印训练统计信息 - 专注前2000轮"""
        first_2000 = self.df[self.df['Epoch'] <= 2000]
        
        # 计算高级统计指标
        loss_reduction = (first_2000['Loss'].iloc[0] - first_2000['Loss'].iloc[-1]) / first_2000['Loss'].iloc[0]
        final_stability = first_2000['Loss'].iloc[-200:].std() / first_2000['Loss'].iloc[-200:].mean()
        
        # 找到最快学习阶段
        loss_diff = np.abs(np.diff(first_2000['Loss'].values))
        max_learning_epoch = first_2000['Epoch'].iloc[np.argmax(loss_diff)]
        
        print(f"""
        ═══════════════════════════════════════
                前2000轮训练深度分析报告
        ═══════════════════════════════════════
        📊 基础统计:
           • 分析轮数: 前{len(first_2000)}轮
           • 初始损失: {first_2000['Loss'].iloc[0]:.6f}
           • 最终损失: {first_2000['Loss'].iloc[-1]:.6f}
           • 最低损失: {first_2000['Loss'].min():.6f}
           • 损失降低率: {loss_reduction*100:.2f}%
        
        📈 学习动态:
           • 最快学习点: 第{max_learning_epoch:.0f}轮
           • 平均学习速度: {-np.mean(np.diff(first_2000['Loss'].values)):.6f}/epoch
           • 学习稳定性: {final_stability:.4f} (变异系数)
           • 收敛趋势: {'收敛中' if final_stability < 0.1 else '需要更多训练'}
        
        🎯 训练质量评估:
           • 学习效率: {'高效' if loss_reduction > 0.9 else '良好' if loss_reduction > 0.7 else '一般'}
           • 稳定性评级: {'优秀' if final_stability < 0.05 else '良好' if final_stability < 0.1 else '需要改进'}
           • 整体评分: {(loss_reduction * (1/(1+final_stability)) * 100):.1f}/100
        
        💡 训练建议:
        {f'   • 训练表现优异，可以考虑减少训练轮数' if loss_reduction > 0.95 and final_stability < 0.05
          else f'   • 建议继续训练至更低损失' if final_stability > 0.1
          else f'   • 当前训练状态良好，可适当调整学习率'}
        ═══════════════════════════════════════
        """)

def main():
    """主函数"""
    print("🚀 开始训练损失可视化分析...")
    print("=" * 50)
    
    try:
        # 创建可视化器
        visualizer = TrainingLossVisualizer()
        
        # 打印统计信息
        print("\n📊 生成训练统计信息...")
        visualizer.print_training_statistics()
        
        # 创建损失总览图
        print("\n📈 生成训练损失总览图...")
        visualizer.create_loss_overview()
        
        # 创建交互式损失图
        print("\n🎯 生成交互式训练损失分析...")
        visualizer.create_interactive_loss_plot()
        
        # 创建收敛性分析
        print("\n🔬 生成训练收敛性分析...")
        visualizer.create_convergence_analysis()
        
        print("\n" + "=" * 50)
        print("✅ 训练损失可视化分析完成!")
        print("📄 生成的文件:")
        print("   • training_loss_overview.png - 训练损失总览图")
        print("   • training_loss_interactive.html - 交互式训练损失分析")
        print("   • training_convergence_analysis.png - 训练收敛性分析")
        print("\n💡 提示: 在浏览器中打开HTML文件查看交互式图表")
        
    except Exception as e:
        print(f"\n❌ 训练损失可视化分析过程中发生错误: {e}")
        print("请检查训练损失历史文件是否存在且格式正确")

if __name__ == "__main__":
    main()
