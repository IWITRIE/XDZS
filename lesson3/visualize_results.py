import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

class MLPResultsVisualizer:
    def __init__(self, csv_file='inference_results.csv'):
        """初始化可视化器"""
        self.df = pd.read_csv(csv_file)
        self.setup_metrics()
        
    def setup_metrics(self):
        """计算各种评估指标"""
        self.mse = mean_squared_error(self.df['True_Value'], self.df['Predicted_Value'])
        self.rmse = np.sqrt(self.mse)
        self.mae = mean_absolute_error(self.df['True_Value'], self.df['Predicted_Value'])
        self.r2 = r2_score(self.df['True_Value'], self.df['Predicted_Value'])
        self.mape = np.mean(np.abs(self.df['Error'] / self.df['True_Value'])) * 100
        
        # 添加相对误差
        self.df['Relative_Error'] = (self.df['Error'] / self.df['True_Value']) * 100
        
    def create_comprehensive_dashboard(self):
        """创建综合仪表板"""
        # 创建子图
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                '真实值 vs 预测值', '误差分布', '时间序列对比',
                '残差图', '误差热力图', 'Q-Q图',
                '累积误差', '预测准确性', '性能指标雷达图'
            ),
            specs=[[{"type": "scatter"}, {"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}, {"type": "scatterpolar"}]],
            vertical_spacing=0.08, horizontal_spacing=0.08
        )
        
        # 1. 真实值 vs 预测值
        fig.add_trace(
            go.Scatter(
                x=self.df['True_Value'], 
                y=self.df['Predicted_Value'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=self.df['Abs_Error'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="绝对误差", x=0.35)
                ),
                name='预测点',
                hovertemplate='真实值: %{x:.2f}<br>预测值: %{y:.2f}<br>误差: %{marker.color:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 添加完美预测线
        min_val, max_val = min(self.df['True_Value'].min(), self.df['Predicted_Value'].min()), \
                          max(self.df['True_Value'].max(), self.df['Predicted_Value'].max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                      mode='lines', name='完美预测', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # 2. 误差分布
        fig.add_trace(
            go.Histogram(
                x=self.df['Error'],
                nbinsx=30,
                name='误差分布',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # 3. 时间序列对比
        fig.add_trace(
            go.Scatter(
                x=self.df['Position'], 
                y=self.df['True_Value'],
                mode='lines+markers',
                name='真实值',
                line=dict(color='blue', width=2)
            ),
            row=1, col=3
        )
        fig.add_trace(
            go.Scatter(
                x=self.df['Position'], 
                y=self.df['Predicted_Value'],
                mode='lines+markers',
                name='预测值',
                line=dict(color='red', width=2)
            ),
            row=1, col=3
        )
        
        # 4. 残差图
        fig.add_trace(
            go.Scatter(
                x=self.df['Predicted_Value'], 
                y=self.df['Error'],
                mode='markers',
                marker=dict(size=6, color='orange', opacity=0.6),
                name='残差'
            ),
            row=2, col=1
        )
        
        # 添加零线
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        
        # 5. 误差热力图 (分段)
        position_bins = pd.cut(self.df['Position'], bins=10)
        value_bins = pd.cut(self.df['True_Value'], bins=10)
        heatmap_data = self.df.groupby([position_bins, value_bins])['Abs_Error'].mean().unstack()
        
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="平均绝对误差", x=0.68)
            ),
            row=2, col=2
        )
        
        # 6. Q-Q图
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(self.df)))
        sample_quantiles = np.sort(self.df['Error'])
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                marker=dict(size=6, color='purple'),
                name='Q-Q点'
            ),
            row=2, col=3
        )
        
        # 7. 累积误差
        cumulative_error = np.cumsum(np.abs(self.df['Error']))
        fig.add_trace(
            go.Scatter(
                x=self.df['Position'],
                y=cumulative_error,
                mode='lines',
                fill='tonexty',
                name='累积误差',
                line=dict(color='green', width=3)
            ),
            row=3, col=1
        )
        
        # 8. 预测准确性分级
        accuracy_bins = ['优秀 (<1%)', '良好 (1-3%)', '一般 (3-5%)', '较差 (>5%)']
        accuracy_counts = [
            sum(self.df['Relative_Error'].abs() < 1),
            sum((self.df['Relative_Error'].abs() >= 1) & (self.df['Relative_Error'].abs() < 3)),
            sum((self.df['Relative_Error'].abs() >= 3) & (self.df['Relative_Error'].abs() < 5)),
            sum(self.df['Relative_Error'].abs() >= 5)
        ]
        
        fig.add_trace(
            go.Bar(
                x=accuracy_bins,
                y=accuracy_counts,
                marker_color=['green', 'yellow', 'orange', 'red'],
                name='准确性分级'
            ),
            row=3, col=2
        )
        
        # 9. 性能指标雷达图
        metrics = ['R²', 'RMSE', 'MAE', 'MAPE']
        values = [self.r2, 1-min(self.rmse/max(self.df['True_Value']), 1), 
                 1-min(self.mae/max(self.df['True_Value']), 1), 1-min(self.mape/100, 1)]
        
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name='性能指标',
                line_color='navy'
            ),
            row=3, col=3
        )
        
        # 更新布局
        fig.update_layout(
            height=1200,
            title_text=f"MLP神经网络预测结果综合分析<br>R²={self.r2:.4f}, RMSE={self.rmse:.4f}, MAE={self.mae:.4f}",
            title_x=0.5,
            showlegend=False,
            font=dict(size=10)
        )
        
        # 保存和显示
        fig.write_html("mlp_comprehensive_analysis.html")
        fig.show()
        
        print(f"""
        ═══════════════════════════════════════
                    模型性能报告
        ═══════════════════════════════════════
        📊 基础指标:
           • R² 决定系数: {self.r2:.6f}
           • RMSE 均方根误差: {self.rmse:.6f}
           • MAE 平均绝对误差: {self.mae:.6f}
           • MAPE 平均绝对百分比误差: {self.mape:.4f}%
        
        📈 准确性分析:
           • 优秀预测 (<1% 误差): {accuracy_counts[0]} 个样本
           • 良好预测 (1-3% 误差): {accuracy_counts[1]} 个样本
           • 一般预测 (3-5% 误差): {accuracy_counts[2]} 个样本
           • 较差预测 (>5% 误差): {accuracy_counts[3]} 个样本
        
        🎯 模型评估:
           预测总体质量: {'优秀' if self.r2 > 0.9 else '良好' if self.r2 > 0.8 else '一般' if self.r2 > 0.7 else '需要改进'}
        ═══════════════════════════════════════
        """)
    
    def create_3d_analysis(self):
        """创建3D分析图"""
        fig = go.Figure()
        
        # 3D散点图
        fig.add_trace(go.Scatter3d(
            x=self.df['Position'],
            y=self.df['True_Value'],
            z=self.df['Predicted_Value'],
            mode='markers',
            marker=dict(
                size=8,
                color=self.df['Abs_Error'],
                colorscale='Rainbow',
                showscale=True,
                colorbar=dict(title="绝对误差")
            ),
            text=[f'位置: {pos}<br>真实: {true:.2f}<br>预测: {pred:.2f}<br>误差: {err:.2f}' 
                  for pos, true, pred, err in zip(self.df['Position'], self.df['True_Value'], 
                                                 self.df['Predicted_Value'], self.df['Error'])],
            hovertemplate='%{text}<extra></extra>',
            name='预测结果'
        ))
        
        fig.update_layout(
            title='MLP预测结果3D可视化',
            scene=dict(
                xaxis_title='时间位置',
                yaxis_title='真实值',
                zaxis_title='预测值',
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
            ),
            height=600
        )
        
        fig.write_html("mlp_3d_analysis.html")
        fig.show()
    
    def create_statistical_analysis(self):
        """创建统计分析图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MLP模型统计分析', fontsize=16, fontweight='bold')
        
        # 1. 误差分布与正态性检验
        axes[0, 0].hist(self.df['Error'], bins=30, alpha=0.7, color='skyblue', density=True)
        mu, sigma = stats.norm.fit(self.df['Error'])
        x = np.linspace(self.df['Error'].min(), self.df['Error'].max(), 100)
        axes[0, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label=f'正态拟合 μ={mu:.3f}, σ={sigma:.3f}')
        axes[0, 0].set_title('误差分布与正态性')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 相关性分析
        correlation = np.corrcoef(self.df['True_Value'], self.df['Predicted_Value'])[0, 1]
        axes[0, 1].scatter(self.df['True_Value'], self.df['Predicted_Value'], alpha=0.6, s=50)
        z = np.polyfit(self.df['True_Value'], self.df['Predicted_Value'], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(self.df['True_Value'], p(self.df['True_Value']), "r--", alpha=0.8)
        axes[0, 1].set_xlabel('真实值')
        axes[0, 1].set_ylabel('预测值')
        axes[0, 1].set_title(f'相关性分析 (r={correlation:.4f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 残差随时间变化
        axes[0, 2].plot(self.df['Position'], self.df['Error'], 'b-', alpha=0.7, linewidth=1)
        axes[0, 2].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[0, 2].fill_between(self.df['Position'], self.df['Error'], alpha=0.3)
        axes[0, 2].set_title('残差时间序列')
        axes[0, 2].set_xlabel('位置')
        axes[0, 2].set_ylabel('误差')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 预测区间分析
        percentiles = [10, 25, 50, 75, 90]
        error_percentiles = [np.percentile(self.df['Abs_Error'], p) for p in percentiles]
        axes[1, 0].bar([f'{p}%' for p in percentiles], error_percentiles, 
                      color=['lightgreen', 'yellow', 'orange', 'red', 'darkred'], alpha=0.7)
        axes[1, 0].set_title('绝对误差百分位数')
        axes[1, 0].set_ylabel('绝对误差')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 滑动窗口误差
        window_size = max(10, len(self.df) // 20)
        rolling_mae = pd.Series(self.df['Abs_Error']).rolling(window=window_size).mean()
        axes[1, 1].plot(self.df['Position'], rolling_mae, 'g-', linewidth=2, label=f'滑动MAE (窗口={window_size})')
        axes[1, 1].fill_between(self.df['Position'], rolling_mae, alpha=0.3, color='green')
        axes[1, 1].set_title('滑动窗口平均绝对误差')
        axes[1, 1].set_xlabel('位置')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 误差箱线图
        error_ranges = pd.cut(self.df['True_Value'], bins=5, labels=['很低', '低', '中', '高', '很高'])
        error_by_range = [self.df[error_ranges == label]['Abs_Error'].values for label in error_ranges.cat.categories]
        bp = axes[1, 2].boxplot(error_by_range, labels=error_ranges.cat.categories, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'yellow', 'orange', 'red']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        axes[1, 2].set_title('不同真实值范围的误差分布')
        axes[1, 2].set_xlabel('真实值范围')
        axes[1, 2].set_ylabel('绝对误差')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('mlp_statistical_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    print("🚀 开始MLP结果可视化分析...")
    
    # 创建可视化器
    visualizer = MLPResultsVisualizer()
    
    # 生成综合仪表板
    print("📊 生成综合分析仪表板...")
    visualizer.create_comprehensive_dashboard()
    
    # 生成3D分析
    print("🎯 生成3D可视化分析...")
    visualizer.create_3d_analysis()
    
    # 生成统计分析
    print("📈 生成统计分析图表...")
    visualizer.create_statistical_analysis()
    
    print("✅ 可视化分析完成!")
    print("📄 生成的文件:")
    print("   • mlp_comprehensive_analysis.html - 交互式综合分析")
    print("   • mlp_3d_analysis.html - 3D可视化分析")
    print("   • mlp_statistical_analysis.png - 统计分析图表")

if __name__ == "__main__":
    main()
