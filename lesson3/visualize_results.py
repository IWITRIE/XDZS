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

# 设置中文字体和样式 - 增强中文支持
try:
    # 尝试设置中文字体
    import matplotlib.font_manager as fm
    font_dirs = ['/usr/share/fonts/', '/System/Library/Fonts/', 'C:/Windows/Fonts/']
    font_files = []
    for font_dir in font_dirs:
        try:
            font_files.extend(fm.findSystemFonts(fontpaths=font_dir, fontext='ttf'))
        except:
            continue
    
    # 查找中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans', 'Arial Unicode MS']
    available_font = None
    for font in chinese_fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            # 测试字体是否可用
            fig, ax = plt.subplots(figsize=(1,1))
            ax.text(0.5, 0.5, '测试中文', fontsize=12)
            plt.close(fig)
            available_font = font
            break
        except:
            continue
    
    if available_font:
        plt.rcParams['font.sans-serif'] = [available_font, 'DejaVu Sans']
        print(f"✅ 使用中文字体: {available_font}")
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("⚠️  未找到中文字体，使用默认字体")
        
except Exception as e:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    print(f"⚠️  字体设置失败: {e}")

plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

class MLPResultsVisualizer:
    def __init__(self, csv_file='inference_results.csv'):
        """初始化可视化器"""
        try:
            self.df = pd.read_csv(csv_file)
            print(f"✅ 成功加载数据文件: {csv_file}")
            print(f"📊 数据形状: {self.df.shape}")
            
            # 检查是否有新的归一化列
            if 'Normalized_True' in self.df.columns and 'Normalized_Pred' in self.df.columns:
                print("📈 检测到归一化数据列，将进行详细分析")
                self.has_normalized_data = True
            else:
                print("📈 使用标准数据列进行分析")
                self.has_normalized_data = False
                
            self.setup_metrics()
            
        except FileNotFoundError:
            print(f"❌ 错误: 找不到文件 {csv_file}")
            print("请确保运行了MLP推理程序并生成了结果文件")
            raise
        except Exception as e:
            print(f"❌ 读取数据文件时发生错误: {e}")
            raise
        
    def setup_metrics(self):
        """计算各种评估指标"""
        self.mse = mean_squared_error(self.df['True_Value'], self.df['Predicted_Value'])
        self.rmse = np.sqrt(self.mse)
        self.mae = mean_absolute_error(self.df['True_Value'], self.df['Predicted_Value'])
        self.r2 = r2_score(self.df['True_Value'], self.df['Predicted_Value'])
        
        # 处理MAPE计算中的除零问题
        non_zero_mask = self.df['True_Value'] != 0
        if non_zero_mask.sum() > 0:
            self.mape = np.mean(np.abs(self.df.loc[non_zero_mask, 'Error'] / self.df.loc[non_zero_mask, 'True_Value'])) * 100
        else:
            self.mape = float('inf')
        
        # 添加相对误差
        self.df['Relative_Error'] = np.where(
            self.df['True_Value'] != 0,
            (self.df['Error'] / self.df['True_Value']) * 100,
            0
        )
        
        # 打印数据统计信息
        print(f"""
        📊 数据统计信息:
           • 样本数量: {len(self.df)}
           • 真实值范围: [{self.df['True_Value'].min():.4f}, {self.df['True_Value'].max():.4f}]
           • 预测值范围: [{self.df['Predicted_Value'].min():.4f}, {self.df['Predicted_Value'].max():.4f}]
           • 误差范围: [{self.df['Error'].min():.4f}, {self.df['Error'].max():.4f}]
        """)
        
    def create_comprehensive_dashboard(self):
        """创建综合仪表板"""
        # 创建子图 - 使用英文标题
        english_titles = (
            'True vs Predicted Values', 'Error Distribution', 'Time Series Comparison',
            'Residual Plot', 'Error Heatmap', 'Q-Q Normality Test',
            'Cumulative Error', 'Prediction Accuracy Classification', 'Performance Metrics Radar'
        )
        
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=english_titles,
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
                    colorbar=dict(title="Absolute Error", x=0.35)
                ),
                name='Prediction Points',
                hovertemplate='True Value: %{x:.2f}<br>Predicted Value: %{y:.2f}<br>Error: %{marker.color:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 添加完美预测线
        min_val, max_val = min(self.df['True_Value'].min(), self.df['Predicted_Value'].min()), \
                          max(self.df['True_Value'].max(), self.df['Predicted_Value'].max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                      mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # 2. 误差分布
        fig.add_trace(
            go.Histogram(
                x=self.df['Error'],
                nbinsx=30,
                name='Error Distribution',
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
                name='True Values',
                line=dict(color='blue', width=2)
            ),
            row=1, col=3
        )
        fig.add_trace(
            go.Scatter(
                x=self.df['Position'], 
                y=self.df['Predicted_Value'],
                mode='lines+markers',
                name='Predicted Values',
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
                name='Residuals'
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
                colorbar=dict(title="Mean Absolute Error", x=0.68)
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
                name='Q-Q Points'
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
                name='Cumulative Error',
                line=dict(color='green', width=3)
            ),
            row=3, col=1
        )
        
        # 8. 预测准确性分级
        accuracy_bins = ['Excellent (<1%)', 'Good (1-3%)', 'Fair (3-5%)', 'Poor (>5%)']
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
                name='Accuracy Classification'
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
                name='Performance Metrics',
                line_color='navy'
            ),
            row=3, col=3
        )
        
        # 更新布局 - 英文标题
        fig.update_layout(
            height=1200,
            title_text=f"MLP Neural Network Prediction Results Comprehensive Analysis<br>R² Coefficient={self.r2:.4f}, RMSE={self.rmse:.4f}, MAE={self.mae:.4f}",
            title_x=0.5,
            showlegend=False,
            font=dict(size=11)
        )
        
        # 保存和显示
        fig.write_html("mlp_comprehensive_analysis.html")
        print("📄 交互式综合分析已保存至: mlp_comprehensive_analysis.html")
        
        # 中文性能报告
        accuracy_counts = [
            sum(self.df['Relative_Error'].abs() < 1),
            sum((self.df['Relative_Error'].abs() >= 1) & (self.df['Relative_Error'].abs() < 3)),
            sum((self.df['Relative_Error'].abs() >= 3) & (self.df['Relative_Error'].abs() < 5)),
            sum(self.df['Relative_Error'].abs() >= 5)
        ]
        
        print(f"""
        ═══════════════════════════════════════
                    模型性能详细报告
        ═══════════════════════════════════════
        📊 核心评估指标:
           • R² 决定系数: {self.r2:.6f} {'(优秀)' if self.r2 > 0.9 else '(良好)' if self.r2 > 0.8 else '(一般)' if self.r2 > 0.7 else '(需要改进)'}
           • RMSE 均方根误差: {self.rmse:.6f}
           • MAE 平均绝对误差: {self.mae:.6f}
           • MAPE 平均绝对百分比误差: {self.mape:.4f}%
        
        📈 预测准确性分析:
           • 🟢 优秀预测 (相对误差 < 1%): {accuracy_counts[0]} 个样本 ({accuracy_counts[0]/len(self.df)*100:.1f}%)
           • 🟡 良好预测 (相对误差 1-3%): {accuracy_counts[1]} 个样本 ({accuracy_counts[1]/len(self.df)*100:.1f}%)
           • 🟠 一般预测 (相对误差 3-5%): {accuracy_counts[2]} 个样本 ({accuracy_counts[2]/len(self.df)*100:.1f}%)
           • 🔴 较差预测 (相对误差 > 5%): {accuracy_counts[3]} 个样本 ({accuracy_counts[3]/len(self.df)*100:.1f}%)
        
        🎯 模型整体评估:
           预测质量等级: {'🏆 优秀' if self.r2 > 0.9 else '✅ 良好' if self.r2 > 0.8 else '⚠️  一般' if self.r2 > 0.7 else '❌ 需要改进'}
           
        {'🔍 调试信息:' if self.has_normalized_data else ''}
        {f'   • 检测到归一化数据，可能存在标准化问题' if self.has_normalized_data and abs(self.df["Predicted_Value"].mean() - self.df["True_Value"].mean()) > self.df["True_Value"].std() else ''}
        ═══════════════════════════════════════
        """)
        
        # 如果有归一化数据，进行额外分析
        if self.has_normalized_data:
            self.analyze_normalization_issues()
    
    def analyze_normalization_issues(self):
        """分析归一化相关问题"""
        print("\n🔬 归一化数据分析:")
        print(f"   • 归一化真实值范围: [{self.df['Normalized_True'].min():.4f}, {self.df['Normalized_True'].max():.4f}]")
        print(f"   • 归一化预测值范围: [{self.df['Normalized_Pred'].min():.4f}, {self.df['Normalized_Pred'].max():.4f}]")
        
        # 检查归一化预测值是否合理
        norm_pred_mean = self.df['Normalized_Pred'].mean()
        norm_true_mean = self.df['Normalized_True'].mean()
        
        if abs(norm_pred_mean - 0.5) > 0.3:
            print(f"   ⚠️  警告: 归一化预测值均值 ({norm_pred_mean:.4f}) 偏离期望值 (0.5)")
        
        if abs(norm_pred_mean - norm_true_mean) > 0.2:
            print(f"   ⚠️  警告: 归一化预测值与真实值均值差异较大 ({abs(norm_pred_mean - norm_true_mean):.4f})")
            print(f"      这可能表明模型训练或归一化过程存在问题")

    def create_statistical_analysis(self):
        """创建统计分析图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MLP Model Statistical Analysis Report', fontsize=16, fontweight='bold')
        
        # 1. 误差分布与正态性检验
        axes[0, 0].hist(self.df['Error'], bins=30, alpha=0.7, color='skyblue', density=True)
        mu, sigma = stats.norm.fit(self.df['Error'])
        x = np.linspace(self.df['Error'].min(), self.df['Error'].max(), 100)
        axes[0, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, 
                       label=f'Normal Fit μ={mu:.3f}, σ={sigma:.3f}')
        axes[0, 0].set_title('Error Distribution and Normality Test')
        axes[0, 0].set_xlabel('Prediction Error')
        axes[0, 0].set_ylabel('Probability Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 相关性分析
        correlation = np.corrcoef(self.df['True_Value'], self.df['Predicted_Value'])[0, 1]
        axes[0, 1].scatter(self.df['True_Value'], self.df['Predicted_Value'], alpha=0.6, s=50)
        z = np.polyfit(self.df['True_Value'], self.df['Predicted_Value'], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(self.df['True_Value'], p(self.df['True_Value']), "r--", alpha=0.8, lw=2)
        axes[0, 1].set_xlabel('True Values')
        axes[0, 1].set_ylabel('Predicted Values')
        axes[0, 1].set_title(f'True vs Predicted Values Correlation (r={correlation:.4f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 残差随时间变化
        axes[0, 2].plot(self.df['Position'], self.df['Error'], 'b-', alpha=0.7, linewidth=1)
        axes[0, 2].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[0, 2].fill_between(self.df['Position'], self.df['Error'], alpha=0.3)
        axes[0, 2].set_title('Residual Time Series')
        axes[0, 2].set_xlabel('Position')
        axes[0, 2].set_ylabel('Error')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 预测区间分析
        percentiles = [10, 25, 50, 75, 90]
        error_percentiles = [np.percentile(self.df['Abs_Error'], p) for p in percentiles]
        axes[1, 0].bar([f'{p}%' for p in percentiles], error_percentiles, 
                      color=['lightgreen', 'yellow', 'orange', 'red', 'darkred'], alpha=0.7)
        axes[1, 0].set_title('Absolute Error Percentiles')
        axes[1, 0].set_ylabel('Absolute Error')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 滑动窗口误差
        window_size = max(10, len(self.df) // 20)
        rolling_mae = pd.Series(self.df['Abs_Error']).rolling(window=window_size).mean()
        axes[1, 1].plot(self.df['Position'], rolling_mae, 'g-', linewidth=2, label=f'Rolling MAE (window={window_size})')
        axes[1, 1].fill_between(self.df['Position'], rolling_mae, alpha=0.3, color='green')
        axes[1, 1].set_title('Rolling Window Mean Absolute Error')
        axes[1, 1].set_xlabel('Position')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 误差箱线图
        error_ranges = pd.cut(self.df['True_Value'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        error_by_range = [self.df[error_ranges == label]['Abs_Error'].values for label in error_ranges.cat.categories]
        bp = axes[1, 2].boxplot(error_by_range, labels=error_ranges.cat.categories, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'yellow', 'orange', 'red']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        axes[1, 2].set_title('Error Distribution by True Value Range')
        axes[1, 2].set_xlabel('True Value Range')
        axes[1, 2].set_ylabel('Absolute Error')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('mlp_statistical_analysis.png', dpi=300, bbox_inches='tight')
        print("📄 统计分析图表已保存至: mlp_statistical_analysis.png")
        plt.show()
    
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
                colorbar=dict(title="Absolute Error")
            ),
            text=[f'Position: {pos}<br>True: {true:.2f}<br>Predicted: {pred:.2f}<br>Error: {err:.2f}' 
                  for pos, true, pred, err in zip(self.df['Position'], self.df['True_Value'], 
                                                 self.df['Predicted_Value'], self.df['Error'])],
            hovertemplate='%{text}<extra></extra>',
            name='Prediction Results'
        ))
        
        fig.update_layout(
            title='MLP Prediction Results 3D Visualization Analysis',
            scene=dict(
                xaxis_title='Time Position',
                yaxis_title='True Values',
                zaxis_title='Predicted Values',
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
            ),
            height=600
        )
        
        fig.write_html("mlp_3d_analysis.html")
        print("📄 3D可视化分析已保存至: mlp_3d_analysis.html")
    
    def create_model_diagnosis(self):
        """创建模型诊断分析"""
        print("\n🔍 深度模型诊断分析:")
        
        # 分析预测值的分布问题
        pred_range = self.df['Predicted_Value'].max() - self.df['Predicted_Value'].min()
        true_range = self.df['True_Value'].max() - self.df['True_Value'].min()
        
        print(f"   📈 预测范围问题:")
        print(f"      • 真实值范围: {true_range:.4f}")
        print(f"      • 预测值范围: {pred_range:.4f}")
        print(f"      • 范围比例: {pred_range/true_range:.4f}")
        
        if pred_range/true_range < 0.5:
            print(f"      ⚠️  严重问题: 预测值范围过窄，模型学习不充分")
            print(f"      💡 建议: 增加训练轮数、调整学习率或检查梯度消失问题")
        
        # 分析偏移问题
        pred_mean = self.df['Predicted_Value'].mean()
        true_mean = self.df['True_Value'].mean()
        bias = pred_mean - true_mean
        
        print(f"\n   📊 预测偏移分析:")
        print(f"      • 真实值均值: {true_mean:.4f}")
        print(f"      • 预测值均值: {pred_mean:.4f}")
        print(f"      • 系统性偏移: {bias:.4f}")
        
        if abs(bias) > true_mean * 0.1:
            print(f"      ⚠️  问题: 存在显著系统性偏移")
            print(f"      💡 建议: 检查数据预处理、权重初始化或学习率设置")
        
        # 归一化问题分析
        if self.has_normalized_data:
            norm_pred_std = self.df['Normalized_Pred'].std()
            norm_true_std = self.df['Normalized_True'].std()
            
            print(f"\n   🔬 归一化空间分析:")
            print(f"      • 归一化真实值标准差: {norm_true_std:.4f}")
            print(f"      • 归一化预测值标准差: {norm_pred_std:.4f}")
            print(f"      • 方差比例: {norm_pred_std/norm_true_std:.4f}")
            
            if norm_pred_std/norm_true_std < 0.3:
                print(f"      ⚠️  严重问题: 预测值在归一化空间中方差过小")
                print(f"      💡 可能原因: 梯度消失、学习率过小、或激活函数饱和")
                
        # 提供改进建议
        print(f"\n   💡 模型改进建议:")
        if self.r2 < 0.3:
            print(f"      🔧 架构调整: 考虑增加隐藏层维度或层数")
            print(f"      📚 数据增强: 检查数据质量和特征工程")
            print(f"      ⚙️  超参数: 尝试不同的学习率和优化器")
        
        if abs(bias) > true_mean * 0.05:
            print(f"      🎯 偏移修正: 考虑添加偏置修正或调整损失函数")
            
        if pred_range/true_range < 0.5:
            print(f"      📈 激活改进: 检查激活函数选择和权重初始化方案")
    
    def create_time_series_plot(self):
        """创建时间序列曲线图"""
        fig = go.Figure()
        
        # 添加真实值曲线
        fig.add_trace(go.Scatter(
            x=self.df['Position'],
            y=self.df['True_Value'],
            mode='lines+markers',
            name='True Values',
            line=dict(color='blue', width=2),
            marker=dict(size=4),
            hovertemplate='Position: %{x}<br>True Value: %{y:.2f}<extra></extra>'
        ))
        
        # 添加预测值曲线
        fig.add_trace(go.Scatter(
            x=self.df['Position'],
            y=self.df['Predicted_Value'],
            mode='lines+markers',
            name='Predicted Values',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=4),
            hovertemplate='Position: %{x}<br>Predicted Value: %{y:.2f}<extra></extra>'
        ))
        
        # 添加误差填充区域
        fig.add_trace(go.Scatter(
            x=self.df['Position'],
            y=self.df['True_Value'] + self.df['Error'],
            fill=None,
            mode='lines',
            line_color='rgba(0,100,80,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df['Position'],
            y=self.df['True_Value'] - self.df['Error'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,80,0)',
            name='Error Band',
            fillcolor='rgba(255,0,0,0.1)'
        ))
        
        fig.update_layout(
            title='Time Series: True Values vs Predicted Values',
            xaxis_title='Time Position',
            yaxis_title='Values',
            hovermode='x unified',
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        fig.write_html("mlp_time_series.html")
        print("📄 时间序列曲线图已保存至: mlp_time_series.html")

    def create_detailed_comparison_plot(self):
        """创建详细对比图表"""
        # 创建多子图布局
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Time Series Comparison',
                'Prediction Error Over Time',
                'Rolling Average Comparison',
                'Prediction Confidence Analysis'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. 时间序列对比
        fig.add_trace(
            go.Scatter(
                x=self.df['Position'],
                y=self.df['True_Value'],
                mode='lines',
                name='True Values',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.df['Position'],
                y=self.df['Predicted_Value'],
                mode='lines',
                name='Predicted Values',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # 2. 误差随时间变化
        fig.add_trace(
            go.Scatter(
                x=self.df['Position'],
                y=self.df['Error'],
                mode='lines',
                name='Prediction Error',
                line=dict(color='orange', width=1.5),
                fill='tonexty'
            ),
            row=1, col=2
        )
        
        # 添加零线
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
        
        # 3. 滑动平均对比
        window_size = max(10, len(self.df) // 30)
        rolling_true = pd.Series(self.df['True_Value']).rolling(window=window_size).mean()
        rolling_pred = pd.Series(self.df['Predicted_Value']).rolling(window=window_size).mean()
        
        fig.add_trace(
            go.Scatter(
                x=self.df['Position'],
                y=rolling_true,
                mode='lines',
                name=f'True Values (MA-{window_size})',
                line=dict(color='darkblue', width=3)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.df['Position'],
                y=rolling_pred,
                mode='lines',
                name=f'Predicted Values (MA-{window_size})',
                line=dict(color='darkred', width=3)
            ),
            row=2, col=1
        )
        
        # 4. 预测置信度分析
        abs_errors = self.df['Abs_Error']
        confidence_levels = pd.cut(abs_errors, bins=5, labels=['Very High', 'High', 'Medium', 'Low', 'Very Low'])
        confidence_counts = confidence_levels.value_counts()
        
        fig.add_trace(
            go.Bar(
                x=confidence_counts.index,
                y=confidence_counts.values,
                name='Prediction Confidence',
                marker_color=['green', 'lightgreen', 'yellow', 'orange', 'red']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Detailed MLP Prediction Analysis",
            showlegend=True
        )
        
        fig.write_html("mlp_detailed_comparison.html")
        print("📄 详细对比分析已保存至: mlp_detailed_comparison.html")

    def create_matplotlib_plots(self):
        """创建matplotlib静态图表"""
        # 创建大图表
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('MLP Neural Network Prediction Analysis', fontsize=16, fontweight='bold')
        
        # 1. 时间序列对比
        axes[0, 0].plot(self.df['Position'], self.df['True_Value'], 
                       'b-', linewidth=2, label='True Values', alpha=0.8)
        axes[0, 0].plot(self.df['Position'], self.df['Predicted_Value'], 
                       'r--', linewidth=2, label='Predicted Values', alpha=0.8)
        axes[0, 0].fill_between(self.df['Position'], 
                               self.df['True_Value'], 
                               self.df['Predicted_Value'],
                               alpha=0.3, color='gray', label='Error Area')
        axes[0, 0].set_title('Time Series: True vs Predicted Values')
        axes[0, 0].set_xlabel('Time Position')
        axes[0, 0].set_ylabel('Values')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 误差时间序列
        axes[0, 1].plot(self.df['Position'], self.df['Error'], 
                       'g-', linewidth=1.5, alpha=0.7)
        axes[0, 1].fill_between(self.df['Position'], self.df['Error'], 
                               alpha=0.3, color='green')
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.8)
        axes[0, 1].set_title('Prediction Error Over Time')
        axes[0, 1].set_xlabel('Time Position')
        axes[0, 1].set_ylabel('Error')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 散点图与拟合线
        axes[1, 0].scatter(self.df['True_Value'], self.df['Predicted_Value'], 
                          alpha=0.6, s=30, c=self.df['Abs_Error'], 
                          cmap='viridis')
        
        # 添加完美预测线
        min_val = min(self.df['True_Value'].min(), self.df['Predicted_Value'].min())
        max_val = max(self.df['True_Value'].max(), self.df['Predicted_Value'].max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 
                       'r--', linewidth=2, label='Perfect Prediction')
        
        # 添加拟合线
        z = np.polyfit(self.df['True_Value'], self.df['Predicted_Value'], 1)
        p = np.poly1d(z)
        axes[1, 0].plot(self.df['True_Value'], p(self.df['True_Value']), 
                       'orange', linewidth=2, label=f'Fit Line (R²={self.r2:.3f})')
        
        axes[1, 0].set_title('True vs Predicted Values (Scatter)')
        axes[1, 0].set_xlabel('True Values')
        axes[1, 0].set_ylabel('Predicted Values')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 预测准确度分布
        accuracy_ranges = ['<1%', '1-3%', '3-5%', '5-10%', '>10%']
        accuracy_counts = [
            sum(self.df['Relative_Error'].abs() < 1),
            sum((self.df['Relative_Error'].abs() >= 1) & (self.df['Relative_Error'].abs() < 3)),
            sum((self.df['Relative_Error'].abs() >= 3) & (self.df['Relative_Error'].abs() < 5)),
            sum((self.df['Relative_Error'].abs() >= 5) & (self.df['Relative_Error'].abs() < 10)),
            sum(self.df['Relative_Error'].abs() >= 10)
        ]
        
        colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
        bars = axes[1, 1].bar(accuracy_ranges, accuracy_counts, color=colors, alpha=0.7)
        
        # 添加数值标签
        for bar, count in zip(bars, accuracy_counts):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{count}\n({count/len(self.df)*100:.1f}%)',
                           ha='center', va='bottom')
        
        axes[1, 1].set_title('Prediction Accuracy Distribution')
        axes[1, 1].set_xlabel('Relative Error Range')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('mlp_comprehensive_curves.png', dpi=300, bbox_inches='tight')
        print("📄 综合曲线分析图已保存至: mlp_comprehensive_curves.png")
        plt.show()

def main():
    """主函数"""
    print("🚀 开始MLP结果可视化分析...")
    print("=" * 50)
    
    try:
        # 创建可视化器
        visualizer = MLPResultsVisualizer()
        
        # 生成综合仪表板
        print("\n📊 生成综合分析仪表板...")
        visualizer.create_comprehensive_dashboard()
        
        # 生成时间序列曲线图
        print("\n📈 生成时间序列曲线图...")
        visualizer.create_time_series_plot()
        
        # 生成详细对比图表
        print("\n🔍 生成详细对比分析...")
        visualizer.create_detailed_comparison_plot()
        
        # 生成matplotlib静态图表
        print("\n📊 生成matplotlib静态图表...")
        visualizer.create_matplotlib_plots()
        
        # 生成3D分析
        print("\n🎯 生成3D可视化分析...")
        visualizer.create_3d_analysis()
        
        # 模型诊断分析
        print("\n🔬 执行模型诊断分析...")
        visualizer.create_model_diagnosis()
        
        # 生成统计分析
        print("\n📈 生成统计分析图表...")
        visualizer.create_statistical_analysis()
        
        print("\n" + "=" * 50)
        print("✅ 可视化分析完成!")
        print("📄 生成的文件:")
        print("   • mlp_comprehensive_analysis.html - 交互式综合分析仪表板")
        print("   • mlp_time_series.html - 时间序列曲线图")
        print("   • mlp_detailed_comparison.html - 详细对比分析")
        print("   • mlp_comprehensive_curves.png - 综合曲线分析图")
        print("   • mlp_3d_analysis.html - 3D立体可视化分析")
        print("   • mlp_statistical_analysis.png - 详细统计分析图表")
        print("\n💡 提示: 在浏览器中打开HTML文件查看交互式图表")
        print("\n🎯 根据诊断结果，建议优化模型架构和训练参数")
        
    except Exception as e:
        print(f"\n❌ 可视化分析过程中发生错误: {e}")
        print("请检查数据文件是否存在且格式正确")

if __name__ == "__main__":
    main()
