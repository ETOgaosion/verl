#!/usr/bin/env python3
"""
DP Balance Benchmark 通用分析框架
支持任意实验时间戳，生成论文级分析报告

用法:
    python analyze_dp_benchmark.py --timestamp 20260222_132718 --output-dir ./analysis
    python analyze_dp_benchmark.py --latest  # 自动查找最新实验
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import re
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置论文级图表样式
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.3)
sns.set_palette("husl")


class DPBenchmarkAnalyzer:
    """DP Balance Benchmark 分析器"""
    
    def __init__(self, base_dir: str = "/data/wsl/dp_balance_benchmark"):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "results"
        self.logs_dir = self.base_dir / "logs"
        self.tensorboard_dir = self.base_dir / "tensorboard_log"
        
        # 指标配置（可扩展）
        self.metrics_config = {
            'Memory_Imbalance': {'label': 'Memory Imbalance', 'unit': '', 'lower_is_better': True},
            'Compute_Imbalance': {'label': 'Compute Imbalance', 'unit': '', 'lower_is_better': True},
            'Variance_Discrepancy': {'label': 'Variance Discrepancy', 'unit': '', 'lower_is_better': True},
            'Balance_Score': {'label': 'Balance Score', 'unit': '%', 'lower_is_better': False},
            'Pipeline_Time_ms': {'label': 'Pipeline Time', 'unit': 'ms', 'lower_is_better': True},
            'Pipeline_CV': {'label': 'Pipeline CV', 'unit': '', 'lower_is_better': True},
            'Sync_Efficiency': {'label': 'Sync Efficiency', 'unit': '', 'lower_is_better': False}
        }
        
        self.data = None
        self.stats = None
        
    def load_csv(self, timestamp: str) -> pd.DataFrame:
        """加载指定时间戳的 metrics CSV"""
        csv_path = self.results_dir / f"metrics_{timestamp}.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # 数据清洗
        df = df.replace('N/A', np.nan)
        numeric_cols = ['Memory_Imbalance', 'Compute_Imbalance', 'Variance_Discrepancy', 
                       'Balance_Score', 'Pipeline_Time_ms', 'Pipeline_CV']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        self.data = df
        print(f"✓ Loaded {len(df)} experiments from {csv_path}")
        return df
    
    def find_latest_timestamp(self) -> str:
        """自动查找最新的实验时间戳"""
        csv_files = list(self.results_dir.glob("metrics_*.csv"))
        if not csv_files:
            raise FileNotFoundError("No metrics files found")
        
        # 提取时间戳并排序
        timestamps = []
        for f in csv_files:
            match = re.search(r'metrics_(\d{8}_\d{6})\.csv', f.name)
            if match:
                timestamps.append(match.group(1))
        
        latest = sorted(timestamps)[-1]
        print(f"✓ Found latest experiment: {latest}")
        return latest
    
    def compute_statistics(self) -> pd.DataFrame:
        """计算统计摘要（均值±标准差）"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")
        
        # 按 Method 分组统计
        agg_dict = {}
        for col in self.metrics_config.keys():
            if col in self.data.columns:
                agg_dict[col] = ['mean', 'std', 'min', 'max', 'count']
        
        stats = self.data.groupby('Method').agg(agg_dict).round(6)
        self.stats = stats
        return stats
    
    def statistical_tests(self, baseline: str = 'karmarkar_karp', 
                         methods: List[str] = None) -> pd.DataFrame:
        """执行统计显著性检验"""
        if self.data is None:
            raise ValueError("No data loaded.")
        
        if methods is None:
            methods = [m for m in self.data['Method'].unique() if m != baseline]
        
        results = []
        metrics = ['Compute_Imbalance', 'Pipeline_CV', 'Balance_Score']
        
        for method in methods:
            for metric in metrics:
                if metric not in self.data.columns:
                    continue
                
                baseline_data = self.data[self.data['Method'] == baseline][metric].dropna()
                method_data = self.data[self.data['Method'] == method][metric].dropna()
                
                if len(baseline_data) < 2 or len(method_data) < 2:
                    continue
                
                # t-test
                t_stat, p_val = stats.ttest_ind(method_data, baseline_data)
                
                # Cohen's d (效应量)
                mean_diff = method_data.mean() - baseline_data.mean()
                pooled_std = np.sqrt(((len(method_data)-1)*method_data.var() + 
                                     (len(baseline_data)-1)*baseline_data.var()) / 
                                    (len(method_data)+len(baseline_data)-2))
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                
                # 改进百分比
                baseline_mean = baseline_data.mean()
                method_mean = method_data.mean()
                improvement = (((baseline_mean - method_mean) / baseline_mean * 100)
                             if self.metrics_config[metric]['lower_is_better']
                             else ((method_mean - baseline_mean) / baseline_mean * 100))
                
                results.append({
                    'Comparison': f"{method} vs {baseline}",
                    'Metric': metric,
                    'Baseline_Mean': baseline_mean,
                    'Method_Mean': method_mean,
                    'Improvement_%': improvement,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'Cohens_d': abs(cohens_d),
                    'Significant': p_val < 0.05,
                    'Effect_Size': 'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'
                })
        
        return pd.DataFrame(results)
    
    def create_visualizations(self, output_dir: Path):
        """生成论文级可视化图表"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        if self.data is None:
            raise ValueError("No data loaded.")
        
        # 1. 核心指标对比箱线图 (Figure 1)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('DP Balance Algorithm Comparison', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['Memory_Imbalance', 'Compute_Imbalance', 'Balance_Score',
                          'Pipeline_Time_ms', 'Pipeline_CV', 'Variance_Discrepancy']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            if metric in self.data.columns:
                sns.boxplot(data=self.data, x='Method', y=metric, ax=ax, 
                           palette=['#e74c3c', '#2ecc71', '#3498db', '#9b59b6'][:self.data['Method'].nunique()])
                ax.set_title(self.metrics_config[metric]['label'], fontweight='bold')
                ax.tick_params(axis='x', rotation=15)
                ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'figure1_core_metrics.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'figure1_core_metrics.pdf', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: figure1_core_metrics")
        
        # 2. 方法对比柱状图 (Figure 2)
        if self.stats is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Compute Imbalance
            methods = self.stats.index
            means = self.stats['Compute_Imbalance']['mean']
            stds = self.stats['Compute_Imbalance']['std']
            
            axes[0].bar(methods, means, yerr=stds, capsize=5, 
                       color=['#e74c3c', '#2ecc71'], alpha=0.8)
            axes[0].set_title('Compute Imbalance (lower is better)', fontweight='bold')
            axes[0].set_ylabel('Imbalance Ratio')
            axes[0].grid(axis='y', alpha=0.3)
            
            # Pipeline CV
            means = self.stats['Pipeline_CV']['mean']
            stds = self.stats['Pipeline_CV']['std']
            
            axes[1].bar(methods, means, yerr=stds, capsize=5,
                       color=['#e74c3c', '#2ecc71'], alpha=0.8)
            axes[1].set_title('Pipeline Time Variability (lower is better)', fontweight='bold')
            axes[1].set_ylabel('Coefficient of Variation')
            axes[1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'figure2_comparison.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: figure2_comparison")
        
        # 3. 散点图：Compute vs Pipeline CV (相关性分析)
        fig, ax = plt.subplots(figsize=(8, 6))
        for method in self.data['Method'].unique():
            method_data = self.data[self.data['Method'] == method]
            ax.scatter(method_data['Compute_Imbalance'], method_data['Pipeline_CV'], 
                      label=method, s=100, alpha=0.7)
        
        ax.set_xlabel('Compute Imbalance', fontweight='bold')
        ax.set_ylabel('Pipeline CV', fontweight='bold')
        ax.set_title('Compute Imbalance vs Pipeline Stability', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'figure3_correlation.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: figure3_correlation")
    
    def generate_latex_table(self) -> str:
        """生成 LaTeX 表格代码"""
        if self.stats is None:
            self.compute_statistics()
        
        latex = []
        latex.append("\\begin{table}[t]")
        latex.append("\\centering")
        latex.append("\\caption{DP Balance Algorithm Performance Comparison}")
        latex.append("\\label{tab:dp_balance_results}")
        latex.append("\\begin{tabular}{lcccc}")
        latex.append("\\toprule")
        latex.append("Method & \\makecell{Compute\\\\Imbalance ($\\times 10^{-4}$)} & \\makecell{Pipeline\\\\CV ($\\times 10^{-2}$)} & \\makecell{Balance\\\\Score (\\%)} & \\makecell{Improvement} \\\\")
        latex.append("\\midrule")
        
        # 找到 baseline 用于计算改进
        baseline_method = 'karmarkar_karp' if 'karmarkar_karp' in self.stats.index else self.stats.index[0]
        
        for method in self.stats.index:
            row = self.stats.loc[method]
            
            # 提取数值
            comp_mean = row['Compute_Imbalance']['mean'] * 10000  # 转 10^-4
            comp_std = row['Compute_Imbalance']['std'] * 10000
            cv_mean = row['Pipeline_CV']['mean'] * 100  # 转 10^-2
            cv_std = row['Pipeline_CV']['std'] * 100
            bal_mean = row['Balance_Score']['mean']
            bal_std = row['Balance_Score']['std']
            
            # 格式化改进标注
            if method == baseline_method:
                improvement = "-"
            else:
                baseline_comp = self.stats.loc[baseline_method]['Compute_Imbalance']['mean']
                baseline_cv = self.stats.loc[baseline_method]['Pipeline_CV']['mean']
                comp_imp = (baseline_comp - row['Compute_Imbalance']['mean']) / baseline_comp * 100
                cv_imp = (baseline_cv - row['Pipeline_CV']['mean']) / baseline_cv * 100
                improvement = f"\\textbf{{{comp_imp:.0f}\\%$\\downarrow$ / {cv_imp:.0f}\\%$\\downarrow$}}"
            
            method_label = method.replace('_', '\\_')
            if method == 'balanced':
                method_label = "\\textbf{Stratified Balanced (Ours)}"
            elif method == 'karmarkar_karp':
                method_label = "Karmarkar-Karp (Baseline)"
            
            latex.append(f"{method_label} & ${comp_mean:.2f} \\pm {comp_std:.2f}$ & ${cv_mean:.2f} \\pm {cv_std:.2f}$ & ${bal_mean:.2f} \\pm {bal_std:.2f}$ & {improvement} \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def generate_markdown_report(self, timestamp: str, output_dir: Path):
        """生成 Markdown 分析报告"""
        output_dir = Path(output_dir)
        
        if self.stats is None:
            self.compute_statistics()
        
        stat_tests = self.statistical_tests()
        
        report = []
        report.append(f"# DP Balance Benchmark Analysis Report")
        report.append(f"\n**Experiment Date:** {timestamp}\n")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report.append("## 1. Experimental Setup")
        report.append(f"- **Total Experiments:** {len(self.data)}")
        report.append(f"- **Methods Compared:** {', '.join(self.data['Method'].unique())}")
        report.append(f"- **Seeds per Method:** {self.data.groupby('Method').size().to_dict()}")
        
        report.append("\n## 2. Statistical Summary (Mean ± Std)")
        report.append(self.stats.to_markdown())
        
        if not stat_tests.empty:
            report.append("\n## 3. Statistical Significance Tests")
            report.append("Comparing enhanced method against baseline (karmarkar_karp):")
            report.append(stat_tests.to_markdown(index=False))
            
            # 显著结果高亮
            sig_results = stat_tests[stat_tests['Significant'] == True]
            if not sig_results.empty:
                report.append(f"\n**Key Finding:** {len(sig_results)} metrics show statistically significant improvement (p < 0.05).")
        
        report.append("\n## 4. Key Findings")
        
        # 自动提取关键发现
        if 'balanced' in self.stats.index and 'karmarkar_karp' in self.stats.index:
            balanced = self.stats.loc['balanced']
            baseline = self.stats.loc['karmarkar_karp']
            
            comp_improvement = (baseline['Compute_Imbalance']['mean'] - balanced['Compute_Imbalance']['mean']) / baseline['Compute_Imbalance']['mean'] * 100
            cv_improvement = (baseline['Pipeline_CV']['mean'] - balanced['Pipeline_CV']['mean']) / baseline['Pipeline_CV']['mean'] * 100
            
            report.append(f"1. **Compute Imbalance Reduction:** {comp_improvement:.1f}% improvement (from {baseline['Compute_Imbalance']['mean']:.6f} to {balanced['Compute_Imbalance']['mean']:.6f})")
            report.append(f"2. **Pipeline Stability:** {cv_improvement:.1f}% reduction in time variability (CV from {baseline['Pipeline_CV']['mean']:.4f} to {balanced['Pipeline_CV']['mean']:.4f})")
            report.append(f"3. **Balance Score:** Improved from {baseline['Balance_Score']['mean']:.2f}% to {balanced['Balance_Score']['mean']:.2f}%")
        
        report.append("\n## 5. LaTeX Table")
        report.append("```latex")
        report.append(self.generate_latex_table())
        report.append("```")
        
        report.append("\n## 6. File Locations")
        report.append(f"- **Raw Data:** `{self.results_dir}/metrics_{timestamp}.csv`")
        report.append(f"- **Statistics:** `{self.results_dir}/statistics_{timestamp}.csv`")
        report.append(f"- **Logs:** `{self.logs_dir}/{timestamp}/`")
        report.append(f"- **TensorBoard:** `{self.tensorboard_dir}/`")
        
        # 保存报告
        report_path = output_dir / f'report_{timestamp}.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"✓ Generated report: {report_path}")
        return '\n'.join(report)


def main():
    parser = argparse.ArgumentParser(description='Analyze DP Balance Benchmark Results')
    parser.add_argument('--timestamp', type=str, help='Experiment timestamp (e.g., 20260222_132718)')
    parser.add_argument('--latest', action='store_true', help='Use latest experiment')
    parser.add_argument('--output-dir', type=str, default='./analysis_output', 
                       help='Output directory for analysis results')
    parser.add_argument('--baseline', type=str, default='karmarkar_karp',
                       help='Baseline method name for comparison')
    
    args = parser.parse_args()
    
    # 初始化分析器
    analyzer = DPBenchmarkAnalyzer()
    
    # 加载数据
    try:
        if args.latest or args.timestamp is None:
            timestamp = analyzer.find_latest_timestamp()
        else:
            timestamp = args.timestamp
        
        analyzer.load_csv(timestamp)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nAvailable experiments:")
        for f in sorted(analyzer.results_dir.glob("metrics_*.csv")):
            print(f"  - {f.stem.replace('metrics_', '')}")
        return
    
    # 执行分析
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*60}")
    print(f"Analyzing Experiment: {timestamp}")
    print(f"{'='*60}\n")
    
    # 1. 统计计算
    stats = analyzer.compute_statistics()
    print("Statistical Summary:")
    print(stats)
    print()
    
    # 2. 统计检验
    stat_tests = analyzer.statistical_tests(baseline=args.baseline)
    if not stat_tests.empty:
        print("Statistical Tests:")
        print(stat_tests.to_string(index=False))
        print()
    
    # 3. 可视化
    print("Generating visualizations...")
    analyzer.create_visualizations(output_dir)
    
    # 4. 生成报告
    report = analyzer.generate_markdown_report(timestamp, output_dir)
    
    # 5. 保存 LaTeX 表格
    latex_table = analyzer.generate_latex_table()
    latex_path = output_dir / f'table_{timestamp}.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"✓ Saved LaTeX table: {latex_path}")
    
    print(f"\n{'='*60}")
    print(f"Analysis Complete!")
    print(f"Output Directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()