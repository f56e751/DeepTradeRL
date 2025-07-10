import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import time

warnings.filterwarnings('ignore')


class TrainingMetricsCallback(BaseCallback):
    """
    Callback to collect training metrics for visualization
    """
    def __init__(self, verbose=0):
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.learning_rates = []
        self.timesteps = []
        self.episodes = []
        
        # Temporary storage for current episode
        self.current_episode_reward = 0
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Collect episode rewards from the environment info
        infos = self.locals.get('infos', [])
        if infos and len(infos) > 0:
            info = infos[0]
            if 'episode' in info:
                # Episode finished, collect the total reward
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                self.episode_rewards.append(episode_reward)
                self.episodes.append(self.episode_count)
                self.timesteps.append(self.num_timesteps)
                self.episode_count += 1
                
                if self.verbose >= 1:
                    print(f"Episode {self.episode_count}: Reward = {episode_reward:.4f}, Length = {episode_length}")
            
        return True
    
    def _on_rollout_end(self) -> None:
        # Collect training metrics from logger
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            # Get the latest training metrics
            log_dict = self.model.logger.name_to_value
            
            if 'train/policy_gradient_loss' in log_dict:
                self.policy_losses.append(log_dict['train/policy_gradient_loss'])
            elif 'train/policy_loss' in log_dict:
                self.policy_losses.append(log_dict['train/policy_loss'])
                
            if 'train/value_loss' in log_dict:
                self.value_losses.append(log_dict['train/value_loss'])
                
            if 'train/entropy_loss' in log_dict:
                self.entropy_losses.append(log_dict['train/entropy_loss'])
            elif 'train/entropy' in log_dict:
                self.entropy_losses.append(-log_dict['train/entropy'])  # Negative entropy as loss
                
            if 'train/learning_rate' in log_dict:
                self.learning_rates.append(log_dict['train/learning_rate'])
    
    def get_metrics(self):
        """Return collected metrics"""
        return {
            'episode_rewards': self.episode_rewards,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses,
            'learning_rates': self.learning_rates,
            'timesteps': self.timesteps,
            'episodes': self.episodes
        }


def classify_action(action, hold_threshold=0.2, h_max=250):
    """
    Classify continuous action into Buy/Hold/Sell based on environment logic
    
    Args:
        action: Raw action from model (-1 to 1)
        hold_threshold: Threshold for hold action
        h_max: Maximum action multiplier
    
    Returns:
        str: 'Buy', 'Hold', or 'Sell'
    """
    act = float(np.clip(action, -1.0, 1.0))
    
    # Hold 판정: abs(act) < hold_threshold
    if abs(act) < hold_threshold:
        real_act = 0  # Hold
    else:
        real_act = int(np.rint(act * h_max))
    
    # 최종 행동 분류
    if real_act < 0:
        return 'Sell'
    elif real_act > 0:
        return 'Buy'
    else:
        return 'Hold'


def calculate_action_statistics(actions, hold_threshold=0.2, h_max=250):
    """
    Calculate action distribution statistics
    
    Args:
        actions: List of raw actions from model
        hold_threshold: Hold threshold from environment
        h_max: Max action multiplier from environment
        
    Returns:
        dict: Action statistics including counts and percentages
    """
    action_types = [classify_action(action, hold_threshold, h_max) for action in actions]
    
    buy_count = action_types.count('Buy')
    hold_count = action_types.count('Hold')
    sell_count = action_types.count('Sell')
    total_count = len(action_types)
    
    return {
        'action_types': action_types,
        'buy_count': buy_count,
        'hold_count': hold_count,
        'sell_count': sell_count,
        'total_count': total_count,
        'buy_ratio': buy_count / total_count if total_count > 0 else 0,
        'hold_ratio': hold_count / total_count if total_count > 0 else 0,
        'sell_ratio': sell_count / total_count if total_count > 0 else 0,
        'raw_actions': actions
    }


def plot_action_analysis(action_stats, dataset_name, save_path=None):
    """
    Create comprehensive action analysis visualization
    
    Args:
        action_stats: Action statistics from calculate_action_statistics
        dataset_name: Name of dataset for titles
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Action Analysis - {dataset_name}', fontsize=16, fontweight='bold')
    
    # Color scheme
    colors = {
        'Buy': '#2E8B57',    # Green
        'Hold': '#FFD700',   # Gold  
        'Sell': '#DC143C'    # Red
    }
    
    # 1. Action Distribution Pie Chart
    ax1 = axes[0, 0]
    sizes = [action_stats['buy_ratio'], action_stats['hold_ratio'], action_stats['sell_ratio']]
    labels = ['Buy', 'Hold', 'Sell']
    colors_pie = [colors['Buy'], colors['Hold'], colors['Sell']]
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                      colors=colors_pie, startangle=90)
    ax1.set_title('Action Distribution', fontweight='bold')
    
    # Make percentage text bold and larger
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    # 2. Action Counts Bar Chart
    ax2 = axes[0, 1]
    counts = [action_stats['buy_count'], action_stats['hold_count'], action_stats['sell_count']]
    bars = ax2.bar(labels, counts, color=[colors['Buy'], colors['Hold'], colors['Sell']], alpha=0.8)
    ax2.set_title('Action Counts', fontweight='bold')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Action Time Series
    ax3 = axes[0, 2]
    action_types = action_stats['action_types']
    action_numeric = []
    for action in action_types:
        if action == 'Buy':
            action_numeric.append(1)
        elif action == 'Sell':
            action_numeric.append(-1)
        else:  # Hold
            action_numeric.append(0)
    
    # Create color map for time series
    color_map = []
    for action in action_types:
        color_map.append(colors[action])
    
    ax3.scatter(range(len(action_numeric)), action_numeric, c=color_map, alpha=0.6, s=10)
    ax3.set_title('Actions Over Time', fontweight='bold')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Action Type')
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(['Sell', 'Hold', 'Buy'])
    ax3.grid(True, alpha=0.3)
    
    # 4. Raw Action Distribution
    ax4 = axes[1, 0]
    raw_actions = action_stats['raw_actions']
    ax4.hist(raw_actions, bins=50, alpha=0.7, color='skyblue', density=True, edgecolor='black')
    ax4.axvline(np.mean(raw_actions), color='red', linestyle='--', 
               label=f'Mean: {np.mean(raw_actions):.3f}')
    ax4.axvline(np.median(raw_actions), color='orange', linestyle='--', 
               label=f'Median: {np.median(raw_actions):.3f}')
    ax4.set_title('Raw Action Values Distribution', fontweight='bold')
    ax4.set_xlabel('Raw Action Value')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Running Action Ratios
    ax5 = axes[1, 1]
    window_size = max(50, len(action_types) // 20)  # Adaptive window size
    
    if len(action_types) > window_size:
        running_buy = []
        running_hold = []
        running_sell = []
        
        for i in range(window_size, len(action_types) + 1):
            window_actions = action_types[i-window_size:i]
            buy_ratio = window_actions.count('Buy') / window_size
            hold_ratio = window_actions.count('Hold') / window_size
            sell_ratio = window_actions.count('Sell') / window_size
            
            running_buy.append(buy_ratio)
            running_hold.append(hold_ratio)
            running_sell.append(sell_ratio)
        
        x_range = range(window_size, len(action_types) + 1)
        ax5.plot(x_range, running_buy, label='Buy', color=colors['Buy'], linewidth=2)
        ax5.plot(x_range, running_hold, label='Hold', color=colors['Hold'], linewidth=2)
        ax5.plot(x_range, running_sell, label='Sell', color=colors['Sell'], linewidth=2)
        ax5.set_title(f'Running Action Ratios (Window: {window_size})', fontweight='bold')
        ax5.set_xlabel('Time Step')
        ax5.set_ylabel('Ratio')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Insufficient data for\nrunning ratios', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Running Action Ratios', fontweight='bold')
    
    # 6. Action Statistics Table
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    stats_data = [
        ['Action', 'Count', 'Ratio', 'Percentage'],
        ['Buy', f'{action_stats["buy_count"]}', f'{action_stats["buy_ratio"]:.3f}', f'{action_stats["buy_ratio"]*100:.1f}%'],
        ['Hold', f'{action_stats["hold_count"]}', f'{action_stats["hold_ratio"]:.3f}', f'{action_stats["hold_ratio"]*100:.1f}%'],
        ['Sell', f'{action_stats["sell_count"]}', f'{action_stats["sell_ratio"]:.3f}', f'{action_stats["sell_ratio"]*100:.1f}%'],
        ['Total', f'{action_stats["total_count"]}', '1.000', '100.0%']
    ]
    
    table = ax6.table(cellText=stats_data[1:], colLabels=stats_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.2, 0.2, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style the table
    for i in range(len(stats_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code action rows
    for i in range(1, len(stats_data) - 1):  # Exclude total row
        action_name = stats_data[i][0]
        if action_name in colors:
            for j in range(len(stats_data[0])):
                table[(i, j)].set_facecolor(colors[action_name])
                table[(i, j)].set_alpha(0.3)
    
    # Style total row
    for j in range(len(stats_data[0])):
        table[(len(stats_data) - 1, j)].set_facecolor('#E0E0E0')
        table[(len(stats_data) - 1, j)].set_text_props(weight='bold')
    
    ax6.set_title('Action Statistics Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Action analysis saved as '{save_path}'")
    
    plt.show()
    
    # Print action insights
    print(f"\n🎯 Action Analysis Insights - {dataset_name}")
    print("="*50)
    print(f"Most frequent action: {max(labels, key=lambda x: action_stats[f'{x.lower()}_count'])}")
    print(f"Action diversity: {1 - max(sizes):.3f} (closer to 1 = more diverse)")
    
    if action_stats['buy_ratio'] > 0.4:
        print("📈 Aggressive buying strategy detected")
    elif action_stats['sell_ratio'] > 0.4:
        print("📉 Aggressive selling strategy detected")
    elif action_stats['hold_ratio'] > 0.6:
        print("⏸️  Conservative holding strategy detected")
    else:
        print("⚖️  Balanced trading strategy")
    
    print(f"Raw action statistics:")
    print(f"  Mean: {np.mean(raw_actions):.4f}")
    print(f"  Std: {np.std(raw_actions):.4f}")
    print(f"  Min: {np.min(raw_actions):.4f}")
    print(f"  Max: {np.max(raw_actions):.4f}")


def calculate_financial_metrics(step_rewards, cumulative_rewards, portfolio_values, initial_cash=100000):
    """
    Calculate comprehensive financial performance metrics using actual portfolio values
    
    Args:
        step_rewards: Step-wise rewards from environment
        cumulative_rewards: Cumulative rewards 
        portfolio_values: Actual portfolio values from get_portfolio_value()
        initial_cash: Initial cash amount
    """
    step_rewards = np.array(step_rewards)
    cumulative_rewards = np.array(cumulative_rewards)
    portfolio_values = np.array(portfolio_values)
    
    if len(portfolio_values) <= 1:
        return None
    
    # Calculate returns based on actual portfolio values
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    returns = returns[~np.isnan(returns)]  # Remove NaN values
    
    if len(returns) == 0:
        return None
    
    # Basic Performance Metrics using actual portfolio values
    total_return = (portfolio_values[-1] - initial_cash) / initial_cash
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1  # Assuming daily data
    volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
    
    # Maximum Drawdown (MDD) based on actual portfolio values
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # Sharpe Ratio (assuming risk-free rate = 0)
    sharpe_ratio = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
    
    # Calmar Ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win/Loss Analysis - Fixed to use portfolio returns instead of step rewards
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    
    win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
    avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
    avg_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0
    profit_factor = abs(np.sum(positive_returns) / np.sum(negative_returns)) if np.sum(negative_returns) != 0 else np.inf
    
    # Additional Metrics
    sortino_ratio = calculate_sortino_ratio(returns)
    var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
    cvar_95 = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else 0
    
    # Skewness and Kurtosis
    skewness = stats.skew(returns) if len(returns) > 3 else 0
    kurtosis_val = stats.kurtosis(returns) if len(returns) > 3 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'sortino_ratio': sortino_ratio,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'skewness': skewness,
        'kurtosis': kurtosis_val,
        'portfolio_values': portfolio_values,
        'returns': returns,
        'drawdown': drawdown,
        'step_rewards': step_rewards
    }


def calculate_sortino_ratio(returns, target_return=0):
    """Calculate Sortino Ratio"""
    excess_returns = returns - target_return
    downside_returns = excess_returns[excess_returns < 0]
    downside_deviation = np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 0
    
    if downside_deviation == 0:
        return 0
    
    return (np.mean(excess_returns) * 252) / (downside_deviation * np.sqrt(252))


def plot_comprehensive_financial_analysis(metrics, dataset_name, action_stats=None, save_path=None):
    """
    Create comprehensive financial performance visualization with optional action analysis
    """
    if metrics is None:
        print("⚠️ No metrics available for visualization")
        return
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # Color scheme
    colors = {
        'primary': '#2E8B57',
        'secondary': '#FF6B6B', 
        'accent': '#4ECDC4',
        'warning': '#FFE66D',
        'danger': '#FF8E53'
    }
    
    # 1. Portfolio Value Over Time (using actual portfolio values)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(metrics['portfolio_values'], linewidth=2, color=colors['primary'], label='Portfolio Value')
    initial_value = metrics['portfolio_values'][0]
    ax1.axhline(y=initial_value, color='gray', linestyle='--', alpha=0.7, label='Initial Value')
    ax1.set_title(f'{dataset_name} - Portfolio Value Over Time (Actual)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add annotations for key points
    max_value = np.max(metrics['portfolio_values'])
    min_value = np.min(metrics['portfolio_values'])
    max_idx = np.argmax(metrics['portfolio_values'])
    min_idx = np.argmin(metrics['portfolio_values'])
    
    ax1.annotate(f'Peak: ${max_value:,.0f}', 
                xy=(max_idx, max_value), xytext=(max_idx, max_value + (max_value-min_value)*0.1),
                arrowprops=dict(arrowstyle='->', color=colors['primary']), fontsize=10)
    ax1.annotate(f'Trough: ${min_value:,.0f}', 
                xy=(min_idx, min_value), xytext=(min_idx, min_value - (max_value-min_value)*0.1),
                arrowprops=dict(arrowstyle='->', color=colors['danger']), fontsize=10)
    
    # 2. Drawdown Analysis (항상 MDD 분석 표시)
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.fill_between(range(len(metrics['drawdown'])), metrics['drawdown'], 0, 
                     alpha=0.7, color=colors['danger'], label='Drawdown')
    ax2.axhline(y=metrics['max_drawdown'], color='red', linestyle='--', 
                label=f'Max Drawdown: {metrics["max_drawdown"]:.2%}')
    ax2.set_title(f'{dataset_name} - Drawdown Analysis', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Returns Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    if len(metrics['returns']) > 0:
        ax3.hist(metrics['returns'], bins=50, alpha=0.7, color=colors['accent'], density=True)
        ax3.axvline(np.mean(metrics['returns']), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(metrics["returns"]):.4f}')
        ax3.axvline(np.median(metrics['returns']), color='orange', linestyle='--', 
                   label=f'Median: {np.median(metrics["returns"]):.4f}')
    ax3.set_title('Returns Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Returns')
    ax3.set_ylabel('Density')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Step Rewards Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    positive_rewards = metrics['step_rewards'][metrics['step_rewards'] > 0]
    negative_rewards = metrics['step_rewards'][metrics['step_rewards'] < 0]
    
    ax4.hist(positive_rewards, bins=30, alpha=0.7, color=colors['primary'], label='Positive', density=True)
    ax4.hist(negative_rewards, bins=30, alpha=0.7, color=colors['danger'], label='Negative', density=True)
    ax4.set_title('Step Rewards Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Step Reward')
    ax4.set_ylabel('Density')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Risk-Return Scatter (Rolling)
    ax5 = fig.add_subplot(gs[1, 2])
    if len(metrics['returns']) > 30:
        window = 30
        rolling_returns = pd.Series(metrics['returns']).rolling(window).mean()
        rolling_vol = pd.Series(metrics['returns']).rolling(window).std()
        
        scatter = ax5.scatter(rolling_vol, rolling_returns, c=range(len(rolling_vol)), 
                             cmap='viridis', alpha=0.6, s=30)
        ax5.set_title('Risk-Return Profile (Rolling 30)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Volatility')
        ax5.set_ylabel('Return')
        ax5.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax5, label='Time')
    
    # 6. Performance Metrics Table
    ax6 = fig.add_subplot(gs[1, 3])
    ax6.axis('off')
    
    metrics_data = [
        ['Metric', 'Value'],
        ['Total Return', f'{metrics["total_return"]:.2%}'],
        ['Annualized Return', f'{metrics["annualized_return"]:.2%}'],
        ['Volatility', f'{metrics["volatility"]:.2%}'],
        ['Sharpe Ratio', f'{metrics["sharpe_ratio"]:.3f}'],
        ['Sortino Ratio', f'{metrics["sortino_ratio"]:.3f}'],
        ['Max Drawdown', f'{metrics["max_drawdown"]:.2%}'],
        ['Calmar Ratio', f'{metrics["calmar_ratio"]:.3f}'],
        ['Win Rate', f'{metrics["win_rate"]:.2%}'],
        ['Profit Factor', f'{metrics["profit_factor"]:.2f}' if metrics["profit_factor"] != np.inf else '∞'],
    ]
    
    table = ax6.table(cellText=metrics_data[1:], colLabels=metrics_data[0],
                     cellLoc='center', loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Color code the table
    for i in range(len(metrics_data[0])):
        table[(0, i)].set_facecolor(colors['primary'])
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code performance metrics
    for i in range(1, len(metrics_data)):
        if 'Return' in metrics_data[i][0] or 'Ratio' in metrics_data[i][0]:
            value_str = metrics_data[i][1].rstrip('%').replace('∞', '999')
            try:
                value = float(value_str)
                if value > 0:
                    table[(i, 1)].set_facecolor('#E8F5E8')
                else:
                    table[(i, 1)].set_facecolor('#FFE8E8')
            except:
                pass
    
    ax6.set_title('Performance Metrics', fontsize=12, fontweight='bold')
    
    # 7. Win/Loss Analysis
    ax7 = fig.add_subplot(gs[2, 0])
    win_loss_data = [metrics['win_rate'], 1 - metrics['win_rate']]
    labels = ['Wins', 'Losses']
    colors_pie = [colors['primary'], colors['danger']]
    
    wedges, texts, autotexts = ax7.pie(win_loss_data, labels=labels, autopct='%1.1f%%', 
                                      colors=colors_pie, startangle=90)
    ax7.set_title('Win/Loss Ratio', fontsize=12, fontweight='bold')
    
    # 8. Monthly Returns Heatmap (if enough data)
    ax8 = fig.add_subplot(gs[2, 1:])
    if len(metrics['returns']) > 60:  # At least 2 months of data
        # Create synthetic monthly data for demonstration
        monthly_returns = []
        days_per_month = 21  # Approximate trading days per month
        
        for i in range(0, len(metrics['returns']), days_per_month):
            month_returns = metrics['returns'][i:i+days_per_month]
            if len(month_returns) > 0:
                monthly_returns.append(np.sum(month_returns))
        
        if len(monthly_returns) >= 3:
            # Reshape for heatmap
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            years = list(range(2019, 2019 + (len(monthly_returns) // 12) + 1))
            
            # Create matrix
            heatmap_data = np.zeros((len(years), 12))
            heatmap_data[:] = np.nan
            
            for i, ret in enumerate(monthly_returns[:len(years)*12]):
                year_idx = i // 12
                month_idx = i % 12
                if year_idx < len(years):
                    heatmap_data[year_idx, month_idx] = ret
            
            # Create heatmap
            im = ax8.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
            ax8.set_xticks(range(12))
            ax8.set_xticklabels(months)
            ax8.set_yticks(range(len(years)))
            ax8.set_yticklabels(years)
            ax8.set_title('Monthly Returns Heatmap', fontsize=12, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax8)
            cbar.set_label('Monthly Return')
            
            # Add text annotations
            for i in range(len(years)):
                for j in range(12):
                    if not np.isnan(heatmap_data[i, j]):
                        text = ax8.text(j, i, f'{heatmap_data[i, j]:.2%}',
                                       ha="center", va="center", color="black", fontsize=8)
    else:
        ax8.text(0.5, 0.5, 'Insufficient data for\nmonthly analysis', 
                ha='center', va='center', transform=ax8.transAxes, fontsize=12)
        ax8.set_title('Monthly Returns Heatmap', fontsize=12, fontweight='bold')
    
    # 9. Risk Metrics
    ax9 = fig.add_subplot(gs[3, 0])
    risk_metrics = ['VaR 95%', 'CVaR 95%', 'Skewness', 'Kurtosis']
    risk_values = [metrics['var_95'], metrics['cvar_95'], 
                   metrics['skewness'], metrics['kurtosis']]
    
    bars = ax9.bar(risk_metrics, risk_values, color=[colors['warning'], colors['danger'], 
                                                    colors['accent'], colors['secondary']])
    ax9.set_title('Risk Metrics', fontsize=12, fontweight='bold')
    ax9.set_ylabel('Value')
    ax9.tick_params(axis='x', rotation=45)
    ax9.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, risk_values):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 10. Performance Summary
    ax10 = fig.add_subplot(gs[3, 1:])
    ax10.axis('off')
    
    profit_factor_str = "∞" if metrics['profit_factor'] == np.inf else f"{metrics['profit_factor']:.2f}"
    
    # Create performance summary text
    summary_text = f"""
    📊 PERFORMANCE SUMMARY - {dataset_name}
    
    🎯 Overall Performance (Actual Portfolio Values):
    • Total Return: {metrics['total_return']:.2%}
    • Annualized Return: {metrics['annualized_return']:.2%}
    • Volatility: {metrics['volatility']:.2%}
    
    📈 Risk-Adjusted Returns:
    • Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
    • Sortino Ratio: {metrics['sortino_ratio']:.3f}
    • Calmar Ratio: {metrics['calmar_ratio']:.3f}
    
    📉 Risk Management:
    • Maximum Drawdown: {metrics['max_drawdown']:.2%}
    • Win Rate: {metrics['win_rate']:.2%}
    • Profit Factor: {profit_factor_str}
    
    🎲 Distribution Characteristics:
    • Skewness: {metrics['skewness']:.3f}
    • Kurtosis: {metrics['kurtosis']:.3f}
    • VaR 95%: {metrics['var_95']:.4f}
    """
    
    if action_stats:
        summary_text += f"""
    
    🎯 Action Distribution:
    • Buy: {action_stats['buy_ratio']:.1%}
    • Hold: {action_stats['hold_ratio']:.1%}
    • Sell: {action_stats['sell_ratio']:.1%}
        """
    
    ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['accent'], alpha=0.1))
    
    # Overall title
    fig.suptitle(f'Comprehensive Financial Analysis - {dataset_name} (Portfolio-Based)', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comprehensive financial analysis saved as '{save_path}'")
    
    plt.tight_layout()
    plt.show()
    
    return metrics


def plot_training_reward_curves(training_metrics, save_directory, args=None):
    """
    Plot training reward curves and loss curves
    """
    # Create subplot layout: 2 rows, 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress Visualization', fontsize=16, fontweight='bold')
    
    # Plot 1: Episode Rewards
    if training_metrics.get('episode_rewards'):
        episodes = training_metrics['episodes']
        episode_rewards = training_metrics['episode_rewards']
        
        axes[0, 0].plot(episodes, episode_rewards, color='blue', linewidth=2, alpha=0.7)
        
        # Add moving average if we have enough data points
        if len(episode_rewards) > 10:
            window_size = min(50, len(episode_rewards) // 5)
            moving_avg = pd.Series(episode_rewards).rolling(window=window_size, center=True).mean()
            axes[0, 0].plot(episodes, moving_avg, color='red', linewidth=2, 
                           label=f'Moving Average (window={window_size})')
            axes[0, 0].legend()
        
        axes[0, 0].set_title('Episode Rewards During Training', fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Episode Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Print statistics
        print(f"📊 Training Episode Statistics:")
        print(f"Total episodes: {len(episode_rewards)}")
        print(f"Average episode reward: {np.mean(episode_rewards):.4f}")
        print(f"Episode reward std: {np.std(episode_rewards):.4f}")
        print(f"Min episode reward: {np.min(episode_rewards):.4f}")
        print(f"Max episode reward: {np.max(episode_rewards):.4f}")
    else:
        axes[0, 0].text(0.5, 0.5, 'No episode reward data collected\n(Training may be too short)', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Episode Rewards During Training', fontweight='bold')
    
    # Plot 2: Policy Loss
    if training_metrics.get('policy_losses'):
        update_steps = range(len(training_metrics['policy_losses']))
        axes[0, 1].plot(update_steps, training_metrics['policy_losses'], 
                       color='red', linewidth=2, alpha=0.7)
        axes[0, 1].set_title('Policy Loss During Training', fontweight='bold')
        axes[0, 1].set_xlabel('Training Update')
        axes[0, 1].set_ylabel('Policy Loss')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No policy loss data collected', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Policy Loss During Training', fontweight='bold')
    
    # Plot 3: Value Loss
    if training_metrics.get('value_losses'):
        update_steps = range(len(training_metrics['value_losses']))
        axes[1, 0].plot(update_steps, training_metrics['value_losses'], 
                       color='orange', linewidth=2, alpha=0.7)
        axes[1, 0].set_title('Value Loss During Training', fontweight='bold')
        axes[1, 0].set_xlabel('Training Update')
        axes[1, 0].set_ylabel('Value Loss')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No value loss data collected', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Value Loss During Training', fontweight='bold')
    
    # Plot 4: Entropy Loss
    if training_metrics.get('entropy_losses'):
        update_steps = range(len(training_metrics['entropy_losses']))
        axes[1, 1].plot(update_steps, training_metrics['entropy_losses'], 
                       color='purple', linewidth=2, alpha=0.7)
        axes[1, 1].set_title('Entropy Loss During Training', fontweight='bold')
        axes[1, 1].set_xlabel('Training Update')
        axes[1, 1].set_ylabel('Entropy Loss')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No entropy loss data collected', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Entropy Loss During Training', fontweight='bold')
    
    plt.tight_layout()
    training_curves_filename = f'runs/{save_directory}/training_curves.png'
    plt.savefig(training_curves_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📈 Training curves saved as '{training_curves_filename}'")
    
    # 🔍 Loss 분석 및 해석
    print(f"\n🔍 Training Loss Analysis:")
    if training_metrics.get('policy_losses'):
        policy_losses = training_metrics['policy_losses']
        print(f"Policy Loss - Initial: {policy_losses[0]:.6f}, Final: {policy_losses[-1]:.6f}")
        print(f"Policy Loss - Mean: {np.mean(policy_losses):.6f}, Std: {np.std(policy_losses):.6f}")
        
        # 정책 손실의 변화 패턴 분석
        if len(policy_losses) > 10:
            first_half_mean = np.mean(policy_losses[:len(policy_losses)//2])
            second_half_mean = np.mean(policy_losses[len(policy_losses)//2:])
            if abs(first_half_mean - second_half_mean) < 0.001:
                print("⚠️  Policy loss has converged - training may need more exploration or different hyperparameters")
    
    if training_metrics.get('value_losses'):
        value_losses = training_metrics['value_losses']
        print(f"Value Loss - Initial: {value_losses[0]:.2f}, Final: {value_losses[-1]:.2f}")
        print(f"Value Loss - Mean: {np.mean(value_losses):.2f}, Std: {np.std(value_losses):.2f}")
        
        # 가치 손실이 너무 높으면 문제가 있을 수 있음
        if np.mean(value_losses) > 1000:
            print("⚠️  Value loss is very high - check reward scaling or value function architecture")
    
    print(f"\n💡 Loss Analysis Insights:")
    print(f"• PPO policy loss naturally fluctuates (not always decreasing)")
    print(f"• Value loss spikes indicate value function is learning complex patterns")
    print(f"• If losses plateau early, consider:")
    if args:
        print(f"  - Increasing learning rate ({args.lr})")
    else:
        print(f"  - Increasing learning rate (current)")
    print(f"  - Adjusting entropy coefficient for more exploration")
    print(f"  - Checking if reward signal is informative enough")
    print(f"  - Ensuring sufficient training steps per episode")


def plot_reward_curves(step_rewards, cumulative_rewards, num_episodes, dataset_name):
    """
    Plot reward curves for visualization (original function, unchanged)
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    for episode in range(num_episodes):
        episode_steps = range(len(step_rewards[episode]))
        
        # Plot step-wise rewards
        axes[0].plot(episode_steps, step_rewards[episode], 
                    label=f'Episode {episode + 1}', alpha=0.7)
        axes[0].set_title(f'{dataset_name} - Step-wise Rewards', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Reward')
        axes[0].grid(True, alpha=0.3)
        if num_episodes <= 5:  # Only show legend if not too many episodes
            axes[0].legend()
        
        # Plot cumulative rewards
        axes[1].plot(episode_steps, cumulative_rewards[episode], 
                    label=f'Episode {episode + 1}', linewidth=2)
        axes[1].set_title(f'{dataset_name} - Cumulative Rewards', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Cumulative Reward')
        axes[1].grid(True, alpha=0.3)
        if num_episodes <= 5:
            axes[1].legend()
    
    # Add moving average for step-wise rewards if there's only one episode
    if num_episodes == 1:
        rewards = step_rewards[0]
        if len(rewards) > 50:
            window_size = min(50, len(rewards) // 10)
            moving_avg = pd.Series(rewards).rolling(window=window_size, center=True).mean()
            axes[0].plot(range(len(rewards)), moving_avg, 
                        color='red', linewidth=2, 
                        label=f'Moving Average (window={window_size})')
            axes[0].legend()
    
    plt.tight_layout()
    filename = f'reward_curves_{dataset_name.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📈 {dataset_name} reward curves saved as '{filename}'")
    
    # Additional statistics for single episode
    if num_episodes == 1:
        rewards = step_rewards[0]
        print(f"\n📊 {dataset_name} Step-wise Reward Statistics")
        print(f"Total steps: {len(rewards)}")
        print(f"Average step reward: {np.mean(rewards):.4f}")
        print(f"Step reward std: {np.std(rewards):.4f}")
        print(f"Min step reward: {np.min(rewards):.4f}")
        print(f"Max step reward: {np.max(rewards):.4f}")
        print(f"Final cumulative reward: {cumulative_rewards[0][-1]:.4f}")


def compare_performance(val_results, test_results, save_directory):
    """
    Compare validation and test performance
    """
    print("\n" + "="*50)
    print("📊 PERFORMANCE COMPARISON")
    print("="*50)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Average rewards comparison
    datasets = ['Validation', 'Test']
    avg_rewards = [val_results['avg_reward'], test_results['avg_reward']]
    std_rewards = [val_results['std_reward'], test_results['std_reward']]
    
    axes[0, 0].bar(datasets, avg_rewards, yerr=std_rewards, capsize=5, 
                   color=['skyblue', 'lightcoral'], alpha=0.7)
    axes[0, 0].set_title('Average Reward Comparison', fontweight='bold')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (avg, std) in enumerate(zip(avg_rewards, std_rewards)):
        axes[0, 0].text(i, avg + std + 0.01, f'{avg:.3f}', 
                       ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Portfolio values comparison (if single episode)
    if len(val_results['portfolio_values']) == 1 and len(test_results['portfolio_values']) == 1:
        val_portfolio = val_results['portfolio_values'][0]
        test_portfolio = test_results['portfolio_values'][0]
        
        axes[0, 1].plot(val_portfolio, label='Validation', linewidth=2, color='skyblue')
        axes[0, 1].plot(test_portfolio, label='Test', linewidth=2, color='lightcoral')
        axes[0, 1].set_title('Portfolio Values Over Time', fontweight='bold')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Portfolio Value ($)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Step rewards distribution
    val_step_rewards = np.concatenate(val_results['step_rewards'])
    test_step_rewards = np.concatenate(test_results['step_rewards'])
    
    axes[1, 0].hist(val_step_rewards, bins=50, alpha=0.7, label='Validation', 
                    color='skyblue', density=True)
    axes[1, 0].hist(test_step_rewards, bins=50, alpha=0.7, label='Test', 
                    color='lightcoral', density=True)
    axes[1, 0].set_title('Step Rewards Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Step Reward')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Performance metrics table
    axes[1, 1].axis('off')
    
    # Create performance summary table
    metrics_data = [
        ['Metric', 'Validation', 'Test', 'Difference'],
        ['Avg Reward', f'{val_results["avg_reward"]:.4f}', 
         f'{test_results["avg_reward"]:.4f}', 
         f'{test_results["avg_reward"] - val_results["avg_reward"]:.4f}'],
        ['Std Reward', f'{val_results["std_reward"]:.4f}', 
         f'{test_results["std_reward"]:.4f}', 
         f'{test_results["std_reward"] - val_results["std_reward"]:.4f}'],
        ['Min Reward', f'{np.min(val_results["episode_rewards"]):.4f}', 
         f'{np.min(test_results["episode_rewards"]):.4f}', 
         f'{np.min(test_results["episode_rewards"]) - np.min(val_results["episode_rewards"]):.4f}'],
        ['Max Reward', f'{np.max(val_results["episode_rewards"]):.4f}', 
         f'{np.max(test_results["episode_rewards"]):.4f}', 
         f'{np.max(test_results["episode_rewards"]) - np.max(val_results["episode_rewards"]):.4f}']
    ]
    
    table = axes[1, 1].table(cellText=metrics_data[1:], colLabels=metrics_data[0],
                            cellLoc='center', loc='center',
                            colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(metrics_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1, 1].set_title('Performance Metrics Comparison', fontweight='bold', pad=20)
    
    plt.tight_layout()
    comparison_filename = f'performance_comparison.png'
    plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Performance comparison saved as '{comparison_filename}'")
    
    # Print detailed comparison
    print(f"\n📈 Detailed Performance Analysis:")
    print(f"Validation avg reward: {val_results['avg_reward']:.4f} ± {val_results['std_reward']:.4f}")
    print(f"Test avg reward: {test_results['avg_reward']:.4f} ± {test_results['std_reward']:.4f}")
    
    diff = test_results['avg_reward'] - val_results['avg_reward']
    if diff > 0:
        print(f"✅ Test performance is {diff:.4f} better than validation")
    else:
        print(f"⚠️  Test performance is {abs(diff):.4f} worse than validation")
    
    # Check for overfitting
    if val_results['avg_reward'] > test_results['avg_reward']:
        print("⚠️  Potential overfitting detected (validation > test performance)")
    else:
        print("✅ No obvious overfitting (test >= validation performance)")


def evaluate_model(model, env, num_episodes=1, dataset_name="", initial_cash=100000):
    """
    Enhanced evaluation with comprehensive financial metrics and action analysis
    """
    print(f"\n📈 Enhanced Evaluation on {dataset_name} Dataset")
    print("-" * 50)
    
    all_episode_rewards = []
    all_step_rewards = []
    all_cumulative_rewards = []
    all_portfolio_values = []
    all_actions = []  # Track raw actions for analysis
    
    # Get environment parameters for action classification
    hold_threshold = getattr(env, 'hold_threshold', 0.2)
    h_max = getattr(env, 'h_max', 250)
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_step_rewards = []
        episode_cumulative_rewards = []
        episode_portfolio_values = []
        episode_actions = []
        done = False
        step = 0
        
        print(f"Episode {episode + 1}/{num_episodes} - {dataset_name}")
        
        # 초기 포트폴리오 가치 기록
        mid_price = env.get_price()
        initial_portfolio_value = env.inventory.get_portfolio_value({'TICKER': mid_price})
        episode_portfolio_values.append(initial_portfolio_value)
        
        # Use tqdm for progress bar
        pbar = tqdm(desc=f"Steps", unit="step")
        
        while not done:
            # Use the model to predict the next action (evaluation mode)
            action, _states = model.predict(obs, deterministic=True)
            
            # Record raw action for analysis
            raw_action = float(action[0]) if hasattr(action, '__len__') else float(action)
            episode_actions.append(raw_action)
            
            # Take the action in the environment
            obs, reward, done, info = env.step(action)
            
            # Record step rewards
            episode_step_rewards.append(reward)
            episode_reward += reward
            episode_cumulative_rewards.append(episode_reward)
            
            # Record actual portfolio value using mid price
            mid_price = env.get_price()
            portfolio_value = env.inventory.get_portfolio_value({'TICKER': mid_price})
            episode_portfolio_values.append(portfolio_value)
            
            step += 1
            pbar.update(1)
            pbar.set_postfix({
                'Reward': f'{reward:.4f}', 
                'Cumulative': f'{episode_reward:.4f}',
                'Portfolio': f'${portfolio_value:,.0f}',
                'Action': f'{raw_action:.3f}'
            })
        
        pbar.close()
        
        all_episode_rewards.append(episode_reward)
        all_step_rewards.append(episode_step_rewards)
        all_cumulative_rewards.append(episode_cumulative_rewards)
        all_portfolio_values.append(episode_portfolio_values)
        all_actions.extend(episode_actions)  # Flatten all actions
        
        final_portfolio_value = episode_portfolio_values[-1]
        portfolio_return = (final_portfolio_value - initial_cash) / initial_cash
        
        print(f"✅ Episode {episode + 1} finished")
        print(f"   Total reward: {episode_reward:.4f}")
        print(f"   Portfolio value: ${final_portfolio_value:,.2f}")
        print(f"   Portfolio return: {portfolio_return:.2%}")
        print(f"   Total actions taken: {len(episode_actions)}")
    
    # Calculate action statistics
    print(f"\n🎯 Calculating action distribution...")
    action_stats = calculate_action_statistics(all_actions, hold_threshold, h_max)
    
    # Create action analysis visualization
    action_save_path = f'action_analysis_{dataset_name.lower()}.png'
    plot_action_analysis(action_stats, dataset_name, action_save_path)
    
    # Calculate comprehensive financial metrics using actual portfolio values
    print(f"\n🔬 Calculating comprehensive financial metrics using actual portfolio values...")
    
    # Use first episode data for detailed analysis
    step_rewards = all_step_rewards[0]
    cumulative_rewards = all_cumulative_rewards[0]
    portfolio_values = all_portfolio_values[0]
    
    # Calculate metrics with actual portfolio values
    metrics = calculate_financial_metrics(step_rewards, cumulative_rewards, portfolio_values, initial_cash)
    
    if metrics:
        # Create comprehensive visualization with action stats
        save_path = f'comprehensive_analysis_{dataset_name.lower()}.png'
        plot_comprehensive_financial_analysis(metrics, dataset_name, action_stats, save_path)
        
        # Print detailed metrics
        print(f"\n📊 Detailed Financial Metrics - {dataset_name} (Portfolio-Based)")
        print("="*70)
        print(f"💼 Portfolio Performance:")
        print(f"   Initial Value: ${initial_cash:,.2f}")
        print(f"   Final Value: ${portfolio_values[-1]:,.2f}")
        print(f"   Absolute P&L: ${portfolio_values[-1] - initial_cash:,.2f}")
        
        print(f"\n📈 Return Metrics:")
        print(f"   Total Return: {metrics['total_return']:.2%}")
        print(f"   Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"   Volatility: {metrics['volatility']:.2%}")
        
        print(f"\n🎯 Risk-Adjusted Performance:")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"   Sortino Ratio: {metrics['sortino_ratio']:.4f}")  
        print(f"   Calmar Ratio: {metrics['calmar_ratio']:.4f}")
        
        print(f"\n📉 Risk Analysis:")
        print(f"   Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"   VaR (95%): {metrics['var_95']:.4f}")
        print(f"   CVaR (95%): {metrics['cvar_95']:.4f}")
        
        print(f"\n🎲 Trading Performance:")
        print(f"   Win Rate: {metrics['win_rate']:.2%}")
        print(f"   Average Win: {metrics['avg_win']:.4f}")
        print(f"   Average Loss: {metrics['avg_loss']:.4f}")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}" if metrics['profit_factor'] != np.inf else "   Profit Factor: ∞")
        
        print(f"\n📊 Distribution Analysis:")
        print(f"   Skewness: {metrics['skewness']:.4f}")
        print(f"   Kurtosis: {metrics['kurtosis']:.4f}")
        
        # Performance interpretation
        print(f"\n💡 Performance Interpretation:")
        if metrics['sharpe_ratio'] > 1.0:
            print("   ✅ Excellent risk-adjusted performance (Sharpe > 1.0)")
        elif metrics['sharpe_ratio'] > 0.5:
            print("   ✅ Good risk-adjusted performance (Sharpe > 0.5)")
        else:
            print("   ⚠️  Poor risk-adjusted performance (Sharpe < 0.5)")
            
        if metrics['max_drawdown'] < -0.2:
            print("   ⚠️  High drawdown risk (MDD > 20%)")
        else:
            print("   ✅ Acceptable drawdown risk (MDD < 20%)")
            
        if metrics['win_rate'] > 0.5:
            print("   ✅ Positive win rate")
        else:
            print("   ⚠️  Low win rate - strategy may rely on few large wins")
    
    # Also create the original visualization
    plot_reward_curves(all_step_rewards, all_cumulative_rewards, num_episodes, dataset_name)
    
    # Print evaluation summary
    avg_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    
    print(f"\n📊 {dataset_name} Evaluation Summary")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Standard deviation: {std_reward:.4f}")
    print(f"Min reward: {np.min(all_episode_rewards):.4f}")
    print(f"Max reward: {np.max(all_episode_rewards):.4f}")
    
    return {
        'episode_rewards': all_episode_rewards,
        'step_rewards': all_step_rewards, 
        'cumulative_rewards': all_cumulative_rewards,
        'portfolio_values': all_portfolio_values,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'financial_metrics': metrics,
        'action_stats': action_stats  # 추가: 행동 통계
    }