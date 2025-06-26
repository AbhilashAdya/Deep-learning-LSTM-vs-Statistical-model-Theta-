# ================================
# HOW TO IMPORT FROM PREPROCESSING NOTEBOOK
# ================================

# Option 1: Convert preprocessing notebook to .py script
# Run this in terminal:
# jupyter nbconvert --to script data_preprocessing.ipynb

# Then import in data_exploration.ipynb:
# from data_preprocessing import load_data, clean_data, scale_data

# ================================
# Option 2: Load saved processed data (Recommended)
# ================================

# In data_preprocessing.ipynb - save processed data:
"""
# At the end of preprocessing notebook:
scaled_data.to_csv('../data/scaled_data.csv', index=False)
print("✅ Processed data saved!")
"""

# In data_exploration.ipynb - load processed data:
import pandas as pd
import numpy as np

# Load preprocessed data
scaled_data = pd.read_csv('../data/scaled_data.csv')
scaled_data['datetime'] = pd.to_datetime(scaled_data['datetime'])

print("✅ Preprocessed data loaded!")
print(f"Data shape: {scaled_data.shape}")
print(f"Countries: {scaled_data['country'].unique()}")

# ================================
# EDA STRUCTURE AND THINGS TO DO
# ================================

"""
COMPREHENSIVE EDA CHECKLIST FOR TIME SERIES COVID DATA
"""

# ================================
# 1. DATA OVERVIEW
# ================================

def basic_data_overview(data):
    """Basic exploration of the dataset"""
    
    print("📊 BASIC DATA OVERVIEW")
    print("="*50)
    
    # Dataset info
    print(f"Dataset shape: {data.shape}")
    print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
    print(f"Number of countries: {data['country'].nunique()}")
    print(f"Countries: {data['country'].unique().tolist()}")
    
    # Missing values
    print(f"\n🔍 Missing Values:")
    missing = data.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found ✅")
    
    # Basic statistics
    print(f"\n📈 Target Variable Statistics:")
    print(data['new_cases'].describe())
    
    return data.info()

# ================================
# 2. TIME SERIES ANALYSIS
# ================================

def time_series_analysis(data):
    """Analyze time series patterns"""
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("📈 TIME SERIES ANALYSIS")
    print("="*50)
    
    # Overall time series plot
    plt.figure(figsize=(15, 8))
    
    for country in data['country'].unique():
        country_data = data[data['country'] == country]
        plt.plot(country_data['datetime'], country_data['new_cases'], 
                label=country, linewidth=2, alpha=0.8)
    
    plt.title('COVID-19 New Cases Over Time by Country', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('New Cases (Scaled)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Individual country analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, country in enumerate(data['country'].unique()):
        if i < len(axes):
            country_data = data[data['country'] == country]
            axes[i].plot(country_data['datetime'], country_data['new_cases'], 
                        color='steelblue', linewidth=2)
            axes[i].set_title(f'{country} - Cases Over Time', fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# ================================
# 3. STATISTICAL ANALYSIS
# ================================

def statistical_analysis(data):
    """Statistical analysis of the data"""
    
    print("📊 STATISTICAL ANALYSIS")
    print("="*50)
    
    # Country-wise statistics
    country_stats = data.groupby('country')['new_cases'].agg([
        'count', 'mean', 'std', 'min', 'max', 'skew'
    ]).round(4)
    
    print("Country-wise Statistics:")
    print(country_stats)
    
    # Distribution analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Overall distribution
    axes[0,0].hist(data['new_cases'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Distribution of New Cases (All Countries)')
    axes[0,0].set_xlabel('New Cases (Scaled)')
    axes[0,0].set_ylabel('Frequency')
    
    # Box plot by country
    data.boxplot(column='new_cases', by='country', ax=axes[0,1])
    axes[0,1].set_title('New Cases Distribution by Country')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # QQ plot
    from scipy import stats
    stats.probplot(data['new_cases'], dist="norm", plot=axes[1,0])
    axes[1,0].set_title('Q-Q Plot (Normal Distribution)')
    
    # Correlation heatmap (if multiple numeric columns)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = data[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
        axes[1,1].set_title('Correlation Matrix')
    
    plt.tight_layout()
    plt.show()
    
    return country_stats

# ================================
# 4. SEASONAL PATTERNS
# ================================

def seasonal_analysis(data):
    """Analyze seasonal patterns"""
    
    print("🌟 SEASONAL PATTERNS ANALYSIS")
    print("="*50)
    
    # Add time-based features
    data_copy = data.copy()
    data_copy['day_of_week'] = data_copy['datetime'].dt.day_name()
    data_copy['month'] = data_copy['datetime'].dt.month_name()
    data_copy['week_of_year'] = data_copy['datetime'].dt.isocalendar().week
    
    # Day of week patterns
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Day of week analysis
    dow_stats = data_copy.groupby('day_of_week')['new_cases'].mean()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_stats = dow_stats.reindex(day_order)
    
    axes[0,0].bar(dow_stats.index, dow_stats.values, color='lightcoral')
    axes[0,0].set_title('Average Cases by Day of Week')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Monthly patterns
    monthly_stats = data_copy.groupby('month')['new_cases'].mean()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_stats = monthly_stats.reindex(month_order)
    
    axes[0,1].bar(monthly_stats.index, monthly_stats.values, color='lightgreen')
    axes[0,1].set_title('Average Cases by Month')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Weekly patterns over time
    weekly_data = data_copy.groupby(['week_of_year', 'country'])['new_cases'].mean().reset_index()
    
    for country in data_copy['country'].unique():
        country_weekly = weekly_data[weekly_data['country'] == country]
        axes[1,0].plot(country_weekly['week_of_year'], country_weekly['new_cases'], 
                      label=country, alpha=0.8, linewidth=2)
    
    axes[1,0].set_title('Weekly Patterns by Country')
    axes[1,0].set_xlabel('Week of Year')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Heatmap of country vs day of week
    heatmap_data = data_copy.pivot_table(values='new_cases', 
                                        index='country', 
                                        columns='day_of_week', 
                                        aggfunc='mean')
    heatmap_data = heatmap_data.reindex(columns=day_order)
    
    sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', ax=axes[1,1])
    axes[1,1].set_title('Cases Heatmap: Country vs Day of Week')
    
    plt.tight_layout()
    plt.show()
    
    return dow_stats, monthly_stats

# ================================
# 5. TIME SERIES DECOMPOSITION (Your existing code)
# ================================

def perform_decomposition_analysis(data):
    """Perform time series decomposition for each country"""
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    import matplotlib.dates as mdates
    
    print("🔬 TIME SERIES DECOMPOSITION")
    print("="*50)
    
    for country in data['country'].unique():
        print(f"\n📈 Decomposing time series for {country}")
        
        country_data = data[data['country'] == country].sort_values('datetime')
        country_data = country_data.set_index('datetime')
        
        # Perform decomposition
        additive_decomposition = seasonal_decompose(
            country_data['new_cases'], 
            model='additive', 
            period=7
        )
        
        # Plot decomposition
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        additive_decomposition.observed.plot(ax=axes[0], color='blue', linewidth=1.5)
        axes[0].set_title('Original Time Series', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Cases', fontweight='bold')
        
        additive_decomposition.trend.plot(ax=axes[1], color='red', linewidth=2)
        axes[1].set_title('Trend', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Trend', fontweight='bold')
        
        additive_decomposition.seasonal.plot(ax=axes[2], color='green', linewidth=1.5)
        axes[2].set_title('Seasonal (Weekly Pattern)', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Seasonal', fontweight='bold')
        
        additive_decomposition.resid.plot(ax=axes[3], color='orange', linewidth=1)
        axes[3].set_title('Residual (Random Noise)', fontsize=14, fontweight='bold')
        axes[3].set_ylabel('Residual', fontweight='bold')
        axes[3].set_xlabel('Date', fontweight='bold', fontsize=14)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Time Series Decomposition - {country}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

# ================================
# 6. COMPARATIVE ANALYSIS
# ================================

def comparative_analysis(data):
    """Compare patterns across countries"""
    
    print("🌍 COMPARATIVE ANALYSIS")
    print("="*50)
    
    # Peak analysis
    peak_analysis = data.groupby('country')['new_cases'].agg([
        'max', 'idxmax', 'min', 'idxmin'
    ])
    
    # Add peak dates
    peak_dates = []
    min_dates = []
    
    for country in data['country'].unique():
        country_data = data[data['country'] == country]
        max_idx = peak_analysis.loc[country, 'idxmax']
        min_idx = peak_analysis.loc[country, 'idxmin']
        
        peak_date = country_data.loc[max_idx, 'datetime']
        min_date = country_data.loc[min_idx, 'datetime']
        
        peak_dates.append(peak_date)
        min_dates.append(min_date)
    
    peak_analysis['peak_date'] = peak_dates
    peak_analysis['min_date'] = min_dates
    
    print("Peak Analysis by Country:")
    print(peak_analysis)
    
    # Correlation between countries
    pivot_data = data.pivot(index='datetime', columns='country', values='new_cases')
    country_corr = pivot_data.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(country_corr, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Cross-Country Correlation of COVID Cases')
    plt.tight_layout()
    plt.show()
    
    return peak_analysis, country_corr

# ================================
# COMPLETE EDA FUNCTION
# ================================

def run_complete_eda(data):
    """Run complete EDA pipeline"""
    
    print("🚀 COMPREHENSIVE EDA PIPELINE")
    print("="*70)
    
    # 1. Basic overview
    basic_data_overview(data)
    
    # 2. Time series analysis
    time_series_analysis(data)
    
    # 3. Statistical analysis
    stats = statistical_analysis(data)
    
    # 4. Seasonal patterns
    dow_stats, monthly_stats = seasonal_analysis(data)
    
    # 5. Decomposition
    perform_decomposition_analysis(data)
    
    # 6. Comparative analysis
    peak_analysis, country_corr = comparative_analysis(data)
    
    print("\n✅ EDA COMPLETED!")
    print("="*70)
    
    return {
        'country_stats': stats,
        'day_of_week_patterns': dow_stats,
        'monthly_patterns': monthly_stats,
        'peak_analysis': peak_analysis,
        'country_correlations': country_corr
    }

# ================================
# USAGE EXAMPLE
# ================================

"""
# In your data_exploration.ipynb:

# 1. Load preprocessed data
scaled_data = pd.read_csv('../data/scaled_data.csv')
scaled_data['datetime'] = pd.to_datetime(scaled_data['datetime'])

# 2. Run complete EDA
eda_results = run_complete_eda(scaled_data)

# 3. Access specific results
print("Country Statistics:")
print(eda_results['country_stats'])

print("Day of Week Patterns:")
print(eda_results['day_of_week_patterns'])
"""