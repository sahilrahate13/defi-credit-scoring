# DeFi Credit Scoring System for Aave V2 Protocol

## Overview

This system implements a machine learning-based credit scoring model for DeFi wallets interacting with the Aave V2 protocol. The model assigns credit scores between 0-1000 based on historical transaction behavior, where higher scores indicate reliable and responsible usage, while lower scores reflect risky, bot-like, or exploitative behavior.

## Architecture

### Core Components

1. **Data Processing Engine** (`DeFiCreditScorer` class)
   - JSON transaction data parser
   - Feature engineering pipeline
   - Data cleaning and normalization

2. **Machine Learning Pipeline**
   - Anomaly detection using Isolation Forest
   - Behavioral clustering using K-Means
   - Ensemble scoring methodology

3. **Analysis & Visualization**
   - Statistical analysis of score distributions
   - Risk behavior pattern identification
   - Comprehensive reporting system

## Methodology

### Feature Engineering

The system extracts and engineers 24+ features from raw transaction data:

#### 1. **Basic Transaction Metrics**
- `total_transactions`: Total number of transactions
- `unique_assets`: Number of different assets used
- `total_volume_usd`: Total transaction volume in USD
- `avg_transaction_size`: Average transaction size
- `account_age_days`: Age of the wallet in days

#### 2. **Action Distribution Features**
- Individual counts for: deposits, borrows, repays, redeems, liquidations
- Volume metrics for each action type
- Behavioral ratios (borrow-to-deposit, repay-to-borrow)

#### 3. **Risk Indicators**
- `liquidation_ratio`: Proportion of liquidation transactions
- `borrow_to_deposit_ratio`: Leverage indicator
- `repay_to_borrow_ratio`: Repayment consistency

#### 4. **Activity Patterns**
- `transaction_frequency`: Transactions per day
- `avg_time_between_tx`: Average time between transactions
- `tx_size_cv`: Transaction size coefficient of variation

#### 5. **Behavioral Scores (0-1 scale)**
- `reliability_score`: Based on repayment behavior and liquidations
- `activity_score`: Consistent usage over time
- `sophistication_score`: Diverse protocol usage
- `volume_score`: Normalized transaction volume

### Credit Score Calculation

The final credit score (0-1000) is calculated using a multi-step process:

1. **Base Score Calculation** (Weighted Average):
   ```
   Base Score = (reliability_score × 0.3 + 
                activity_score × 0.25 + 
                sophistication_score × 0.25 + 
                volume_score × 0.2) × 1000
   ```

2. **Anomaly Detection Penalty**:
   - Uses Isolation Forest to detect unusual behavior patterns
   - Applies 200-point penalty for anomalous wallets

3. **Cluster-based Adjustments**:
   - Groups wallets into 5 behavioral clusters
   - Applies cluster-specific adjustments (-50 to +50 points)

4. **Final Score Clamping**:
   - Ensures all scores remain within 0-1000 range

### Risk Assessment Logic

#### High-Risk Indicators (Lower Scores):
- High liquidation ratios (>10%)
- Low repayment ratios (<50%)
- Anomalous transaction patterns
- Very low transaction volumes (<$100)
- Irregular activity patterns

#### Low-Risk Indicators (Higher Scores):
- Consistent repayment behavior (>90% repay ratio)
- Diverse asset usage
- Regular activity patterns
- Substantial transaction volumes
- No liquidation events

## Processing Flow

```
Raw JSON Data → Data Loading → Feature Engineering → Anomaly Detection
                                        ↓
Score Clamping ← Cluster Adjustments ← Base Score Calculation
                                        ↓
Final Scores → Analysis Generation → Visualization → Output Files
```

## Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Running the Model
```python
python defi_credit_scorer.py
```

### Input
- `user-wallet-transactions.json`: Raw transaction data from Aave V2 protocol

### Output Files
- `wallet_credit_scores.csv`: Final scores for each wallet
- `detailed_analysis.json`: Comprehensive analysis results
- `credit_score_analysis.png`: Visualization plots

## Model Validation

### Feature Importance
The model prioritizes features that historically correlate with responsible DeFi behavior:
- Repayment consistency (30% weight)
- Activity patterns (25% weight)
- Protocol sophistication (25% weight)
- Transaction volume (20% weight)

### Anomaly Detection
- Uses Isolation Forest with 10% contamination rate
- Identifies wallets with unusual transaction patterns
- Helps detect bot-like or exploitative behavior

### Clustering Analysis
- K-Means clustering (k=5) for behavioral segmentation
- Enables peer comparison and risk assessment
- Provides cluster-specific score adjustments

## Score Interpretation

| Score Range | Risk Level | Characteristics |
|-------------|------------|----------------|
| 800-1000 | Very Low | Excellent repayment, diverse usage, high volume |
| 600-799 | Low | Good repayment history, regular activity |
| 400-599 | Medium | Mixed behavior, some risk indicators |
| 200-399 | High | Poor repayment, irregular patterns |
| 0-199 | Very High | Multiple liquidations, anomalous behavior |

## Extensibility

The system is designed for easy extension:

1. **New Features**: Add features in `engineer_features()` method
2. **Different Protocols**: Modify data parsing for other DeFi protocols
3. **Alternative Models**: Replace scoring algorithm in `calculate_credit_scores()`
4. **Custom Risk Rules**: Adjust cluster-based adjustments logic

## Limitations

- Historical data dependent (requires transaction history)
- Protocol-specific (designed for Aave V2)
- Market condition agnostic (doesn't account for external factors)
- Limited real-world validation data

## Future Enhancements

- Integration with external risk factors (market volatility, gas prices)
- Multi-protocol support (Compound, MakerDAO, etc.)
- Real-time scoring updates
- Machine learning model retraining pipeline
- Integration with on-chain reputation systems

## Technical Specifications

- **Language**: Python 3.7+
- **Dependencies**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **Input Format**: JSON (MongoDB export format)
- **Output Format**: CSV, JSON, PNG
- **Processing Time**: ~1-2 minutes for 100K transactions
- **Memory Usage**: ~500MB for 100K transactions

## Contact & Support

For questions, improvements, or bug reports, please refer to the analysis.md file for detailed behavioral insights and score distribution analysis.
