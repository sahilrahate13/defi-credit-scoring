#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
"""
DeFi Credit Scoring Model for Aave V2 Protocol
Author: AI Assistant
Description: Machine learning model to assign credit scores (0-1000) to wallets based on transaction behavior
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class DeFiCreditScorer:
    """
    DeFi Credit Scoring Model for Aave V2 Protocol
    
    This model analyzes wallet transaction patterns to assign credit scores between 0-1000.
    Higher scores indicate reliable, responsible usage; lower scores indicate risky behavior.
    """
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.feature_importance = {}
        
    def load_data(self, json_file_path: str) -> pd.DataFrame:
        """Load and parse the JSON transaction data"""
        print("Loading transaction data...")
        
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Flatten the nested structure
        transactions = []
        for record in data:
            flat_record = {
                'userWallet': record['userWallet'],
                'network': record['network'],
                'protocol': record['protocol'],
                'txHash': record['txHash'],
                'timestamp': record['timestamp'],
                'blockNumber': record['blockNumber'],
                'action': record['action'],
                'amount': float(record['actionData']['amount']) if record['actionData']['amount'] else 0,
                'assetSymbol': record['actionData']['assetSymbol'],
                'assetPriceUSD': float(record['actionData']['assetPriceUSD']) if record['actionData']['assetPriceUSD'] else 0,
                'poolId': record['actionData']['poolId'],
                'userId': record['actionData']['userId']
            }
            transactions.append(flat_record)
        
        df = pd.DataFrame(transactions)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['usd_value'] = df['amount'] * df['assetPriceUSD']
        
        print(f"Loaded {len(df)} transactions for {df['userWallet'].nunique()} unique wallets")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive features for credit scoring"""
        print("Engineering features...")
        
        wallet_features = []
        
        for wallet in df['userWallet'].unique():
            wallet_data = df[df['userWallet'] == wallet].copy()
            wallet_data = wallet_data.sort_values('timestamp')
            
            # Basic transaction metrics
            total_transactions = len(wallet_data)
            unique_assets = wallet_data['assetSymbol'].nunique()
            total_volume_usd = wallet_data['usd_value'].sum()
            avg_transaction_size = wallet_data['usd_value'].mean()
            
            # Time-based features
            first_tx = wallet_data['timestamp'].min()
            last_tx = wallet_data['timestamp'].max()
            account_age_days = (last_tx - first_tx) / 86400  # Convert to days
            activity_span_days = account_age_days if account_age_days > 0 else 1
            
            # Action distribution
            action_counts = wallet_data['action'].value_counts()
            deposits = action_counts.get('deposit', 0)
            borrows = action_counts.get('borrow', 0)
            repays = action_counts.get('repay', 0)
            redeems = action_counts.get('redeemunderlying', 0)
            liquidations = action_counts.get('liquidationcall', 0)
            
            # Financial behavior metrics
            deposit_volume = wallet_data[wallet_data['action'] == 'deposit']['usd_value'].sum()
            borrow_volume = wallet_data[wallet_data['action'] == 'borrow']['usd_value'].sum()
            repay_volume = wallet_data[wallet_data['action'] == 'repay']['usd_value'].sum()
            
            # Risk indicators
            liquidation_ratio = liquidations / total_transactions if total_transactions > 0 else 0
            borrow_to_deposit_ratio = borrow_volume / deposit_volume if deposit_volume > 0 else 0
            repay_to_borrow_ratio = repay_volume / borrow_volume if borrow_volume > 0 else 1
            
            # Activity patterns
            transaction_frequency = total_transactions / activity_span_days if activity_span_days > 0 else 0
            
            # Advanced behavioral features
            # Transaction size variance (consistency indicator)
            tx_size_std = wallet_data['usd_value'].std()
            tx_size_cv = tx_size_std / avg_transaction_size if avg_transaction_size > 0 else 0
            
            # Time between transactions (regularity indicator)
            time_diffs = wallet_data['timestamp'].diff().dropna()
            avg_time_between_tx = time_diffs.mean() / 3600 if len(time_diffs) > 0 else 0  # Convert to hours
            
            # Asset diversification
            asset_diversification = unique_assets / total_transactions if total_transactions > 0 else 0
            
            # Behavioral scoring components
            # 1. Reliability Score (0-1)
            reliability_score = min(1.0, repay_to_borrow_ratio) * (1 - liquidation_ratio)
            
            # 2. Activity Score (0-1) - based on consistent usage
            activity_score = min(1.0, transaction_frequency / 0.1) * min(1.0, account_age_days / 30)
            
            # 3. Sophistication Score (0-1) - based on diverse usage
            sophistication_score = min(1.0, unique_assets / 3) * min(1.0, (deposits + borrows + repays) / total_transactions)
            
            # 4. Volume Score (0-1) - normalized by percentile
            volume_score = min(1.0, np.log1p(total_volume_usd) / 15)  # Log scale for volume
            
            features = {
                'userWallet': wallet,
                'total_transactions': total_transactions,
                'unique_assets': unique_assets,
                'total_volume_usd': total_volume_usd,
                'avg_transaction_size': avg_transaction_size,
                'account_age_days': account_age_days,
                'deposits': deposits,
                'borrows': borrows,
                'repays': repays,
                'redeems': redeems,
                'liquidations': liquidations,
                'deposit_volume': deposit_volume,
                'borrow_volume': borrow_volume,
                'repay_volume': repay_volume,
                'liquidation_ratio': liquidation_ratio,
                'borrow_to_deposit_ratio': borrow_to_deposit_ratio,
                'repay_to_borrow_ratio': repay_to_borrow_ratio,
                'transaction_frequency': transaction_frequency,
                'tx_size_cv': tx_size_cv,
                'avg_time_between_tx': avg_time_between_tx,
                'asset_diversification': asset_diversification,
                'reliability_score': reliability_score,
                'activity_score': activity_score,
                'sophistication_score': sophistication_score,
                'volume_score': volume_score
            }
            
            wallet_features.append(features)
        
        features_df = pd.DataFrame(wallet_features)
        
        # Handle infinite and NaN values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)
        
        print(f"Engineered {len(features_df.columns)-1} features for {len(features_df)} wallets")
        return features_df
    
    def detect_anomalies(self, features_df: pd.DataFrame) -> np.ndarray:
        """Detect anomalous wallet behavior using Isolation Forest"""
        feature_cols = [col for col in features_df.columns if col != 'userWallet']
        X = features_df[feature_cols]
        
        # Fit anomaly detector
        anomaly_scores = self.anomaly_detector.fit_predict(X)
        return anomaly_scores
    
    def calculate_credit_scores(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate final credit scores using ensemble approach"""
        print("Calculating credit scores...")
        
        feature_cols = [col for col in features_df.columns if col != 'userWallet']
        X = features_df[feature_cols].copy()
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Detect anomalies
        anomaly_scores = self.detect_anomalies(features_df)
        
        # Cluster wallets for behavior segmentation
        clusters = self.kmeans.fit_predict(X_scaled)
        
        # Calculate base scores using weighted combination of key metrics
        base_scores = (
            features_df['reliability_score'] * 0.3 +
            features_df['activity_score'] * 0.25 +
            features_df['sophistication_score'] * 0.25 +
            features_df['volume_score'] * 0.2
        ) * 1000
        
        # Apply anomaly penalty (reduce score for anomalous behavior)
        anomaly_penalty = np.where(anomaly_scores == -1, 200, 0)  # 200 point penalty for anomalies
        
        # Cluster-based adjustments
        cluster_adjustments = self._calculate_cluster_adjustments(features_df, clusters)
        
        # Final score calculation
        final_scores = base_scores - anomaly_penalty + cluster_adjustments
        
        # Ensure scores are within 0-1000 range
        final_scores = np.clip(final_scores, 0, 1000)
        
        # Add metadata to results
        results_df = features_df.copy()
        results_df['credit_score'] = final_scores
        results_df['is_anomaly'] = anomaly_scores == -1
        results_df['cluster'] = clusters
        results_df['score_bucket'] = pd.cut(final_scores, bins=10, labels=[f"{i*100}-{(i+1)*100}" for i in range(10)])
        
        return results_df
    
    def _calculate_cluster_adjustments(self, features_df: pd.DataFrame, clusters: np.ndarray) -> np.ndarray:
        """Calculate cluster-based score adjustments"""
        adjustments = np.zeros(len(features_df))
        
        for cluster_id in np.unique(clusters):
            cluster_mask = clusters == cluster_id
            cluster_data = features_df[cluster_mask]
            
            # Calculate cluster characteristics
            avg_liquidation_ratio = cluster_data['liquidation_ratio'].mean()
            avg_repay_ratio = cluster_data['repay_to_borrow_ratio'].mean()
            avg_volume = cluster_data['total_volume_usd'].mean()
            
            # Assign cluster adjustments based on risk profile
            if avg_liquidation_ratio > 0.1:  # High liquidation cluster
                adjustment = -50
            elif avg_repay_ratio > 0.9 and avg_volume > 10000:  # High-quality cluster
                adjustment = 50
            elif avg_volume < 100:  # Low-volume cluster
                adjustment = -25
            else:
                adjustment = 0
            
            adjustments[cluster_mask] = adjustment
        
        return adjustments
    
    def generate_analysis(self, results_df: pd.DataFrame) -> Dict:
        """Generate comprehensive analysis of wallet scores"""
        print("Generating analysis...")
        
        analysis = {
            'total_wallets': len(results_df),
            'score_distribution': {},
            'cluster_analysis': {},
            'risk_insights': {}
        }
        
        # Score distribution
        for bucket in results_df['score_bucket'].unique():
            if pd.isna(bucket):
                continue
            bucket_data = results_df[results_df['score_bucket'] == bucket]
            analysis['score_distribution'][bucket] = {
                'count': len(bucket_data),
                'percentage': len(bucket_data) / len(results_df) * 100,
                'avg_volume': bucket_data['total_volume_usd'].mean(),
                'avg_transactions': bucket_data['total_transactions'].mean(),
                'liquidation_rate': bucket_data['liquidation_ratio'].mean()
            }
        
        # Cluster analysis
        for cluster_id in results_df['cluster'].unique():
            cluster_data = results_df[results_df['cluster'] == cluster_id]
            analysis['cluster_analysis'][f'cluster_{cluster_id}'] = {
                'count': len(cluster_data),
                'avg_score': cluster_data['credit_score'].mean(),
                'characteristics': {
                    'avg_volume': cluster_data['total_volume_usd'].mean(),
                    'avg_transactions': cluster_data['total_transactions'].mean(),
                    'avg_liquidation_ratio': cluster_data['liquidation_ratio'].mean(),
                    'avg_repay_ratio': cluster_data['repay_to_borrow_ratio'].mean()
                }
            }
        
        # Risk insights
        low_score_wallets = results_df[results_df['credit_score'] < 300]
        high_score_wallets = results_df[results_df['credit_score'] > 700]
        
        analysis['risk_insights'] = {
            'low_score_behavior': {
                'count': len(low_score_wallets),
                'common_traits': {
                    'high_liquidation_rate': (low_score_wallets['liquidation_ratio'] > 0.1).sum(),
                    'low_repay_ratio': (low_score_wallets['repay_to_borrow_ratio'] < 0.5).sum(),
                    'anomalous_behavior': low_score_wallets['is_anomaly'].sum()
                }
            },
            'high_score_behavior': {
                'count': len(high_score_wallets),
                'common_traits': {
                    'consistent_repayment': (high_score_wallets['repay_to_borrow_ratio'] > 0.9).sum(),
                    'diverse_assets': (high_score_wallets['unique_assets'] > 2).sum(),
                    'high_volume': (high_score_wallets['total_volume_usd'] > 10000).sum()
                }
            }
        }
        
        return analysis
    
    def create_visualizations(self, results_df: pd.DataFrame, analysis: Dict):
        """Create visualization plots for analysis"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Score distribution histogram
        axes[0, 0].hist(results_df['credit_score'], bins=20, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Credit Score Distribution')
        axes[0, 0].set_xlabel('Credit Score')
        axes[0, 0].set_ylabel('Number of Wallets')
        
        # Score by bucket
        bucket_data = []
        bucket_labels = []
        for bucket, data in analysis['score_distribution'].items():
            bucket_data.append(data['count'])
            bucket_labels.append(bucket)
        
        axes[0, 1].bar(range(len(bucket_data)), bucket_data)
        axes[0, 1].set_title('Wallets by Score Range')
        axes[0, 1].set_xlabel('Score Range')
        axes[0, 1].set_ylabel('Number of Wallets')
        axes[0, 1].set_xticks(range(len(bucket_labels)))
        axes[0, 1].set_xticklabels(bucket_labels, rotation=45)
        
        # Volume vs Score scatter
        axes[1, 0].scatter(results_df['total_volume_usd'], results_df['credit_score'], alpha=0.6)
        axes[1, 0].set_title('Volume vs Credit Score')
        axes[1, 0].set_xlabel('Total Volume (USD)')
        axes[1, 0].set_ylabel('Credit Score')
        axes[1, 0].set_xscale('log')
        
        # Liquidation ratio vs Score
        axes[1, 1].scatter(results_df['liquidation_ratio'], results_df['credit_score'], alpha=0.6)
        axes[1, 1].set_title('Liquidation Ratio vs Credit Score')
        axes[1, 1].set_xlabel('Liquidation Ratio')
        axes[1, 1].set_ylabel('Credit Score')
        
        plt.tight_layout()
        plt.savefig('credit_score_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, results_df: pd.DataFrame, analysis: Dict):
        """Save results and analysis to files"""
        # Save wallet scores
        output_df = results_df[['userWallet', 'credit_score', 'score_bucket', 'is_anomaly', 'cluster']].copy()
        output_df.to_csv('wallet_credit_scores.csv', index=False)
        
        # Save detailed analysis
        with open('detailed_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print("Results saved to:")
        print("- wallet_credit_scores.csv")
        print("- detailed_analysis.json")
        print("- credit_score_analysis.png")

def main():
    """Main execution function"""
    # Initialize the credit scorer
    scorer = DeFiCreditScorer()
    
    # Load and process data
    json_file = "user-wallet-transactions.json"  # Assuming file is in current directory
    
    try:
        # Load transaction data
        transactions_df = scorer.load_data(json_file)
        
        # Engineer features
        features_df = scorer.engineer_features(transactions_df)
        
        # Calculate credit scores
        results_df = scorer.calculate_credit_scores(features_df)
        
        # Generate analysis
        analysis = scorer.generate_analysis(results_df)
        
        # Create visualizations
        scorer.create_visualizations(results_df, analysis)
        
        # Save results
        scorer.save_results(results_df, analysis)
        
        # Print summary
        print("\n" + "="*50)
        print("CREDIT SCORING COMPLETE")
        print("="*50)
        print(f"Total wallets analyzed: {len(results_df)}")
        print(f"Average credit score: {results_df['credit_score'].mean():.2f}")
        print(f"Score range: {results_df['credit_score'].min():.0f} - {results_df['credit_score'].max():.0f}")
        print(f"Anomalous wallets detected: {results_df['is_anomaly'].sum()}")
        
        # Print score distribution
        print("\nScore Distribution:")
        for bucket, data in analysis['score_distribution'].items():
            print(f"  {bucket}: {data['count']} wallets ({data['percentage']:.1f}%)")
        
    except FileNotFoundError:
        print(f"Error: Could not find {json_file}")
        print("Please ensure the JSON file is in the current directory")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()


# In[23]:


import json
import pandas as pd
import numpy as np
import os
import sys

def create_sample_data():
    """Create sample transaction data for testing"""
    sample_transactions = []
    
    # Sample wallet 1: High-quality user
    for i in range(10):
        sample_transactions.append({
            "_id": {"$oid": f"test_id_{i}_1"},
            "userWallet": "0x1111111111111111111111111111111111111111",
            "network": "polygon",
            "protocol": "aave_v2",
            "txHash": f"0xhash_{i}_1",
            "logId": f"0xhash_{i}_1_Deposit",
            "timestamp": 1629178166 + i * 86400,
            "blockNumber": 1629178166 + i * 86400,
            "action": "deposit" if i % 3 != 2 else "repay",
            "actionData": {
                "type": "Deposit" if i % 3 != 2 else "Repay",
                "amount": str(1000000000 * (i + 1)),
                "assetSymbol": "USDC" if i % 2 == 0 else "WMATIC",
                "assetPriceUSD": "1.0" if i % 2 == 0 else "1.5",
                "poolId": "0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
                "userId": "0x1111111111111111111111111111111111111111"
            },
            "__v": 0,
            "createdAt": {"$date": "2025-05-08T23:06:39.465Z"},
            "updatedAt": {"$date": "2025-05-08T23:06:39.465Z"}
        })
    
    # Sample wallet 2: Medium-risk user with some liquidations
    for i in range(8):
        action = "borrow" if i % 4 == 0 else ("liquidationcall" if i == 6 else "deposit")
        sample_transactions.append({
            "_id": {"$oid": f"test_id_{i}_2"},
            "userWallet": "0x2222222222222222222222222222222222222222",
            "network": "polygon",
            "protocol": "aave_v2",
            "txHash": f"0xhash_{i}_2",
            "logId": f"0xhash_{i}_2_Action",
            "timestamp": 1629178166 + i * 172800,
            "blockNumber": 1629178166 + i * 172800,
            "action": action,
            "actionData": {
                "type": action.title(),
                "amount": str(500000000),
                "assetSymbol": "USDC",
                "assetPriceUSD": "1.0",
                "poolId": "0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
                "userId": "0x2222222222222222222222222222222222222222"
            },
            "__v": 0,
            "createdAt": {"$date": "2025-05-08T23:06:39.465Z"},
            "updatedAt": {"$date": "2025-05-08T23:06:39.465Z"}
        })
    
    # Sample wallet 3: High-risk user with many liquidations
    for i in range(5):
        action = "liquidationcall" if i >= 2 else "deposit"
        sample_transactions.append({
            "_id": {"$oid": f"test_id_{i}_3"},
            "userWallet": "0x3333333333333333333333333333333333333333",
            "network": "polygon",
            "protocol": "aave_v2",
            "txHash": f"0xhash_{i}_3",
            "logId": f"0xhash_{i}_3_Action",
            "timestamp": 1629178166 + i * 3600,
            "blockNumber": 1629178166 + i * 3600,
            "action": action,
            "actionData": {
                "type": action.title(),
                "amount": str(100000000),
                "assetSymbol": "WMATIC",
                "assetPriceUSD": "1.5",
                "poolId": "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270",
                "userId": "0x3333333333333333333333333333333333333333"
            },
            "__v": 0,
            "createdAt": {"$date": "2025-05-08T23:06:39.465Z"},
            "updatedAt": {"$date": "2025-05-08T23:06:39.465Z"}
        })
     # ✅ Add extra dummy wallets to ensure we have >= 5 wallets
    for wallet_suffix in ['44', '55']:
        for tx in sample_transactions[:5]:  # Just copy first 5 from wallet 1
            tx_copy = tx.copy()
            tx_copy["userWallet"] = f"0x{wallet_suffix * 20}"
            tx_copy["actionData"]["userId"] = f"0x{wallet_suffix * 20}"
            sample_transactions.append(tx_copy)
    return sample_transactions

def validate_model_output(results_df):
    """Validate the model output for correctness"""
    validation_results = {'passed': True, 'tests': {}}
    
    score_range_valid = (results_df['credit_score'].min() >= 0) and (results_df['credit_score'].max() <= 1000)
    validation_results['tests']['score_range'] = {
        'passed': score_range_valid,
        'message': f"Scores in valid range 0-1000: {score_range_valid}",
        'details': f"Min: {results_df['credit_score'].min():.2f}, Max: {results_df['credit_score'].max():.2f}"
    }
    
    wallet_1_score = results_df[results_df['userWallet'] == '0x1111111111111111111111111111111111111111']['credit_score'].iloc[0]
    wallet_3_score = results_df[results_df['userWallet'] == '0x3333333333333333333333333333333333333333']['credit_score'].iloc[0]
    score_logic_valid = wallet_1_score > wallet_3_score
    validation_results['tests']['score_logic'] = {
        'passed': score_logic_valid,
        'message': f"High-quality wallet scored higher than risky wallet: {score_logic_valid}",
        'details': f"Wallet 1: {wallet_1_score:.2f}, Wallet 3: {wallet_3_score:.2f}"
    }
    
    required_columns = ['userWallet', 'credit_score', 'is_anomaly', 'cluster']
    columns_exist = all(col in results_df.columns for col in required_columns)
    validation_results['tests']['required_columns'] = {
        'passed': columns_exist,
        'message': f"All required columns present: {columns_exist}",
        'details': f"Required: {required_columns}, Present: {list(results_df.columns)}"
    }
    
    no_nan_scores = not results_df['credit_score'].isna().any()
    validation_results['tests']['no_nan_scores'] = {
        'passed': no_nan_scores,
        'message': f"No NaN values in credit scores: {no_nan_scores}",
        'details': f"NaN count: {results_df['credit_score'].isna().sum()}"
    }
    
    validation_results['passed'] = all(test['passed'] for test in validation_results['tests'].values())
    return validation_results

def run_validation():
    """Run complete validation test"""
    print("=" * 60)
    print("DEFI CREDIT SCORER VALIDATION TEST")
    print("=" * 60)
    
    try:
        # Step 1: Create sample data
        print("\n1. Creating sample test data...")
        sample_data = create_sample_data()
        
        with open('test_sample_data.json', 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"   Created {len(sample_data)} sample transactions for 3 test wallets")
        
        # Step 2: Initialize scorer (assumes DeFiCreditScorer is defined above)
        print("\n2. Initializing DeFi Credit Scorer...")
        scorer = DeFiCreditScorer()
        
        # Step 3: Run scoring
        print("\n3. Processing test data...")
        transactions_df = scorer.load_data('test_sample_data.json')
        features_df = scorer.engineer_features(transactions_df)
        results_df = scorer.calculate_credit_scores(features_df)
        
        print("\n4. Test Results:")
        print("-" * 40)
        for _, row in results_df.iterrows():
            wallet = row['userWallet'][-4:]
            print(f"   Wallet ...{wallet}: Score={row['credit_score']:.0f}, Anomaly={row['is_anomaly']}, Cluster={row['cluster']}")
        
        print("\n5. Validation Tests:")
        print("-" * 40)
        validation = validate_model_output(results_df)
        
        for test_name, test_result in validation['tests'].items():
            status = "✓ PASS" if test_result['passed'] else "✗ FAIL"
            print(f"   {status}: {test_result['message']}")
            if not test_result['passed']:
                print(f"      Details: {test_result['details']}")
        
        print("\n" + "=" * 60)
        if validation['passed']:
            print("🎉 ALL VALIDATION TESTS PASSED!")
        else:
            print("❌ SOME VALIDATION TESTS FAILED!")
        print("=" * 60)
        
        os.remove('test_sample_data.json')
        return validation['passed']
    
    except Exception as e:
        print(f"\n❌ VALIDATION ERROR: {str(e)}")
        return False

# Run this in a Jupyter cell
run_validation()


# In[ ]:





# In[ ]:





# In[ ]:




