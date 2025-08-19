import pandas as pd
import argparse
class Transform:
    

    # Creates new columns with feature engineering
    def feature_engineering(self,df, output_path):
       
        # Since TotalPremium had 0 values, construct it using CalculatedPremiumPerTerm
        df['TotalPremium'] = df['CalculatedPremiumPerTerm'] * 0.8772
        df['ClaimToPremiumRatio'] = df.groupby('PolicyID').apply(
            lambda group: group['TotalClaims'] / (group['TotalPremium'] + 1e-6)
        ).reset_index(level=0, drop=True)
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
        df['VehicleAge'] = df['TransactionMonth'].dt.year - df['RegistrationYear']
        df.to_csv(output_path, index=False)
        df = df  # Update the instance variable
        return df
    #This classifies the customers using ploicyID 
    def classify_by_cus(self,df, output_path):
        agge = df.groupby('PolicyID').agg({
            'TotalPremium': 'mean',  # Average TotalPremium
            'TotalClaims': 'mean',  # Average TotalClaims
            'make': lambda x: x.mode().iloc[0] if not x.mode().empty else None,  # Most frequent value in 'make'
            'Province': lambda x: x.mode().iloc[0] if not x.mode().empty else None  
        }).reset_index()
        #aggregating to get the claim ratio
        agge['ClaimToPremiumRatio'] = agge.groupby('PolicyID').apply(
            lambda group: group['TotalClaims'] / (group['TotalPremium'] + 1)
        ).reset_index(level=0, drop=True)
        threshold = agge['ClaimToPremiumRatio'].mean()
        agge['RiskCategory'] = agge['ClaimToPremiumRatio'].apply(lambda x: 'High Risk' if x > threshold else 'Low Risk')
        agge.to_csv(output_path, index=False)
        return agge
    #Using province to clasiffy
    def classify_by_province(self,df, output_path):
        agg = df.groupby('Province').agg({
            'TotalPremium': 'mean',
            'TotalClaims': 'mean',
            'ClaimToPremiumRatio': 'mean'
        }).reset_index()
        threshold = agg['ClaimToPremiumRatio'].mean()  # Defines threshold for low/high risk
        agg['RiskCategory'] = agg['ClaimToPremiumRatio'].apply(lambda x: 'High Risk' if x > threshold else 'Low Risk')
        agg.to_csv(output_path, index=False)
        return agg
def main():
    parser = argparse.ArgumentParser(description="Transformation")
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--function", required=True, help="Function to execute")
    parser.add_argument("--output", required=False, help="Output file path for transformed data", default="transformed_data.csv")
    args = parser.parse_args()

    # Load the data
    df = pd.read_csv(args.input)
    trans = Transform()

    # Execute the requested function
    if args.function == "feature_engineering":
        trans.feature_engineering(df, args.output)  # Use default if not specified
    elif args.function == "classify_by_cus" and args.output:
        trans.classify_by_cus(df, args.output)
    elif args.function == "classify_by_province" and args.output:
        trans.classify_by_province(df, args.output)
    else:
        print("Invalid function name or missing column argument for univariate/boxplot analysis.")

if __name__ == "__main__":
    main()