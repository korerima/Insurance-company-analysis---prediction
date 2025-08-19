import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class EDA:
    #This function gives us the description of our dataset
    def description(self, df):
        #To know how many customers there are
        unique_customers = df['PolicyID'].nunique()
        print(f"Unique Customers: {unique_customers}")
        return unique_customers
    #Tells us about the total permium and claim which are vital for this analysis
    def stat(self, df):
        num = ['TotalPremium', 'TotalClaims']
        for col in num:
            mean_value = df[col].mean()
            std_value = df[col].std()
            var_value = df[col].var()
            range_value = df[col].max() - df[col].min()
            iqr_value = df[col].quantile(0.75) - df[col].quantile(0.25)
            #Printing our findings
            print(f"\n{col} Descriptive Statistics:")
            print(f"Mean: {mean_value}")
            print(f"Standard Deviation: {std_value}")
            print(f"Variance: {var_value}")
            print(f"Range: {range_value}")
            print(f"IQR: {iqr_value}")
    #By providing the columns it will compute numarical univariate analysis
    def numerical_univariate(self, df, col):
        plt.figure(figsize=(15, 5))
        sns.histplot(df[col], bins=10, kde=True)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('./Local_Storage/eda_results/numerical_univariate.png')
        plt.close()
    #By providing the columns it will compute catagorcal univariate analysis
    def categorical_univariate(self, df, col):
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x=col)
        plt.title(f'Bar Chart of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('./Local_Storage/eda_results/categorical_univariate.png')
        plt.close()
    #By providing the columns it will compute bivariate analysis
    def bivariate_analysis(self, df):
        df['TotalPremium_Change'] = df.groupby('PostalCode')['TotalPremium'].diff()
        df['TotalClaims_Change'] = df.groupby('PostalCode')['TotalClaims'].diff()
        plt.figure(figsize=(12, 6))
        sns.scatterplot(
            data=df, 
            x='TotalPremium_Change', 
            y='TotalClaims_Change', 
            hue='PostalCode', 
            palette='tab10', 
            alpha=0.7
        )
        plt.title('Monthly Changes: TotalPremium vs TotalClaims (by PostalCode)', fontsize=16)
        plt.xlabel('Change in Total Premium', fontsize=14)
        plt.ylabel('Change in Total Claims', fontsize=14)
        plt.tight_layout()
        plt.savefig('./Local_Storage/eda_results/bivariate_analysis.png')
        plt.close()
    #Provided the column using visualization it we'll show as distrbution of our data
    def detect_outliers_with_boxplot(self, df, column):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[column], color="skyblue")
        plt.title(f"Box Plot of {column}", fontsize=16)
        plt.xlabel(column, fontsize=14)
        plt.savefig('./Local_Storage/eda_results/detect_outliers_with_boxplot.png')
        plt.close()

    def agg_province(self, df):
        summary= df.groupby('Province').agg({
            'TotalPremium':'mean',
            'TotalClaims':'mean'
        }).reset_index()
          # Sort and get the top three provinces by TotalPremium
        top_premium = summary.nlargest(3, 'TotalPremium')
        top_claims = summary.nlargest(3, 'TotalClaims')
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
        # Bar chart for TotalPremium
        axs[0].bar(top_premium['Province'], top_premium['TotalPremium'], color='blue')
        axs[0].set_title('Top 3 Provinces by Total Premium')
        axs[0].set_xlabel('Province')
        axs[0].set_ylabel('Total Premium')
        
        # Bar chart for TotalClaims
        axs[1].bar(top_claims['Province'], top_claims['TotalClaims'], color='orange')
        axs[1].set_title('Top 3 Provinces by Total Claims')
        axs[1].set_xlabel('Province')
        axs[1].set_ylabel('Total Claims')
        
        # Saves the plots
        plt.savefig('./Local_Storage/eda_results/agg_province.png')
        plt.close()
        return summary
    #What kind of company or individsuals make the highest claim
    def agg_legaltype(self,df):
        summary= df.groupby('LegalType').agg({
            'TotalPremium':'sum',
            'TotalClaims':'sum'
        })
        return summary
   
    #Aggregation per customer average premium and claims made 
    def customer_agg(self,df):
        summary= df.groupby('PolicyID').agg({
            'TotalPremium':'mean',
            'TotalClaims':'mean'
        })
        return summary
    #Trend based on provinances
    def geograpic_trend(self,df):
        
        geo_trends = df.groupby('Province').agg({
        'TotalPremium': ['mean', 'sum'],
        'CoverType': pd.Series.mode ,
        'make': pd.Series.mode  # Get the most common AutoMake
        }).reset_index()

        return geo_trends
    
    #Calcualted paymnet and actual paymnet
    def plan_and_payemnt(self,df):
        plan=df['CalculatedPremiumPerTerm'].sum()
        payment=df['TotalPremium'].sum()
        categories = ['Total Plan', 'Total Payment']
        values = [plan, payment]
        plt.figure(figsize=(8, 6))
        plt.bar(categories, values, color=['blue', 'orange', 'green'])
        plt.title('planned payment vs recived payment', fontsize=16)
        plt.savefig('./Local_Storage/eda_results/plan_and_payemnt.png')
        plt.close()
    
    #Total claims by car companies
    def per_car(self,df):
        df['make'] = df['make'].str.strip().str.upper()
        cars=df.groupby('make').agg({
            'TotalClaims':'mean'
        }).reset_index()
        cars = cars.sort_values(by='TotalClaims', ascending=True)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='TotalClaims', y='make', data=cars, palette='viridis')
        plt.title('Average Total Claims by Car brand', fontsize=16)
        plt.xlabel('Average Total Claims', fontsize=14)
        plt.ylabel('Make', fontsize=14)
        plt.savefig('./Local_Storage/eda_results/per_car.png')
        plt.close()
        return cars

def main():
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis CLI")
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--function", required=True, help="EDA function to execute (e.g., 'description', 'stat', 'numerical_univariate')")
    parser.add_argument("--column", required=False, help="Column name for univariate analysis or box plot")
    args = parser.parse_args()

    # Load the data
    df = pd.read_csv(args.input)
    eda = EDA()

    # Execute the requested function
    if args.function == "description":
        eda.description(df)
    elif args.function == "stat":
        eda.stat(df)
    elif args.function == "numerical_univariate" and args.column:
        eda.numerical_univariate(df, args.column)
    elif args.function == "categorical_univariate" and args.column:
        eda.categorical_univariate(df, args.column)
    elif args.function == "bivariate_analysis":
        eda.bivariate_analysis(df)
    elif args.function == "detect_outliers_with_boxplot" and args.column:
        eda.detect_outliers_with_boxplot(df, args.column)
    elif args.function == "agg_province":
        eda.agg_province(df)
    elif args.function == "agg_legaltype":
        eda.agg_legaltype(df)
    elif args.function == "customer_agg":
        eda.customer_agg(df)
    elif args.function == "geograpic_trend":
        eda.geograpic_trend(df) 
    elif args.function == "plan_and_payemnt":
        eda.plan_and_payemnt(df)
    elif args.function == "per_car":
        eda.per_car(df)
    else:
        print("Invalid function name or missing column argument for univariate/boxplot analysis.")

if __name__ == "__main__":
    main()
