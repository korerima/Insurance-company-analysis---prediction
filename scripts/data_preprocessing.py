import pandas as pd
import os, sys
#df= pd.read_csv('C:/Users/hello/Desktop/MachineLearningRating_v3.csv')
class prepocess:
    def load_data(self, file):
        return pd.read_csv(file)
    
    def check_datatype(self, df):
        return df.dtypes
    
    def missing_values(self, df):
        return df.isnull().sum()
    
    def handle_dataset(self, df, output_file):
        drops = ['NumberOfVehiclesInFleet', 'CrossBorder', 'Converted', 'Rebuilt', 'WrittenOff', 'CustomValueEstimate']
        df_edited = df.drop(drops, axis=1)
        fill = ['Bank', 'AccountType', 'MaritalStatus', 'Gender', 'NewVehicle']
        df_edited[fill] = df_edited[fill].fillna('Not specified')
        df_edited = df_edited.dropna(subset=['mmcode'])
        df_edited = df_edited[df_edited['TotalPremium'] >= 0]
        df_edited = df_edited[df_edited['TotalClaims'] >= 0]
        df_edited.to_csv(output_file, index=False)
        return df_edited
    