import pandas as pd
import numpy as np
from scipy.stats import f_oneway,ttest_ind
from scipy.stats import ttest_ind, chi2_contingency
df=pd.read_csv('C:/Users/hello/Desktop/git/Insurance-company-analysis-and-prediction/Local_Storage/Preprocessed/edited.csv')
class hypothesis:
    def __init__(self,df):
        self.df=df    

    def perform_t_test(group_a, group_b, kpi):
        """Performs a T-Test."""
        t_stat, p_value = ttest_ind(group_a, group_b)
        print(f"T-Test for {kpi}:")
        print(f"T-Statistic: {t_stat}, P-Value: {p_value}")
        return p_value

    def perform_chi_squared(group_a, group_b, col):
        """Performs a Chi-Squared Test."""
        contingency_table = pd.crosstab(group_a, group_b)
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-Squared Test for {col}:")
        print(f"Chi2-Statistic: {chi2_stat}, P-Value: {p_value}")
        return p_value
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    
    # 1. Risk differences across provinces
    province_median = df.groupby('Province')['TotalClaims'].median()
    df['ProvinceMedian'] = df['Province'].map(province_median)

    # Create the risk group
    df['Provinance_RiskGroup'] = df.apply(
        lambda row: 'Low Risk' if row['TotalClaims'] <= row['ProvinceMedian'] else 'High Risk',
        axis=1
    )
    group_a_prov = df[df['Provinance_RiskGroup'] == 'Low Risk']['TotalClaims']
    group_b_prov = df[df['Provinance_RiskGroup'] == 'High Risk']['TotalClaims']
    perform_t_test(group_a_prov, group_b_prov, 'TotalClaims (Provinces)')

    # 2. Risk differences between zip codes
    Postal_median = df.groupby('PostalCode')['TotalClaims'].median()
    df['Postal_median'] = df['PostalCode'].map(Postal_median)

    # Create the risk group
    df['Postal_RiskGroup'] = df.apply(
        lambda row: 'Low Risk' if row['TotalClaims'] <= row['Postal_median'] else 'High Risk',
        axis=1
    )
    group_a_zip = df[df['Postal_RiskGroup'] == 'Low Risk']['TotalClaims']
    group_b_zip = df[df['Postal_RiskGroup'] == 'High Risk']['TotalClaims']
    perform_t_test(group_a_zip, group_b_zip, 'TotalClaims (Postal Codes)')

    # 3. Margin differences between zip codes
    postal_margin_median = df.groupby('PostalCode')['Margin'].median()
    df['postal_margin_median'] = df['PostalCode'].map(postal_margin_median)

    # Create the risk group
    df['Postal_margin_RiskGroup'] = df.apply(
        lambda row: 'Low Risk' if row['Margin'] <= row['postal_margin_median'] else 'High Risk',
        axis=1
    )
    group_a_margin = df[df['Postal_margin_RiskGroup'] == 'Low Margin']['Margin']
    group_b_margin = df[df['Postal_margin_RiskGroup'] == 'High Margin']['Margin']
    perform_t_test(group_a_margin, group_b_margin, 'Margin (Postal Codes)')

    # 4. Risk differences between Women and Men
    group_a_gender = df[df['Gender'] == 'Female']['TotalClaims']
    group_b_gender = df[df['Gender'] == 'Male']['TotalClaims']
    perform_t_test(group_a_gender, group_b_gender, 'TotalClaims (Gender)')