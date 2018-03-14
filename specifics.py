import warnings
import data
import pandas as pd

def build_data(feature_df, outcome_df):
    feature_df_cleaned = clean_feature_df(feature_df)
    return data.ClassificationData(feature_df, feature_df_cleaned, outcome_df)

# Number of subvillage is at 20000 level,
# Number of ward is at 2000 level,
# Number of funder is at 2000 level,
# Number of installer is at 2500 level,
# Number of scheme_name is at 2500 level,
# Number of wpt_name  is at 45000 level.

def clean_feature_df(feature_df_raw):
    return (feature_df_raw
            .assign(
                created_year=lambda x: pd.to_datetime(x['date_recorded']).dt.year,
                created_month=lambda x: pd.to_datetime(x['date_recorded']).dt.month,
                created_day=lambda x: pd.to_datetime(x['date_recorded']).dt.day,
                created_dow=lambda x: pd.to_datetime(x['date_recorded']).dt.weekday                
            )
            #.pipe(data.cols_to_numeric, cols= ['postCode', 
            #                                        'houseHoldSize.noOfFemaleChildren', 
            #                                        'houseHoldSize.noOfFemaleAdults', 
            #                                        'houseHoldSize.noOfMaleChildren',
            #                                        'houseHoldSize.noOfMaleAdults',
            #                                        'XPLANOccupationIdTrimmed']
            #)
            #.pipe(data.boolean_to_numeric, col = 'smoker')
            # drop all unused columns and dependent variables
            .drop(['date_recorded', 'funder', 'installer', 'wpt_name',    
                   'scheme_name', 'subvillage', 'ward'], axis=1)
            .pipe(data.cleaner)
            # Remove all columns which are constant - removed for testing
            # .pipe(lambda x: x.loc[:, x.apply(pd.Series.nunique) != 1])
            )
