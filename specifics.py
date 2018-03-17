import warnings
import data
import pandas as pd

def build_data(train_feature_df, train_outcome_df, test_feature_df):
    # Combine train feature df and test feature df for pre-processing
    train_feature_df['train_test_tag'] = 'train'
    test_feature_df['train_test_tag'] = 'test'
    combined_df = pd.concat([train_feature_df, test_feature_df], axis = 0)
    
    # Do the pre-processing
    combined_df_cleaned = clean_feature_df(combined_df)
    train_outcome_df_encoded = encode_outcome_df(train_outcome_df)
        
    # Seperate train and test
    train_feature_df_cleaned = combined_df_cleaned.loc[combined_df_cleaned['train_test_tag__train'] == 1].drop(
        ['train_test_tag__train', 'train_test_tag__test'], axis = 1)
    test_feature_df_cleaned = combined_df_cleaned.loc[combined_df_cleaned['train_test_tag__test'] == 1].drop(
        ['train_test_tag__train', 'train_test_tag__test'], axis = 1)
    
    # Construct the data object using cleaned feature
    return data.ClassificationData(train_feature_df_cleaned, train_outcome_df_encoded, test_feature_df_cleaned)


# Number of subvillage is at 20000 level,
# Number of ward is at 2000 level,
# Number of funder is at 2000 level,
# Number of installer is at 2500 level,
# Number of scheme_name is at 2500 level,
# Number of wpt_name  is at 45000 level.

def clean_feature_df(df_raw):
    return (df_raw
            .assign(
                created_year=lambda x: pd.to_datetime(x['date_recorded']).dt.year,
                created_month=lambda x: pd.to_datetime(x['date_recorded']).dt.month,
                created_day=lambda x: pd.to_datetime(x['date_recorded']).dt.day,
                created_dow=lambda x: pd.to_datetime(x['date_recorded']).dt.weekday
            )
            .assign(age = lambda x: x['created_year'] - x['construction_year'])
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
            .pipe(lambda x: x.loc[:, x.apply(pd.Series.nunique) != 1])
            )


def encode_outcome_df(df_raw):
    return(df_raw.replace(to_replace = ['functional', 'functional needs repair', 'non functional'], 
                          value = [2, 1, 0]))
