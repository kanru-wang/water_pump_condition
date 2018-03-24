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
                created_dow=lambda x: pd.to_datetime(x['date_recorded']).dt.weekday # Monday is 0, Sunday is 6. 
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
            .assign(
                categorical_district_code = lambda x: x['district_code'].apply(lambda x: data.turn_string(x)),
                #categorical_construction_year = lambda x: x['construction_year'].apply(lambda x: data.turn_string(x)),
                categorical_created_year = lambda x: x['created_year'].apply(lambda x: data.turn_string(x)),
                categorical_created_month = lambda x: x['created_month'].apply(lambda x: data.turn_string(x)),
                #categorical_created_day = lambda x: x['created_day'].apply(lambda x: data.turn_string(x)),
                categorical_created_dow = lambda x: x['created_dow'].apply(lambda x: data.turn_string(x))
            )
            .assign(
                funder_length = lambda x: x['funder'].apply(lambda x: data.measure_length(x)),
                installer_length = lambda x: x['installer'].apply(lambda x: data.measure_length(x)),
                wpt_name_length = lambda x: x['wpt_name'].apply(lambda x: data.measure_length(x)),
                scheme_name_length = lambda x: x['scheme_name'].apply(lambda x: data.measure_length(x)),
                subvillage_name_length = lambda x: x['subvillage'].apply(lambda x: data.measure_length(x)),
                ward_length = lambda x: x['ward'].apply(lambda x: data.measure_length(x))
            )
            .assign(
                funder_freq = lambda x: x.groupby('funder')['funder'].transform('count'),
                installer_freq = lambda x: x.groupby('installer')['installer'].transform('count'),
                wpt_name_freq = lambda x: x.groupby('wpt_name')['wpt_name'].transform('count'),
                scheme_name_freq = lambda x: x.groupby('scheme_name')['scheme_name'].transform('count'),
                subvillage_freq = lambda x: x.groupby('subvillage')['subvillage'].transform('count'),
                ward_freq = lambda x: x.groupby('ward')['ward'].transform('count')                
            )
            .assign(
                funder_small_levels_grouped = lambda x: data.group_small_levels(x, 'funder', 'funder_freq', 80),
                installer_small_levels_grouped = lambda x: data.group_small_levels(x, 'installer', 'installer_freq', 80),
                wpt_name_small_levels_grouped = lambda x: data.group_small_levels(x, 'wpt_name', 'wpt_name_freq', 80),
                scheme_name_small_levels_grouped = lambda x: data.group_small_levels(x, 'scheme_name', 'scheme_name_freq', 80),
                subvillage_small_levels_grouped = lambda x: data.group_small_levels(x, 'subvillage', 'subvillage_freq', 80),
                ward_small_levels_grouped = lambda x: data.group_small_levels(x, 'ward', 'ward_freq', 80)
            )
            # drop all unused columns and dependent variables
            .drop(['date_recorded', 'funder', 'installer', 'wpt_name', 'district_code',   
                   'scheme_name', 'subvillage', 'ward'], axis=1)
            .pipe(data.cleaner)
            # Remove all columns which are constant - removed for testing
            .pipe(lambda x: x.loc[:, x.apply(pd.Series.nunique) != 1])
            # Each column need to in this case have at least xx cases
            .pipe(lambda x: x.loc[:, x.apply(pd.Series.sum) >= 80])
            )


def encode_outcome_df(df_raw):
    return(df_raw.replace(to_replace = ['functional', 'functional needs repair', 'non functional'], 
                          value = [2, 1, 0]))
