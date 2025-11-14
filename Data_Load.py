import re
import pandas as pd
import numpy as np


def data_load(name, redundancy=False):
    directory_path = "C:/Users/DELL/Desktop/sw-ALR/dataset/"
    # directory_path= "/data1/BCChen/sw-ALR/dataset/"

    # load Concrete
    if name == "concrete":
        file_path = f'{directory_path}concrete_com_str/Concrete_Data.xls'
        df = pd.read_excel(file_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

    # load EE-heat
    if name == "energy-heat":
        file_path = f'{directory_path}energy_efficiency/ENB2012_data.xlsx'
        df = pd.read_excel(file_path)
        X = df.iloc[:, :-2].values
        y = df.iloc[:, -2].values

    # load EE-cool
    if name == "energy-cool":
        file_path = f'{directory_path}energy_efficiency/ENB2012_data.xlsx'
        df = pd.read_excel(file_path)
        X = df.iloc[:, :-2].values
        y = df.iloc[:, -1].values

    # load yacht
    if name == "yacht":
        file_path = f'{directory_path}yacht_hydrodynamics/yacht_hydrodynamics.data'
        df = pd.read_csv(file_path, header=None)
        data = df.values
        processed_data = [list(map(float, row[0].split())) for row in data]
        data_array = np.array(processed_data)
        X = data_array[:, :-1]
        y = data_array[:, -1]

    # load abalone
    if name == "abalone":
        # detail
        # with open("/data1/BCChen/abalone/abalone.names", 'r') as f:
        #     print(f.read())
        file_path = f'{directory_path}abalone/abalone.data'
        df = pd.read_csv(file_path, header=None)
        #  get_dummies  'Gender' one-hot
        df_encoded = pd.get_dummies(df, columns=[0], drop_first=False)
        y = df_encoded[4].values
        X = df_encoded.drop(columns=[4]).values

    # load cps
    if name == "cps":
        # file_path = "/data1/BCChen/cps/cps.txt"
        # column_names = ['EDUCATION', 'SOUTH', 'SEX', 'EXPERIENCE', 'UNION', 'WAGE', 'AGE', 'RACE',
        #                 'OCCUPATION', 'SECTOR', 'MARR']
        # data = pd.read_csv(file_path, sep='\t', header=None, names=column_names)

        #  get_dummies one-hot
        # data_encoded = pd.get_dummies(data, columns=['RACE', 'OCCUPATION', 'SECTOR'], drop_first=False)
        # data_encoded.to_excel("/data1/BCChen/cps/cps_encoded.xlsx", index=True)  # save data after preprocess

        file_path =f'{directory_path}cps/cps_encoded.xlsx'
        df = pd.read_excel(file_path, header=0, index_col=0)

        df = df.astype(float)
        y = df["WAGE"].values
        X = df.drop(columns=["WAGE"]).values
        y = y.astype(float)
        X = X.astype(float)

    # load Wine-white
    if name == "wine-white":
        file_path = f'{directory_path}wine_quality/winequality-white.csv'
        data = pd.read_csv(file_path, sep=';', header=0)
        y = data["quality"].values
        X = data.drop(columns=["quality"]).values

    # load Wine-red
    if name == "wine-red":
        file_path = f'{directory_path}wine_quality/winequality-red.csv'
        data = pd.read_csv(file_path, sep=';', header=0)
        y = data["quality"].values
        X = data.drop(columns=["quality"]).values

    # load concrete-cs-slump
    if name == "concrete-cs-slump":
        # detail
        # with open("/data1/BCChen/concrete_slump_test/slump_test.names", 'r') as f:
        #     print(f.read())
        # file_path = "/data1/BCChen/concrete_slump_test/slump_test.data"
        # df = pd.read_csv(file_path, header=0, index_col=0)
        # df.to_excel("/data1/BCChen/concrete_slump_test/slump_test_done.xlsx", index=True)  # excel save

        file_path = f'{directory_path}concrete_slump_test/slump_test_done.xlsx'
        df = pd.read_excel(file_path, header=0, index_col=0)
        output_column = ['SLUMP(cm)', 'FLOW(cm)', 'Compressive Strength (28-day)(Mpa)']

        y = df[output_column[0]].values
        X = df.drop(columns=output_column).values

    # load concrete-cs-flow
    if name == "concrete-cs-flow":
        file_path = f'{directory_path}concrete_slump_test/slump_test_done.xlsx'
        df = pd.read_excel(file_path, header=0, index_col=0)
        output_column = ['SLUMP(cm)', 'FLOW(cm)', 'Compressive Strength (28-day)(Mpa)']

        y = df[output_column[1]].values
        X = df.drop(columns=output_column).values

    # load concrete-cs-mpa
    if name == "concrete-cs-mpa":
        file_path = f'{directory_path}concrete_slump_test/slump_test_done.xlsx'
        df = pd.read_excel(file_path, header=0, index_col=0)
        output_column = ['SLUMP(cm)', 'FLOW(cm)', 'Compressive Strength (28-day)(Mpa)']

        y = df[output_column[2]].values
        X = df.drop(columns=output_column).values

    # load IEMOCAP-V
    if name == "IEMOCAP-V":
        file_path_data = f'{directory_path}IEMOCAP/IEMOCAP_data.xlsx'
        df = pd.read_excel(file_path_data, header=None)
        X = df.values
        file_path_label = f'{directory_path}IEMOCAP/IEMOCAP_label.xlsx'
        df = pd.read_excel(file_path_label, header=None)
        y = df.iloc[:, 0].values

    # load IEMOCAP-A
    if name == "IEMOCAP-A":
        file_path_data = f'{directory_path}IEMOCAP/IEMOCAP_data.xlsx'
        df = pd.read_excel(file_path_data, header=None)
        X = df.values
        file_path_label = f'{directory_path}IEMOCAP/IEMOCAP_label.xlsx'
        df = pd.read_excel(file_path_label, header=None)
        y = df.iloc[:, 1].values

    # load IEMOCAP-D
    if name == "IEMOCAP-D":
        file_path_data = f'{directory_path}IEMOCAP/IEMOCAP_data.xlsx'
        df = pd.read_excel(file_path_data, header=None)
        X = df.values
        file_path_label = f'{directory_path}IEMOCAP/IEMOCAP_label.xlsx'
        df = pd.read_excel(file_path_label, header=None)
        y = df.iloc[:, 2].values

    # load airfoil
    if name == "airfoil":
        file_path_data =f'{directory_path}airfoil/airfoil_self_noise.dat'
        df = pd.read_csv(file_path_data, header=None)
        data = df.values
        processed_data = [list(map(float, row[0].split())) for row in data]
        data_array = np.array(processed_data)
        X = data_array[:, :-1]
        y = data_array[:, -1]

    # load Autompg
    if name == "autompg":
        file_path = f'{directory_path}auto_mpg/auto-mpg.data'
        df = pd.read_csv(file_path, header=None, sep='\s+', na_values='?')
        print(df.head())

        # missing value
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        # non-numerical
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns
        df[non_numeric_cols] = df[non_numeric_cols].fillna("unknown")

        #  get_dummies   one-hot
        df = pd.get_dummies(df, columns=[7], drop_first=False)

        X = df.iloc[:, 1:-1].values
        y = df.iloc[:, 0].values

    # load Housing
    if name == "housing":
        file_path = f'{directory_path}housing/housing.xlsx'
        df = pd.read_excel(file_path, header=None)
        data = df.values

        processed_data = []
        for row in data:
            values = row[0].split()
            float_values = [float(val) for val in values]
            processed_data.append(float_values)

        data_array = np.array(processed_data)

        X = data_array[:, :-1]
        y = data_array[:, -1]

    # load PM10
    if name == "PM10":
        file_path = f'{directory_path}PM10/PM10.txt'
        df = pd.read_csv(file_path,header=None)
        data = df.values

        X = data[:, :-1]
        y = data[:, -1]

    # load NO2
    if name == "NO2":
        file_path = f'{directory_path}NO2/NO2.txt'
        df = pd.read_csv(file_path, sep='\t', header=None)
        data = df.values

        X = data[:, :-1]
        y = data[:, -1]

    # load wind
    if name == "wind":
        file_path = f'{directory_path}wind/wind.txt'
        df = pd.read_csv(file_path, sep='\s+', header=None)
        data = df.values
        # print(data)

        X = data[:, 3:-1]
        y = data[:, -1]

    # load Bike
    if name == "bike-sharing":
        def preprocess_bike_data(df):
            df_processed = df.copy()
            # time feature
            df_processed['is_rush_hour'] = ((df_processed['hr'] >= 7) & (df_processed['hr'] <= 9)) | \
                                           ((df_processed['hr'] >= 17) & (df_processed['hr'] <= 19))
            df_processed['is_night'] = (df_processed['hr'] >= 22) | (df_processed['hr'] <= 6)
            df_processed['is_weekend'] = (df_processed['weekday'] >= 5).astype(int)

            # weather feature
            weathersit_dummies = pd.get_dummies(df_processed['weathersit'], prefix='weathersit', drop_first=False)
            df_processed = pd.concat([df_processed, weathersit_dummies], axis=1)

            # season feature
            season_dummies = pd.get_dummies(df_processed['season'], prefix='season', drop_first=False)
            df_processed = pd.concat([df_processed, season_dummies], axis=1)

            # Hour feature
            def map_hour_to_period(hour):
                if 0 <= hour < 6:
                    return 'night'
                elif 6 <= hour < 12:
                    return 'morning'
                elif 12 <= hour < 18:
                    return 'afternoon'
                else:
                    return 'evening'

            df_processed['time_period'] = df_processed['hr'].apply(map_hour_to_period)
            period_dummies = pd.get_dummies(df_processed['time_period'], prefix='period')
            df_processed = pd.concat([df_processed, period_dummies], axis=1)

            return df_processed

        file_path = f'{directory_path}bike-sharing/hour.csv'
        df = pd.read_csv(file_path)  #

        # data preprocess
        df_processed = preprocess_bike_data(df)

        # cnt-target     remove redundant time feature/ target feature
        X = df_processed.drop(['cnt', 'casual', 'registered', 'dteday', 'instant', 'mnth', 'hr', 'weekday', 'season', 'weathersit', 'time_period'], axis=1, errors='ignore')
        y = df_processed['cnt']

        # X = X.values[:12000]
        # y = y.values[:12000]

    # load cahouse
    if name == "cahouse":
        file_path = f'{directory_path}cahouse/cahouse.txt'
        df = pd.read_csv(file_path, sep='\s+', header=None)
        data = df.values

        X = data[:, :-1]
        y = data[:, -1]

    # load tecator
    if name == "tecator":
        file_path = f'{directory_path}tecator/tecator.txt'
        df = pd.read_csv(file_path, sep='\s+', header=None)
        data = df.values
        # print(data)

        X = data[:, :-1]
        y = data[:, -1]

    # load carbon-u
    if name == "carbon-u":
        file_path = f'{directory_path}carbon/carbon_nanotubes.csv'
        df = pd.read_csv(file_path, sep=';', decimal=',')

        feature_columns = ['Chiral indice n', 'Chiral indice m',
                          'Initial atomic coordinate u',
                          'Initial atomic coordinate v',
                          'Initial atomic coordinate w']

        target_columns = ['Calculated atomic coordinates u\'',
                          'Calculated atomic coordinates v\'',
                          'Calculated atomic coordinates w\'']

        X = df[feature_columns].values
        y = df[target_columns[0]].values

        # print(f"dataset shpe: X {X.shape}, y {y.shape}")
        # print(f"feature_columns: {feature_columns}")
        # print(f"target_columns: {target_columns[0]}")

    # load carbon-v
    if name == "carbon-v":
        file_path = f'{directory_path}carbon/carbon_nanotubes.csv'
        df = pd.read_csv(file_path, sep=';', decimal=',')

        feature_columns = ['Chiral indice n', 'Chiral indice m',
                           'Initial atomic coordinate u',
                           'Initial atomic coordinate v',
                           'Initial atomic coordinate w']

        target_columns = ['Calculated atomic coordinates u\'',
                          'Calculated atomic coordinates v\'',
                          'Calculated atomic coordinates w\'']

        X = df[feature_columns].values
        y = df[target_columns[1]].values

        # print(f"dataset shape: X {X.shape}, y {y.shape}")
        # print(f"feature_columns: {feature_columns}")
        # print(f"target_columns: {target_columns[1]}")

    # load carbon-w
    if name == "carbon-w":
        file_path = f'{directory_path}carbon/carbon_nanotubes.csv'
        df = pd.read_csv(file_path, sep=';', decimal=',')

        feature_columns = ['Chiral indice n', 'Chiral indice m',
                           'Initial atomic coordinate u',
                           'Initial atomic coordinate v',
                           'Initial atomic coordinate w']

        target_columns = ['Calculated atomic coordinates u\'',
                          'Calculated atomic coordinates v\'',
                          'Calculated atomic coordinates w\'']

        X = df[feature_columns].values
        y = df[target_columns[2]].values

        print(f"dataset shape: X {X.shape}, y {y.shape}")
        print(f"feature_columns: {feature_columns}")
        print(f"target_columns: {target_columns[2]}")

    # load NPP_kMc
    if name == "NPP_kMc":
        file_path = f'{directory_path}NPP/data.txt'
        df = pd.read_csv(file_path, sep='\s+', header=None)

        feature_names = [
            'lp', 'v', 'GTT', 'GTn', 'GGn', 'Ts', 'Tp', 'T48', 'T1', 'T2',
            'P48', 'P1', 'P2', 'Pexh', 'TIC', 'mf',
            'kMc', 'kMt'
        ]

        df.columns = feature_names
        X = df.iloc[:, :-2].values
        y = df.iloc[:, -2] .values


    # load NPP_kMt
    if name == "NPP_kMt":
        file_path = f'{directory_path}NPP/data.txt'
        df = pd.read_csv(file_path, sep='\s+', header=None)

        feature_names = [
            'lp', 'v', 'GTT', 'GTn', 'GGn', 'Ts', 'Tp', 'T48', 'T1', 'T2',
            'P48', 'P1', 'P2', 'Pexh', 'TIC', 'mf',
            'kMc', 'kMt'
        ]

        df.columns = feature_names
        X = df.iloc[:, :-2] .values
        y = df.iloc[:, -1] .values

    print(X.shape)
    print(y.shape)

    # feature redundant
    if redundancy == True:
        redundant_X = X[:, :2]
        X = np.hstack((X, redundant_X))
        print("====redundant====")
        print(X.shape)
        print(y.shape)

    return X, y


if __name__ == '__main__':
    data_load("autompg", redundancy=False)
