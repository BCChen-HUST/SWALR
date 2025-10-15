import re
import pandas as pd
import numpy as np


def data_load(name, redundancy=False):
    # directory_path = "C:/Users/DELL/Desktop/sw-ALR/dataset/"
    directory_path= "/data1/BCChen/sw-ALR/dataset/"
    # 加载 混凝土数据集
    if name == "concrete":
        file_path = f'{directory_path}concrete_com_str/Concrete_Data.xls'  # 请替换为你的文件路径
        df = pd.read_excel(file_path)
        X = df.iloc[:, :-1].values  # 取前八列作为特征
        y = df.iloc[:, -1].values  # 取最后一列作为标签

    # 加载 能源数据集1
    if name == "energy-heat":
        file_path = f'{directory_path}energy_efficiency/ENB2012_data.xlsx'  # 请替换为你的文件路径
        df = pd.read_excel(file_path)
        X = df.iloc[:, :-2].values  # 取特征
        y = df.iloc[:, -2].values  # 取第一个回归值

    # 加载 能源数据集2
    if name == "energy-cool":
        file_path = f'{directory_path}energy_efficiency/ENB2012_data.xlsx'  # 请替换为你的文件路径
        df = pd.read_excel(file_path)
        X = df.iloc[:, :-2].values  # 取特征
        y = df.iloc[:, -1].values  # 取第一个回归值

    # 加载 yacht数据集
    if name == "yacht":
        file_path = f'{directory_path}yacht_hydrodynamics/yacht_hydrodynamics.data'  # 请替换为你的文件路径
        df = pd.read_csv(file_path, header=None)
        data = df.values
        processed_data = [list(map(float, row[0].split())) for row in data]
        data_array = np.array(processed_data)
        X = data_array[:, :-1]  # 取特征
        y = data_array[:, -1]  # 取回归值

    # 加载 abalone鲍鱼数据集
    if name == "abalone":
        # # 打开 .names 文件查看数据集描述
        # with open("/data1/BCChen/abalone/abalone.names", 'r') as f:
        #     print(f.read())
        file_path = f'{directory_path}abalone/abalone.data'  # 请替换为你的文件路径
        df = pd.read_csv(file_path, header=None)
        # 使用 get_dummies 对 'Gender' 列进行独热编码
        df_encoded = pd.get_dummies(df, columns=[0], drop_first=False)  # drop_first=True 可以避免多重共线性
        y = df_encoded[4].values
        X = df_encoded.drop(columns=[4]).values

    # 加载 cps数据集  statlib
    if name == "cps":
        # #处理数据
        # file_path = "/data1/BCChen/cps/cps.txt"  # 请替换为你的文件路径
        # column_names = ['EDUCATION', 'SOUTH', 'SEX', 'EXPERIENCE', 'UNION', 'WAGE', 'AGE', 'RACE',
        #                 'OCCUPATION', 'SECTOR', 'MARR']
        # data = pd.read_csv(file_path, sep='\t', header=None, names=column_names)
        # # 使用 get_dummies 独热编码
        # data_encoded = pd.get_dummies(data, columns=['RACE', 'OCCUPATION', 'SECTOR'], drop_first=False)  # drop_first=True 可以避免多重共线性
        # data_encoded.to_excel("/data1/BCChen/cps/cps_encoded.xlsx", index=True)  # 将 DataFrame 写入 Excel 文件
        file_path =f'{directory_path}cps/cps_encoded.xlsx'  # 请替换为你的文件路径
        df = pd.read_excel(file_path, header=0, index_col=0)
        # 确保 'WAGE' 列和其他所有列的数据类型都是 float
        df = df.astype(float)
        y = df["WAGE"].values
        X = df.drop(columns=["WAGE"]).values
        y = y.astype(float)
        X = X.astype(float)

        # 加载 WINE-white数据集
    if name == "wine-white":
        file_path = f'{directory_path}wine_quality/winequality-white.csv'  # 请替换为你的文件路径
        data = pd.read_csv(file_path, sep=';', header=0)
        # print(data)
        y = data["quality"].values
        X = data.drop(columns=["quality"]).values

        # 加载 WINE-red数据集
    if name == "wine-red":
        file_path = f'{directory_path}wine_quality/winequality-red.csv'  # 请替换为你的文件路径
        data = pd.read_csv(file_path, sep=';', header=0)
        # print(data)
        y = data["quality"].values
        X = data.drop(columns=["quality"]).values

    # 加载 concrete-cs-slump数据集
    if name == "concrete-cs-slump":
        # 打开 .names 文件查看数据集描述
        # with open("/data1/BCChen/concrete_slump_test/slump_test.names", 'r') as f:
        #     print(f.read())
        # file_path = "/data1/BCChen/concrete_slump_test/slump_test.data"  # 请替换为你的文件路径
        # df = pd.read_csv(file_path, header=0, index_col=0)
        # df.to_excel("/data1/BCChen/concrete_slump_test/slump_test_done.xlsx", index=True)  # 将 DataFrame 写入 Excel 文件
        # print(df)
        file_path = f'{directory_path}concrete_slump_test/slump_test_done.xlsx'  # 请替换为你的文件路径
        df = pd.read_excel(file_path, header=0, index_col=0)
        output_column = ['SLUMP(cm)', 'FLOW(cm)', 'Compressive Strength (28-day)(Mpa)']
        # print(df)
        y = df[output_column[0]].values
        X = df.drop(columns=output_column).values

    # 加载 concrete-cs-flow数据集
    if name == "concrete-cs-flow":
        file_path = f'{directory_path}concrete_slump_test/slump_test_done.xlsx'  # 请替换为你的文件路径
        df = pd.read_excel(file_path, header=0, index_col=0)
        output_column = ['SLUMP(cm)', 'FLOW(cm)', 'Compressive Strength (28-day)(Mpa)']
        # print(df)
        y = df[output_column[1]].values
        X = df.drop(columns=output_column).values

    # 加载 concrete-cs-mpa数据集
    if name == "concrete-cs-mpa":
        file_path = f'{directory_path}concrete_slump_test/slump_test_done.xlsx'  # 请替换为你的文件路径
        df = pd.read_excel(file_path, header=0, index_col=0)
        output_column = ['SLUMP(cm)', 'FLOW(cm)', 'Compressive Strength (28-day)(Mpa)']
        # print(df)
        y = df[output_column[2]].values
        X = df.drop(columns=output_column).values

    # # 加载 IEMOCAP-V数据集
    if name == "IEMOCAP-V":
        file_path_data = f'{directory_path}IEMOCAP/IEMOCAP_data.xlsx'
        df = pd.read_excel(file_path_data, header=None)
        X = df.values  # 取特征
        file_path_label = f'{directory_path}IEMOCAP/IEMOCAP_label.xlsx'
        df = pd.read_excel(file_path_label, header=None)
        y = df.iloc[:, 0].values  # 取特征

    # # 加载 IEMOCAP-A数据集
    if name == "IEMOCAP-A":
        file_path_data = f'{directory_path}IEMOCAP/IEMOCAP_data.xlsx'
        df = pd.read_excel(file_path_data, header=None)
        X = df.values  # 取特征
        file_path_label = f'{directory_path}IEMOCAP/IEMOCAP_label.xlsx'
        df = pd.read_excel(file_path_label, header=None)
        y = df.iloc[:, 1].values  # 取特征

    # # 加载 IEMOCAP-D数据集
    if name == "IEMOCAP-D":
        file_path_data = f'{directory_path}IEMOCAP/IEMOCAP_data.xlsx'
        df = pd.read_excel(file_path_data, header=None)
        X = df.values  # 取特征
        file_path_label = f'{directory_path}IEMOCAP/IEMOCAP_label.xlsx'
        df = pd.read_excel(file_path_label, header=None)
        y = df.iloc[:, 2].values  # 取特征

    # # 加载 airfoil数据集
    if name == "airfoil":
        file_path_data =f'{directory_path}airfoil/airfoil_self_noise.dat'
        df = pd.read_csv(file_path_data, header=None)
        data = df.values
        processed_data = [list(map(float, row[0].split())) for row in data]
        data_array = np.array(processed_data)
        X = data_array[:, :-1]  # 取特征
        y = data_array[:, -1]  # 取回归值

    # # 加载 autopg数据集
    if name == "autompg":
        file_path = f'{directory_path}auto_mpg/auto-mpg.data'
        df = pd.read_csv(file_path, header=None, sep='\s+', na_values='?')

        # 仅对数值列填充缺失值
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        # 处理非数值列（如汽车名称）
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns
        df[non_numeric_cols] = df[non_numeric_cols].fillna("unknown")

        # 提取特征和目标变量
        X = df.iloc[:, 1:-1].values  # 特征（从第2列开始，不要后面car name）
        y = df.iloc[:, 0].values  # 目标变量（第1列）


    # # 加载 housing数据集
    if name == "housing":
        file_path = f'{directory_path}housing/housing.xlsx'
        df = pd.read_excel(file_path, header=None)
        data = df.values  # 取特征

        # 处理数据
        processed_data = []
        for row in data:
            # 拆分字符串
            values = row[0].split()
            # 转换为浮点数
            float_values = [float(val) for val in values]
            processed_data.append(float_values)

        # 转换为NumPy数组
        data_array = np.array(processed_data)

        # 提取特征和目标变量
        X = data_array[:, :-1]  # 特征（除最后一列）
        y = data_array[:, -1]  # 目标变量（最后一列）

    # # 加载 PM10数据集
    if name == "PM10":
        file_path = f'{directory_path}PM10/PM10.txt'
        df = pd.read_csv(file_path,header=None)  # 明确指定制表符分隔
        data = df.values  # 直接转为 NumPy 数组
        # print(data)

        # 提取特征和目标变量（最后一列是目标）
        X = data[:, :-1]  # 所有行，除最后一列
        y = data[:, -1]  # 所有行，最后一列

    # # 加载 NO2数据集
    if name == "NO2":
        file_path = f'{directory_path}NO2/NO2.txt'
        df = pd.read_csv(file_path, sep='\t', header=None)  # 明确指定制表符分隔
        data = df.values  # 直接转为 NumPy 数组
        # print(data)

        # 提取特征和目标变量（最后一列是目标）
        X = data[:, :-1]  # 所有行，除最后一列
        y = data[:, -1]  # 所有行，最后一列

    # # 加载 wind数据集
    if name == "wind":
        file_path = f'{directory_path}wind/wind.txt'
        df = pd.read_csv(file_path, sep='\s+', header=None)  # 明确指定制表符分隔
        data = df.values  # 直接转为 NumPy 数组
        # print(data)

        # 提取特征和目标变量（最后一列是目标）
        X = data[:, 3:-1]  # 所有行，除最后一列
        y = data[:, -1]  # 所有行，最后一列

    # # 加载 bike数据集
    if name == "bike-sharing":
        def preprocess_bike_data(df):
            """
            对自行车共享数据进行预处理
            """
            df_processed = df.copy()

            # 1. 处理日期特征
            # if 'dteday' in df_processed.columns:
                # df_processed['dteday'] = pd.to_datetime(df_processed['dteday'])
                # df_processed['year'] = df_processed['dteday'].dt.year
                # df_processed['month'] = df_processed['dteday'].dt.month
                # df_processed['day'] = df_processed['dteday'].dt.day

            # 2. 创建更有意义的时间特征
            df_processed['is_rush_hour'] = ((df_processed['hr'] >= 7) & (df_processed['hr'] <= 9)) | \
                                           ((df_processed['hr'] >= 17) & (df_processed['hr'] <= 19))
            df_processed['is_night'] = (df_processed['hr'] >= 22) | (df_processed['hr'] <= 6)
            df_processed['is_weekend'] = (df_processed['weekday'] >= 5).astype(int)

            # 3. 创建天气分类
            # df_processed['is_good_weather'] = (df_processed['weathersit'] == 1).astype(int)
            # df_processed['is_bad_weather'] = (df_processed['weathersit'] >= 3).astype(int)
            weathersit_dummies = pd.get_dummies(df_processed['weathersit'], prefix='weathersit', drop_first=False)
            df_processed = pd.concat([df_processed, weathersit_dummies], axis=1)

            # 4. 创建季节特征
            season_dummies = pd.get_dummies(df_processed['season'], prefix='season', drop_first=False)
            df_processed = pd.concat([df_processed, season_dummies], axis=1)

            # # 5. 创建月份特征
            # month_dummies = pd.get_dummies(df_processed['mnth'], prefix='month')
            # df_processed = pd.concat([df_processed, month_dummies], axis=1)

            # 6. 创建小时特征（按时间段分组）
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
        df = pd.read_csv(file_path)  # 明确指定制表符分隔

        # 数据预处理
        df_processed = preprocess_bike_data(df)

        # 提取特征和目标变量
        # 使用cnt作为目标变量（总租车数量）
        X = df_processed.drop(['cnt', 'casual', 'registered', 'dteday', 'instant', 'mnth', 'hr', 'weekday', 'season', 'weathersit', 'time_period'], axis=1, errors='ignore')
        y = df_processed['cnt']
        # print(f"特征列表: {X.columns.tolist()}")
        # print(X[:2])
        # X = X.values[:12000]
        # y = y.values[:12000]

    # # 加载cahouse数据集
    if name == "cahouse":
        file_path = f'{directory_path}cahouse/cahouse.txt'
        df = pd.read_csv(file_path, sep='\s+', header=None)  # 明确指定制表符分隔
        data = df.values  # 直接转为 NumPy 数组
        # print(data)

        # 提取特征和目标变量（最后一列是目标）
        X = data[:, :-1]  # 所有行，除最后一列
        y = data[:, -1]  # 所有行，最后一列

    # # 加载tecator数据集
    if name == "tecator":
        file_path = f'{directory_path}tecator/tecator.txt'
        df = pd.read_csv(file_path, sep='\s+', header=None)  # 明确指定制表符分隔
        data = df.values  # 直接转为 NumPy 数组
        # print(data)

        # 提取特征和目标变量（最后一列是目标）
        X = data[:, :-1]  # 所有行，除最后一列
        y = data[:, -1]  # 所有行，最后一列

    # # 加载 carbon_u数据集
    if name == "carbon-u":
        file_path = f'{directory_path}carbon/carbon_nanotubes.csv'
        df = pd.read_csv(file_path, sep=';', decimal=',')

        # 定义特征和目标
        feature_columns = ['Chiral indice n', 'Chiral indice m',
                          'Initial atomic coordinate u',
                          'Initial atomic coordinate v',
                          'Initial atomic coordinate w']

        target_columns = ['Calculated atomic coordinates u\'',
                          'Calculated atomic coordinates v\'',
                          'Calculated atomic coordinates w\'']

        # 正确的方式提取特征和目标
        X = df[feature_columns].values
        y = df[target_columns[0]].values

        print(f"数据集形状: X {X.shape}, y {y.shape}")
        print(f"特征列: {feature_columns}")
        print(f"目标列: {target_columns[0]}")

    # # 加载 carbon_v数据集
    if name == "carbon-v":
        file_path = f'{directory_path}carbon/carbon_nanotubes.csv'
        df = pd.read_csv(file_path, sep=';', decimal=',')

        # 定义特征和目标
        feature_columns = ['Chiral indice n', 'Chiral indice m',
                           'Initial atomic coordinate u',
                           'Initial atomic coordinate v',
                           'Initial atomic coordinate w']

        target_columns = ['Calculated atomic coordinates u\'',
                          'Calculated atomic coordinates v\'',
                          'Calculated atomic coordinates w\'']

        # 正确的方式提取特征和目标
        X = df[feature_columns].values
        y = df[target_columns[1]].values

        print(f"数据集形状: X {X.shape}, y {y.shape}")
        print(f"特征列: {feature_columns}")
        print(f"目标列: {target_columns[1]}")

    # # 加载 carbon_w数据集 5维度
    if name == "carbon-w":
        file_path = f'{directory_path}carbon/carbon_nanotubes.csv'
        df = pd.read_csv(file_path, sep=';', decimal=',')

        # 定义特征和目标
        feature_columns = ['Chiral indice n', 'Chiral indice m',
                           'Initial atomic coordinate u',
                           'Initial atomic coordinate v',
                           'Initial atomic coordinate w']

        target_columns = ['Calculated atomic coordinates u\'',
                          'Calculated atomic coordinates v\'',
                          'Calculated atomic coordinates w\'']

        # 正确的方式提取特征和目标
        X = df[feature_columns].values
        y = df[target_columns[2]].values

        print(f"数据集形状: X {X.shape}, y {y.shape}")
        print(f"特征列: {feature_columns}")
        print(f"目标列: {target_columns[2]}")

    # # 加载 NPP数据集  16维特征
    if name == "NPP_kMc":
        file_path = f'{directory_path}NPP/data.txt'
        df = pd.read_csv(file_path, sep='\s+', header=None)

        # 特征名称（根据README）
        feature_names = [
            'lp', 'v', 'GTT', 'GTn', 'GGn', 'Ts', 'Tp', 'T48', 'T1', 'T2',
            'P48', 'P1', 'P2', 'Pexh', 'TIC', 'mf',  # 16个传感器特征
            'kMc', 'kMt'  # 2个目标变量
        ]

        df.columns = feature_names
        X = df.iloc[:, :-2].values  # 前16列是特征
        y = df.iloc[:, -2] .values # 最后2列是目标


    # # 加载 NPP数据集  16维特征
    if name == "NPP_kMt":
        file_path = f'{directory_path}NPP/data.txt'
        df = pd.read_csv(file_path, sep='\s+', header=None)

        # 特征名称（根据README）
        feature_names = [
            'lp', 'v', 'GTT', 'GTn', 'GGn', 'Ts', 'Tp', 'T48', 'T1', 'T2',
            'P48', 'P1', 'P2', 'Pexh', 'TIC', 'mf',  # 16个传感器特征
            'kMc', 'kMt'  # 2个目标变量
        ]

        df.columns = feature_names
        X = df.iloc[:, :-2] .values # 前16列是特征
        y = df.iloc[:, -1] .values # 最后2列是目标




    print(X.shape)
    print(y.shape)

    if redundancy == True:
        redundant_X = X[:, :2]
        X = np.hstack((X, redundant_X))
        print("====redundant====")
        print(X.shape)
        print(y.shape)

    # print(X.dtype)
    # print(y.dtype)
    # print(X[:20])
    # print(y[:20])
    # X = 0
    # y = 0
    return X, y


if __name__ == '__main__':
    data_load("wind",redundancy=False)
