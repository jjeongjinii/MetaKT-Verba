


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess_meta_kt(df):



    
    use_cols = [
        'ITEST_id', 'skill', 'correct', 
        'timeTaken', 'hintCount', 'attemptCount',
        'RES_CONFUSED', 'RES_CONCENTRATING', 'RES_BORED', 'RES_FRUSTRATED', 'RES_OFFTASK'
    ]
    
    
    
    df = df.dropna(subset=['skill', 'ITEST_id', 'correct']).copy() 
    df = df[df['correct'].isin([0, 1])].copy() 

    
    df['startTime'] = pd.to_numeric(df['startTime'], errors='coerce')
    df['endTime'] = pd.to_numeric(df['endTime'], errors='coerce')
    df = df.sort_values(by=['ITEST_id', 'startTime']).copy()

    
    df['prev_endTime'] = df.groupby('ITEST_id')['endTime'].shift(1)
    df['lag_time'] = (df['startTime'] - df['prev_endTime']).fillna(0).clip(lower=0)

    
    df['elapsed_time'] = df['timeTaken'].fillna(0).clip(lower=0, upper = 3600) 

    
    df['lag_time_log'] = np.log1p(df['lag_time'])
    df['elapsed_time_log'] = np.log1p(df['elapsed_time'])
    lag_bins = [0, 60, 300, 900, 1800, 3600, 10800, 86400, 259200, 604800]
    elapsed_bins = [0, 2, 5, 10, 20, 30, 60, 120, 300, 600]

    
    df['attempt_bin'] = np.clip(df['attemptCount'].fillna(0).astype(int).values, 0, 10)

    
    df['lag_bin'] = np.digitize(df['lag_time'].values, lag_bins)
    df['elapsed_bin'] = np.digitize(df['elapsed_time'].values, elapsed_bins)
    df['attempt_bin'] = np.digitize(df['attempt_bin'].values, range(11))
    

    
    
    df['Ln-1'] = df['Ln-1'].fillna(0)
    df['RES_GAMING'] = df['RES_GAMING'].fillna(0)

    
    num_features_to_scale = ['hintCount', 'attemptCount', 'lag_time_log', 'elapsed_time_log', 'Ln-1', 'RES_GAMING']
    scaler = MinMaxScaler()
    df[num_features_to_scale] = scaler.fit_transform(df[num_features_to_scale].values)

    res_cols = ['RES_CONFUSED', 'RES_CONCENTRATING', 'RES_BORED', 'RES_FRUSTRATED', 'RES_OFFTASK']
    df[res_cols] = df[res_cols].fillna(0)


    
    '''
    # Inverse-U (Gaussian-like) Function 적용: 너무 빠르거나 느린 응답은 확신도가 낮다고 판단
    def calculate_sc(group):
        # timeTaken이 0인 경우 로그 변환 에러 방지 (최소값 부여)
        rt = group.replace(0, 1e-5)
        log_rt = np.log(rt)
        mu_kc = log_rt.mean()
        sigma_kc = log_rt.std()
        if sigma_kc == 0 or np.isnan(sigma_kc): sigma_kc = 1.0
        # SC = exp(-((log_rt - mu_kc)^2) / (2 * sigma_kc^2))       
        return np.exp(-((log_rt - mu_kc) ** 2) / (2 * (sigma_kc ** 2)))
    
    df['SC'] = df.groupby('skill')['elapsed_time'].transform(calculate_sc)
    '''
    def calculate_personalized_sc(df):
        
        df['log_rt'] = np.log(df['elapsed_time'].replace(0, 1e-5))
        
        
        
        user_baseline = df.groupby('ITEST_id')['log_rt'].mean().rename('mu_u')
        df = df.join(user_baseline, on='ITEST_id')
        
        
        
        skill_stats = df.groupby('skill')['log_rt'].agg(['mean', 'std']).rename(columns={'mean': 'mu_j', 'std': 'sigma_j'})
        df = df.join(skill_stats, on='skill')
        
        
        global_mean = df['log_rt'].mean()
        
        
        
        df['d_j'] = df['mu_j'] - global_mean
        
        
        
        df['target_log_rt'] = df['mu_u'] + df['d_j']
        
        
        
        df['sigma_j'] = df['sigma_j'].replace(0, 1.0).fillna(1.0)
        
        df['SC'] = np.exp(-((df['log_rt'] - df['target_log_rt']) ** 2) / (2 * (df['sigma_j'] ** 2)))
        
        return df['SC']

    df['SC'] = calculate_personalized_sc(df)
    
    
    # [Calibration Group]
    
    df['m_overconfidence'] = (1 - df['correct']) * df['SC'] * (1 - df['RES_CONFUSED'])
    
    df['m_underconfidence'] = df['correct'] * (1 - df['SC']) * df['RES_CONFUSED']

    # [Self-Regulated Learning Group]
    
    df['m_strategic_help'] = df['RES_CONFUSED'] * df['hintCount']
    df['m_cognitive_avoidance'] = (df['RES_CONFUSED'] - df['hintCount']).abs()

    # [ZPD & Struggle Group]
    
    df['m_productive_struggle'] = df['correct'] * df['RES_CONFUSED'] * (1 - df['hintCount']) * df['SC']
    df['m_unproductive_frustration'] = (1 - df['correct']) * df['RES_FRUSTRATED'] * (1 - df['hintCount'])

    # [BKT Noise Group]
    
    df['m_lucky_guess'] = df['correct'] * (1 - df['hintCount']) * (1 - df['SC']) * df['RES_CONFUSED']
    
    df['m_slipping'] = (1 - df['correct']) * (1 - df['hintCount']) * df['SC'] * df['RES_CONCENTRATING']

    # [Flow Group]
    df['m_boredom_offtask'] = df['RES_BORED'] * df['correct'] * (1 - df['RES_CONCENTRATING'])

    return df


def load_and_merge_data(log_folder_path, label_path):


    import glob
    all_files = glob.glob(f"{log_folder_path}/student_log_*.csv")
    df_list = []

    for filename in all_files:
        df_list.append(pd.read_csv(filename, low_memory=False))

    full_df = pd.concat(df_list, axis = 0, ignore_index=True)
    
    labels = pd.read_csv(label_path)

    
    target_ids = labels['ITEST_id'].unique()

    
    filtered_df = full_df[full_df['ITEST_id'].isin(target_ids)].copy()

    return filtered_df
