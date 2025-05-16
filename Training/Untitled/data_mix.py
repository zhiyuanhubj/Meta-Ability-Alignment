import os
import pandas as pd

def mix_train_data_parquet(
    base_path, 
    difficulty_sample_map, 
    output_parquet="train_mixed.parquet",
    random_seed=42
):
    """
    从 base_path 下各个难度级别子目录中只读取 train.parquet 文件，
    按照 difficulty_sample_map 指定的数量进行随机采样并合并输出。
    
    :param base_path: 数据主目录，如 "/data/k"
    :param difficulty_sample_map: dict, 键为难度级别(1~5)，值为需要采样的数量
    :param output_parquet: 输出合并后的 parquet 文件名
    :param random_seed: 随机种子，便于结果复现
    """
    
    all_dfs = []
    
    # 遍历每个难度级别
    for difficulty, n_samples in difficulty_sample_map.items():
        difficulty_path = os.path.join(base_path, str(difficulty))
        
        # 只读取文件名为 train.parquet
        train_file_path = os.path.join(difficulty_path, "train.parquet")
        print(train_file_path)
        
        if not os.path.exists(train_file_path):
            print(f"警告：难度 {difficulty} 目录下没有找到 train.parquet 文件。")
            continue
        
        # 读取该难度下的 train.parquet
        df = pd.read_parquet(train_file_path)
        
        # 如果需要的数量大于可用数据，则取最小值，避免报错
        actual_n_samples = min(n_samples, len(df))
        
        # 随机抽样
        df_sampled = df.sample(
            n=actual_n_samples, 
            random_state=random_seed
        ).copy()
        
        # 可以给数据加一列 difficulty 标注
        df_sampled["difficulty"] = difficulty
        
        all_dfs.append(df_sampled)
    
    # 合并所有难度级别数据
    if not all_dfs:
        print("没有任何数据被读取或采样，请检查目录结构和文件是否正确。")
        return
    
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # 输出到 parquet 文件
    final_df.to_parquet(output_parquet)
    print(f"合并后的数据已保存到 {output_parquet}，总样本数：{len(final_df)}")

if __name__ == "__main__":
    # 配置：各难度需要的采样数量
    difficulty_sample_map = {
        1: 250,
        2: 225,
        3: 200,
        4: 125,
        5: 100
    }
    
    base_path = "/fsx/home/zhiyuan/logic-rl-formula/data/kk"  # 你的数据所在主目录
    output_parquet = "train_mixed.parquet"  # 结果输出文件
    
    mix_train_data_parquet(
        base_path=base_path, 
        difficulty_sample_map=difficulty_sample_map, 
        output_parquet=output_parquet,
        random_seed=42
    )

