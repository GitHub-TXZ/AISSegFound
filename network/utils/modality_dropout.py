import torch

def simulate_missing_modalities(num_modalities, missing_prob=0.5):
    """
    使用伯努利分布模拟模态缺失，确保至少有一个模态不缺失。
    
    参数:
        num_modalities (int): 模态总数。
        missing_prob (float): 每个模态缺失的概率。
    
    返回:
        list: 包含缺失模态索引的列表。
    """
    while True:
        # 使用伯努利分布生成模态缺失指示器
        # p=missing_prob 表示缺失的概率
        indicators = torch.bernoulli(torch.full((num_modalities,), missing_prob)).bool()
        
        # 确保至少有一个模态不缺失
        if not indicators.all():  # 如果不是所有模态都缺失
            missing_modalities = torch.nonzero(indicators, as_tuple=True)[0].tolist()
            return missing_modalities


if __name__ == "__main__":
    # 示例：三个模态的模拟缺失
    num_modalities = 3  # NCCT, CTP_T_Param, CTP_V_Param
    missing_modalities = simulate_missing_modalities(num_modalities, missing_prob=0.5)
    print(f"Missing modalities: {missing_modalities}")


    # 示例数据
    NCCT_data = torch.randn(1, 1, 64, 64)  # NCCT 数据
    CTP_T_Param_data = torch.randn(1, 1, 64, 64)  # CTP_T_Param 数据
    CTP_V_Param_data = torch.randn(1, 1, 64, 64)  # CTP_V_Param 数据

    # 模拟缺失模态
    missing_modalities = simulate_missing_modalities(num_modalities=3, missing_prob=0.5)

    # 将缺失模态的数据设置为零张量
    modalities = [NCCT_data, CTP_T_Param_data, CTP_V_Param_data]
    for idx in missing_modalities:
        modalities[idx] = torch.zeros_like(modalities[idx])

    # 输出结果
    print(f"Missing modalities: {missing_modalities}")
    for i, data in enumerate(modalities):
        print(f"Modality {i} shape: {data.shape}, is_zero: {torch.all(data == 0)}")