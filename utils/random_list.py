import itertools

def lcg(modulus, a, c, seed):
    """线性同余生成器"""
    while True:
        seed = (a * seed + c) % modulus
        yield seed

def get_random_number(probabilities, values, generator):
    assert len(probabilities) == len(values), "Length of probabilities and values must be the same"
    assert abs(sum(probabilities) - 1) < 1e-6, "Probabilities must sum to 1"

    random_float = next(generator) / 256.0  # 转换为0-1之间的浮点数

    for value, accumulated_probability in zip(values, itertools.accumulate(probabilities)):
        if random_float < accumulated_probability:
            return value
    return values[-1]  # 如果由于浮点数精度问题没有返回任何值，返回最后一个值

def get_random_list(probabilities, values, length, seed: int = 0):
    # 创建一个生成器
    generator = lcg(modulus=256, a=1103515245, c=12345, seed=seed)
    return [get_random_number(probabilities, values, generator) for _ in range(length)]