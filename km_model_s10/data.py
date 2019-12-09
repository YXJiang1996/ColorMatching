import torch
import numpy as np
from itertools import combinations

# 21种涂料的反射光谱数据
ingredients = np.array([
    [0.2072717, 0.2314336, 0.2323573, 0.2326192, 0.2315759, 0.2308346, 0.2299640, 0.2295049, 0.2283248, 0.2267959,
     0.2265563, 0.2253986, 0.2243909, 0.2231159, 0.2221052, 0.2208150, 0.2201522, 0.2193270, 0.2182626, 0.2170505,
     0.2162608, 0.2146650, 0.2133304, 0.2119302, 0.2106560, 0.2094081, 0.2085703, 0.2077338, 0.2034682, 0.2023278,
     0.2005492],
    [0.3426769, 0.4345741, 0.4408946, 0.4417734, 0.4448756, 0.4509351, 0.4701447, 0.5371643, 0.6943693, 0.8247025,
     0.8914063, 0.9259378, 0.9373208, 0.9417730, 0.9441082, 0.9462348, 0.9473710, 0.9479665, 0.9482779, 0.9480201,
     0.9483707, 0.9478786, 0.9474954, 0.9472089, 0.9468048, 0.9471580, 0.9465347, 0.9441642, 0.9423221, 0.9387001,
     0.9329507],
    [0.3075271, 0.3970215, 0.4258692, 0.4498420, 0.4777384, 0.5187902, 0.5789000, 0.6485421, 0.7203640, 0.7698454,
     0.7815039, 0.7697773, 0.7418166, 0.7037384, 0.6568713, 0.6019219, 0.5399272, 0.4776893, 0.4137468, 0.3495456,
     0.2899768, 0.2505462, 0.2324735, 0.2232275, 0.2189511, 0.2174184, 0.2212247, 0.2353665, 0.2580868, 0.2678188,
     0.2783012],
    [0.3777322, 0.5588190, 0.6166437, 0.6508363, 0.6942729, 0.7478291, 0.7715947, 0.7660858, 0.7381392, 0.6950833,
     0.6416819, 0.5781880, 0.5089610, 0.4370953, 0.3696142, 0.3065603, 0.2513052, 0.2179904, 0.2023969, 0.1957846,
     0.1897067, 0.1847080, 0.1875919, 0.2009600, 0.2232523, 0.2416958, 0.2510498, 0.2490231, 0.2179781, 0.2109241,
     0.2085472],
    [0.3353273, 0.4391780, 0.4634378, 0.4844812, 0.5122489, 0.5352546, 0.5445021, 0.5498481, 0.5548403, 0.5667891,
     0.5892876, 0.6205880, 0.6589307, 0.7075461, 0.7588425, 0.8063425, 0.8438276, 0.8656531, 0.8760347, 0.8760190,
     0.8742780, 0.8694988, 0.8650322, 0.8608174, 0.8578521, 0.8574630, 0.8571312, 0.8569201, 0.8584669, 0.8579510,
     0.8594775],
    [0.3582383, 0.4589019, 0.4442960, 0.4180548, 0.3943129, 0.3753622, 0.3580040, 0.3467267, 0.3377595, 0.3309189,
     0.3229960, 0.3138701, 0.3117055, 0.3117619, 0.3016199, 0.2815702, 0.2676951, 0.2725082, 0.3257494, 0.5337213,
     0.7479121, 0.8533681, 0.9027772, 0.9256366, 0.9336862, 0.9368781, 0.9375737, 0.9365737, 0.9347985, 0.9322762,
     0.9260198],
    [0.2712101, 0.3016053, 0.2955334, 0.2895256, 0.2859294, 0.2884298, 0.2914248, 0.2936781, 0.3083987, 0.3733792,
     0.5332463, 0.7305964, 0.8455071, 0.9001799, 0.9261169, 0.9373263, 0.9413789, 0.9437845, 0.9448064, 0.9448079,
     0.9450820, 0.9446973, 0.9445449, 0.9442747, 0.9437326, 0.9442004, 0.9438083, 0.9414658, 0.9391597, 0.9355036,
     0.9289310],
    [0.3239402, 0.3913669, 0.3892914, 0.3797518, 0.3718012, 0.3679169, 0.3649561, 0.3648219, 0.3724960, 0.4103487,
     0.5227775, 0.6892728, 0.8143758, 0.8887466, 0.9251493, 0.9383553, 0.9419549, 0.9441490, 0.9450906, 0.9451738,
     0.9456115, 0.9452887, 0.9449737, 0.9445507, 0.9441320, 0.9446933, 0.9444050, 0.9424030, 0.9397488, 0.9361140,
     0.9302571],
    [0.3599522, 0.4651332, 0.4698422, 0.4624575, 0.4524772, 0.4450434, 0.4392531, 0.4362574, 0.4357192, 0.4629371,
     0.5635376, 0.7249229, 0.8456681, 0.9069895, 0.9324164, 0.9409078, 0.9437131, 0.9453641, 0.9461564, 0.9460989,
     0.9463242, 0.9454620, 0.9451574, 0.9448677, 0.9443573, 0.9447857, 0.9443947, 0.9421353, 0.9400899, 0.9365174,
     0.9304954],
    [0.2877704, 0.3306401, 0.3257476, 0.3195266, 0.3157697, 0.3176763, 0.3205532, 0.3241252, 0.3338507, 0.3885163,
     0.5312571, 0.7235649, 0.8461311, 0.9063247, 0.9318116, 0.9404034, 0.9426319, 0.9439566, 0.9447455, 0.9449095,
     0.9453977, 0.9448299, 0.9444428, 0.9442645, 0.9441012, 0.9446317, 0.9440655, 0.9414658, 0.9391804, 0.9359795,
     0.9293869],
])
# 对应的初始浓度
cInit = np.array(
    [0.51, 0.51, 0.497, 0.51, 0.504, 0.503,
     0.51, 0.61, 0.51, 0.5])
# 白漆的反射光谱数据
background = np.array(
    [0.4519222, 0.7445221, 0.8898484, 0.9311465, 0.9374331, 0.9395607, 0.9426168, 0.9435278, 0.9453126, 0.9456188,
     0.9471663, 0.9475049, 0.9473154, 0.9476760, 0.9473940, 0.9471831, 0.9469163, 0.9463666, 0.9459230, 0.9450417,
     0.9446035, 0.9434228, 0.9427990, 0.9421890, 0.9418073, 0.9421464, 0.9415964, 0.9395502, 0.9375165, 0.9339727,
     0.9281333])
# 将白漆的反射光谱数据和涂料的浓度，反射光谱数据合并
info = background.copy()
for i, c in enumerate(cInit):
    info = np.append(info, c)
    info = np.append(info, ingredients[i])
# CIE标准照明体D65光源，10°视场
optical_relevant = np.array(
    [[0.136, 0.667, 1.644, 2.348, 3.463, 3.733, 3.065, 1.934, 0.803, 0.151, 0.036, 0.348, 1.062,
      2.192, 3.385, 4.744, 6.069, 7.285, 8.361, 8.537, 8.707, 7.946, 6.463, 4.641, 3.109, 1.848,
      1.053, 0.575, 0.275, 0.120, 0.059],
     [0.014, 0.069, 0.172, 0.289, 0.560, 0.901, 1.300, 1.831, 2.530, 3.176, 4.337, 5.629, 6.870,
      8.112, 8.644, 8.881, 8.583, 7.922, 7.163, 5.934, 5.100, 4.071, 3.004, 2.031, 1.295, 0.741,
      0.416, 0.225, 0.107, 0.046, 0.023],
     [0.613, 3.066, 7.820, 11.589, 17.755, 20.088, 17.697, 13.025, 7.703, 3.889, 2.056, 1.040,
      0.548, 0.282, 0.123, 0.036, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
      0.000, 0.000, 0.000, 0.000, 0.000, 0.000]])
# 纯白
perfect_white = np.array([[94.83], [100.00], [107.38]])


def generate(total_dataset_size, model='km', ydim=31, info=info, prior_bound=[0, 1, 0, 1], seed=0):
    np.random.seed(seed)
    N = total_dataset_size

    # 获取涂料信息
    background = info[:ydim]  # 背景
    colors = np.arange(0, (info.shape[-1] - ydim) // (ydim + 1), 1)  # 从0到涂料种类数
    initial_concentration = np.zeros(colors.size * 1)  # 初始化浓度为0
    ingredients = np.zeros(colors.size * ydim).reshape(colors.size, ydim)  # 初始化分光反射率为0
    # 涂料信息的初始化
    for i, c in enumerate(colors):
        initial_concentration[i] = info[ydim + i * (ydim + 1)]
        ingredients[i] = info[ydim + i * (ydim + 1) + 1:ydim + (i + 1) * (ydim + 1)]
    # 初始化浓度信息，从0-1的均匀分布中随机采样，维度为：生成样本数*涂料种数
    concentrations = np.random.uniform(0, 1, size=(N, colors.size))
    # 将浓度信息约束到指定的范围中
    for i in colors:
        concentrations[:, i] = prior_bound[0] + (prior_bound[1] - prior_bound[0]) * concentrations[:, i]

    # 在21中颜色中选18种的排列组合
    r = list(combinations(np.arange(0, colors.size, 1), 7))
    r_num = r.__len__()
    n = N // r_num
    # 根据排列组合将对应的位置为0
    for i in range(r_num - 1):
        concentrations[i * n:(i + 1) * n, r[i]] = 0.
    concentrations[(r_num - 1) * n:, r[r_num - 1]] = 0.
    # 波长的序列和索引序列
    xvec = np.arange(400, 710, 10)
    xidx = np.arange(0, ydim, 1)
    # 原本为21*1，重复为ydim=31次，再转变为21*31
    initial_conc_array = np.repeat(initial_concentration.reshape(colors.size, 1), ydim).reshape(colors.size, ydim)

    # 使用km模型的情况
    if model == 'km':
        # 基底的K/S值，论文公式4-6a
        fsb = (np.ones_like(background) - background) ** 2 / (background * 2)
        # 各种色浆的单位K/S值，论文公式4-6b
        fst = ((np.ones_like(ingredients) - ingredients) ** 2 / (ingredients * 2) - fsb) / initial_conc_array
        # ydim*N的0
        fss = np.zeros(N * ydim).reshape(ydim, N)
        # 涂料的K/S值,论文公式4-6c
        for i in xidx:
            for j in colors:
                fss[i, :] += concentrations[:, j] * fst[j, i]
            fss[i, :] += np.ones(N) * fsb[i]
        # 涂料的分光反射率，论文公式4-6d
        reflectance = fss - ((fss + 1) ** 2 - 1) ** 0.5 + 1
        # 转置
        reflectance = reflectance.transpose()
    else:
        print('Sorry no model of that name')
        exit(1)

    # 对数据进行打乱
    shuffling = np.random.permutation(N)
    concentrations = torch.tensor(concentrations[shuffling], dtype=torch.float)
    reflectance = torch.tensor(reflectance[shuffling], dtype=torch.float)
    # concentrations:各个样本对应的浓度
    # reflectance:配方对应的分光反射率
    # xvec:400-710,波长的取值
    # info:基础信息
    return concentrations, reflectance, xvec, info

generate(100)
def get_lik(ydata, n_grid=64, info=info, model='km', bound=[0, 1, 0, 1]):
    mcx = np.linspace(bound[0], bound[1], n_grid)
    dmcx = mcx[1] - mcx[0]

    # Get painting information
    ydim = ydata.size
    background = info[:ydim]
    colors = np.arange(0, (info.shape[-1] - ydim) // (ydim + 1), 1)
    initial_concentration = np.zeros(colors.size * 1)
    ingredients = np.zeros(colors.size * ydim).reshape(colors.size, ydim)
    for i in colors:
        initial_concentration[i] = info[ydim + i * (ydim + 1)]
        ingredients[i] = info[ydim + i * (ydim + 1) + 1:ydim + (i + 1) * (ydim + 1)]

    init_conc_array = np.repeat(initial_concentration.reshape(colors.size, 1), ydim).reshape(colors.size, ydim)

    # concentrations of painting
    # 这里缺乏灵活性，之后再改
    cons = np.zeros((n_grid ** colors.size, colors.size))
    yidx = np.arange(0, ydim, 1)
    for i, c in enumerate(mcx):
        for j, d in enumerate(mcx):
            for k, e in enumerate(mcx):
                for l, f in enumerate(mcx):
                    for m, g in enumerate(mcx):
                        for n, h in enumerate(mcx):
                            cons[i * (n_grid ** 5) + j * (n_grid ** 4) + k * (n_grid ** 3) +
                                 l * (n_grid ** 2) + m * n_grid + n] = [c, d, e, f, g, h]

    diff = np.zeros(n_grid ** colors.size)
    if model == 'km':
        fsb = (np.ones_like(background) - background) ** 2 / (background * 2)
        fst = ((np.ones_like(ingredients) - ingredients) ** 2 / (ingredients * 2) - fsb) / init_conc_array
        fss = np.zeros((n_grid ** colors.size, yidx.size))
        for i in yidx:
            for j in colors:
                fss[:, i] += cons[:, j] * fst[j, i]
            fss[:, i] += np.ones(n_grid ** colors.size) * fsb[i]
        diff = np.array([color_diff(ydata, p - ((p + 1) ** 2 - 1) ** 0.5 + 1) for p in fss]).transpose()

    elif model == 'four_flux':
        print('Sorry the model have not implemented yet')
        exit(1)

    else:
        print('Sorry no model of that name')
        exit(1)

    # normalise the posterior
    diff /= (np.sum(diff.flatten()) * (dmcx ** colors.size))

    # compute integrated probability outwards from max point
    diff = diff.flatten()
    idx = np.argsort(diff)[::-1]
    prob = np.zeros(n_grid ** colors.size)
    prob[idx] = np.cumsum(diff[idx]) * (dmcx ** colors.size)
    return mcx, cons, prob


def recipe_reflectance(recipes, model='km'):
    xidx = np.arange(0, 31, 1)
    init_conc_array = np.repeat(cInit.reshape(21, 1), 31).reshape(21, 31)
    reflectance = np.zeros(31 * recipes.shape[0]).reshape(31, recipes.shape[0])

    if model == 'km':
        fsb = (np.ones_like(background) - background) ** 2 / (background * 2)
        fst = ((np.ones_like(ingredients) - ingredients) ** 2 / (ingredients * 2) - fsb) / init_conc_array
        fss = np.zeros(31 * recipes.shape[0]).reshape(31, recipes.shape[0])
        for i in xidx:
            for j in range(6):
                fss[i, :] += recipes[:, j] * fst[j, i]
            fss[i, :] += np.ones(recipes.shape[0]) * fsb[i]

        reflectance = fss - ((fss + 1) ** 2 - 1) ** 0.5 + 1
        reflectance = reflectance.transpose()

    elif model == 'four_flux':
        ones_background = np.ones_like(background)
        fsb = (8 * background + (ones_background - 6 * background) *
               ((4 * background ** 2 - 4 * background + 25 * ones_background) ** 0.5)
               + 12 * background ** 2 + 5 * ones_background) / (48 * background)

        ones_ingredients = np.ones_like(ingredients).reshape(ingredients.shape[0], ingredients.shape[1])
        fst = ((8 * ingredients + (ones_ingredients - 6 * ingredients) *
                ((4 * ingredients ** 2 - 4 * ingredients + 25 * ones_ingredients) ** 0.5)
                + 12 * ingredients ** 2 + 5 * ones_ingredients) / (48 * ingredients) - fsb) / init_conc_array

        fss = np.zeros(31 * recipes.shape[0]).reshape(31, recipes.shape[0])
        for i in xidx:
            for j in range(6):
                fss[i, :] += recipes[:, j] * fst[j, i]
            fss[i, :] += np.ones(recipes.shape[0]) * fsb[i]

        ones_fss = np.ones_like(fss).reshape(fss.shape[0], fss.shape[1])
        reflectance = 0.5 * (1 / ((4 * (fss ** 2) + 4 * fss) ** 0.5 + 2 * fss + ones_fss)) + 0.5 * (
                (((fss + ones_fss) * ((4 * (fss ** 2) + 4 * fss) ** 0.5)) + 2 * (fss ** 2) - 2 * ones_fss) / (
                2 * (fss + ones_fss) * (3 * fss - ones_fss) * (
                ((4 * (fss ** 2) + 4 * fss) ** 0.5) + 2 * fss + ones_fss)))
        reflectance = reflectance.transpose()

    else:
        print('Sorry no model of that name')
        exit(1)

    return reflectance


def color_diff(reflectance1, reflectance2):
    tri1 = np.dot(optical_relevant, reflectance1.reshape(31, 1))
    tri2 = np.dot(optical_relevant, reflectance2.reshape(31, 1))

    lab1 = xyz2lab(tri1)
    lab2 = xyz2lab(tri2)
    delta_lab = lab1 - lab2

    diff = (delta_lab[0] ** 2 + delta_lab[1] ** 2 + delta_lab[2] ** 2) ** (1 / 2)
    return diff


def xyz2lab(xyz):
    r = 0.008856
    lab = np.zeros(3 * 1)

    if xyz[0] / perfect_white[0] > r and xyz[1] / perfect_white[1] > r and xyz[2] / perfect_white[2] > r:
        lab[0] = (xyz[1] / perfect_white[1]) ** (1 / 3) * 116 - 16
        lab[1] = ((xyz[0] / perfect_white[0]) ** (1 / 3) - (xyz[1] / perfect_white[1]) ** (1 / 3)) * 500
        lab[2] = ((xyz[1] / perfect_white[1]) ** (1 / 3) - (xyz[2] / perfect_white[2]) ** (1 / 3)) * 200
    else:
        lab[0] = (xyz[1] / perfect_white[1]) * 903.3
        lab[1] = (xyz[0] / perfect_white[0] - xyz[1] / perfect_white[1]) * 3893.5
        lab[2] = (xyz[1] / perfect_white[1] - xyz[2] / perfect_white[2]) * 1557.4

    return lab


def test_result():
    test_sample = np.array([[0.2335778, 0.2669207, 0.2692738, 0.2711587, 0.2714446, 0.2716927, 0.2717375, 0.2719162,
                             0.2708953, 0.2690464, 0.2680669, 0.2653399, 0.2614551, 0.2566476, 0.2497642, 0.2403944,
                             0.2294249, 0.2198784, 0.2130321, 0.2085395, 0.2036385, 0.1986061, 0.1988262, 0.2010616,
                             0.2045918, 0.2065673, 0.2080161, 0.2085742, 0.2017922, 0.2004079, 0.1997598]])
    recipe = np.array([[0.36730343, 0.25544596, 0.02905927, 0.21943599, 0, -0.00190955]])
    recipe_ref = recipe_reflectance(recipe)
    diff = color_diff(test_sample, recipe_ref)
    print(recipe_ref)
    print(diff)


def four_flux():
    x = 2.2881
    y = 0.5 / (2 * x + 2 * ((x ** 2 + x) ** (1 / 2)) + 1) + (
            0.5 * (1 + x) * (2 * x - 2 * ((x ** 2 + x) ** (1 / 2)) - 2)) / (
                (6 * (x ** 2) + 4 * x - 2) * (2 * x + 2 * ((x ** 2 + x) ** (1 / 2)) + 1))
    x_rev_1 = ((5 - 2 * y) * ((36 * (y ** 2) - 28 * y + 1) ** (1 / 2)) - 8 * y + 12 * (y ** 2) + 5) / (80 * y)
    x_rev_2 = ((2 * y - 5) * ((36 * (y ** 2) - 28 * y + 1) ** (1 / 2)) - 8 * y + 12 * (y ** 2) + 5) / (80 * y)
    print(y)
    print(x_rev_1)
    print(x_rev_2)
