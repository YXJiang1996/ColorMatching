import torch
import numpy as np
import math_model.math.Mix as mix
from itertools import combinations

# from math_model.math.Mix import get_dfs_KM

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
    [0.2882330, 0.3192795, 0.2996702, 0.2849077, 0.2800260, 0.2811862, 0.2897825, 0.3125680, 0.3337503, 0.3423721,
     0.3767265, 0.4592168, 0.5899043, 0.7359028, 0.8444552, 0.9036496, 0.9298696, 0.9393797, 0.9419743, 0.9430188,
     0.9438195, 0.9437287, 0.9436158, 0.9435386, 0.9432820, 0.9437690, 0.9434277, 0.9418160, 0.9393354, 0.9358864,
     0.9297287],
    [0.2708842, 0.2955388, 0.2835928, 0.2703437, 0.2596675, 0.2515463, 0.2464779, 0.2449044, 0.2445325, 0.2466007,
     0.2538134, 0.2617629, 0.2677513, 0.2809876, 0.3157711, 0.3914984, 0.5052982, 0.6254060, 0.7242598, 0.8023202,
     0.8626911, 0.8989766, 0.9154678, 0.9242462, 0.9278284, 0.9308498, 0.9321416, 0.9321451, 0.9319565, 0.9283142,
     0.9233366],
    [0.3737892, 0.5056624, 0.5303422, 0.5383733, 0.5443333, 0.5562689, 0.5746784, 0.5999998, 0.6406719, 0.6941598,
     0.7595113, 0.8284632, 0.8829190, 0.9176573, 0.9354603, 0.9417856, 0.9447337, 0.9462350, 0.9468061, 0.9464547,
     0.9465482, 0.9460330, 0.9458108, 0.9456140, 0.9452482, 0.9455251, 0.9448371, 0.9429901, 0.9404516, 0.9368277,
     0.9308580],
    [0.3597840, 0.4634646, 0.4685041, 0.4637362, 0.4587243, 0.4569478, 0.4584772, 0.4639183, 0.4731687, 0.4823705,
     0.4972585, 0.5271874, 0.5842780, 0.6700529, 0.7608584, 0.8352169, 0.8860149, 0.9109665, 0.9240378, 0.9305357,
     0.9349512, 0.9380493, 0.9396749, 0.9407066, 0.9412952, 0.9422389, 0.9419153, 0.9406110, 0.9387050, 0.9357519,
     0.9288481],
    [0.3993709, 0.5652254, 0.5701579, 0.5461070, 0.5201841, 0.4957078, 0.4756258, 0.4650650, 0.4493746, 0.4310669,
     0.4210130, 0.4203477, 0.4135541, 0.3959555, 0.3948928, 0.4532020, 0.6005961, 0.7637561, 0.8650616, 0.9100221,
     0.9297992, 0.9363567, 0.9375512, 0.9379768, 0.9381308, 0.9389320, 0.9387877, 0.9366149, 0.9343541, 0.9308279,
     0.9244555],
    [0.3582383, 0.4589019, 0.4442960, 0.4180548, 0.3943129, 0.3753622, 0.3580040, 0.3467267, 0.3377595, 0.3309189,
     0.3229960, 0.3138701, 0.3117055, 0.3117619, 0.3016199, 0.2815702, 0.2676951, 0.2725082, 0.3257494, 0.5337213,
     0.7479121, 0.8533681, 0.9027772, 0.9256366, 0.9336862, 0.9368781, 0.9375737, 0.9365737, 0.9347985, 0.9322762,
     0.9260198],
    [0.4160994, 0.6342332, 0.6881241, 0.6826540, 0.6586606, 0.6262225, 0.5918874, 0.5623005, 0.5190387, 0.4766790,
     0.4453364, 0.4108316, 0.3585283, 0.3157409, 0.3090685, 0.3137436, 0.3092155, 0.3295474, 0.4146603, 0.5391598,
     0.6621714, 0.7592348, 0.8222232, 0.8629542, 0.8859942, 0.9002361, 0.9062362, 0.9085600, 0.9108428, 0.9079770,
     0.9066052],
    [0.3734002, 0.5346663, 0.5803110, 0.5968833, 0.5961596, 0.5762038, 0.5414493, 0.5004809, 0.4368144, 0.3691231,
     0.3134314, 0.2668127, 0.2305206, 0.2116322, 0.2072988, 0.2046728, 0.2012968, 0.2056570, 0.2243125, 0.2557296,
     0.2817601, 0.2780255, 0.2626838, 0.2666584, 0.3057532, 0.3696144, 0.4543115, 0.5407568, 0.6394141, 0.6732088,
     0.7227149],
    [0.3747776, 0.5648752, 0.6199736, 0.6661413, 0.7298144, 0.7726266, 0.7812571, 0.7792725, 0.7607376, 0.7320930,
     0.6943575, 0.6432314, 0.5800607, 0.5089791, 0.4396545, 0.3704733, 0.3041328, 0.2567624, 0.2272359, 0.2094773,
     0.1948587, 0.1825973, 0.1767187, 0.1732840, 0.1733995, 0.1762372, 0.1840847, 0.1860129, 0.1805977, 0.1722357,
     0.1589434],
    [0.3075271, 0.3970215, 0.4258692, 0.4498420, 0.4777384, 0.5187902, 0.5789000, 0.6485421, 0.7203640, 0.7698454,
     0.7815039, 0.7697773, 0.7418166, 0.7037384, 0.6568713, 0.6019219, 0.5399272, 0.4776893, 0.4137468, 0.3495456,
     0.2899768, 0.2505462, 0.2324735, 0.2232275, 0.2189511, 0.2174184, 0.2212247, 0.2353665, 0.2580868, 0.2678188,
     0.2783012],
    [0.3720859, 0.4998843, 0.5114251, 0.5128380, 0.5068206, 0.4906232, 0.4670616, 0.4448574, 0.4133620, 0.3843807,
     0.3632236, 0.3376301, 0.3054249, 0.2861553, 0.2870051, 0.2837292, 0.2620971, 0.2423432, 0.2991848, 0.4617812,
     0.6395882, 0.7728266, 0.8491868, 0.8948321, 0.9179356, 0.9291759, 0.9323577, 0.9334325, 0.9336824, 0.9315314,
     0.9258437],
    [0.3777322, 0.5588190, 0.6166437, 0.6508363, 0.6942729, 0.7478291, 0.7715947, 0.7660858, 0.7381392, 0.6950833,
     0.6416819, 0.5781880, 0.5089610, 0.4370953, 0.3696142, 0.3065603, 0.2513052, 0.2179904, 0.2023969, 0.1957846,
     0.1897067, 0.1847080, 0.1875919, 0.2009600, 0.2232523, 0.2416958, 0.2510498, 0.2490231, 0.2179781, 0.2109241,
     0.2085472],
    [0.3087468, 0.3818603, 0.3858492, 0.3842682, 0.3812729, 0.3790042, 0.3778628, 0.3774856, 0.3758622, 0.3744030,
     0.3748578, 0.3749601, 0.3755584, 0.3774002, 0.3831909, 0.3992164, 0.4275523, 0.4752591, 0.5337597, 0.5910335,
     0.6392522, 0.6741050, 0.6950621, 0.7085229, 0.7201321, 0.7306639, 0.7401660, 0.7497781, 0.7626545, 0.7668678,
     0.7736240],
    [0.3353273, 0.4391780, 0.4634378, 0.4844812, 0.5122489, 0.5352546, 0.5445021, 0.5498481, 0.5548403, 0.5667891,
     0.5892876, 0.6205880, 0.6589307, 0.7075461, 0.7588425, 0.8063425, 0.8438276, 0.8656531, 0.8760347, 0.8760190,
     0.8742780, 0.8694988, 0.8650322, 0.8608174, 0.8578521, 0.8574630, 0.8571312, 0.8569201, 0.8584669, 0.8579510,
     0.8594775],
    [0.2022878, 0.2236471, 0.2248906, 0.2230992, 0.2210833, 0.2189706, 0.2162815, 0.2146686, 0.2124687, 0.2098620,
     0.2086629, 0.2066279, 0.2050257, 0.2029288, 0.2011808, 0.1990332, 0.1977500, 0.1961589, 0.1945706, 0.1927553,
     0.1913664, 0.1891944, 0.1875919, 0.1858389, 0.1840398, 0.1824298, 0.1811835, 0.1801012, 0.1758438, 0.1747804,
     0.1729087]])
# 对应的初始浓度
cInit = np.array(
    [0.51, 0.51, 0.51, 0.61, 0.51, 0.5, 0.53, 0.6, 0.54, 0.53, 0.516, 0.503, 0.508, 0.498, 0.508, 0.497, 0.504, 0.51,
     0.529, 0.504, 0.511])
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
    r = list(combinations(np.arange(0, colors.size, 1), 18))
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
    # print(concentrations.dtype)
    return concentrations, reflectance, xvec, info


# generate(1024)

def math_optimized_generate(info=info):
    data = np.load('math/dataset_Corrected_01.npz')
    concentrations = torch.from_numpy(data['concentrations']).float()
    reflectance = torch.from_numpy(data['reflectance']).float()
    xvec = np.arange(400, 710, 10)
    # print(concentrations.dtype)
    return concentrations, reflectance, xvec, info


# math_optimized_generate(1024)

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


# 使用修正模型计算配方分光反射率的方式
def correct_recipe_reflectance(recipes):
    w = np.load('math/data_w.npy').T
    dfs = mix.get_dfs_KM(w)
    return mix.corrected_Mix(recipes, w, dfs)


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


test_samps = np.array([[0.2673378, 0.3132285, 0.3183329, 0.3234908, 0.3318701, 0.3409707, 0.3604081, 0.4168356,
                        0.5351773, 0.6202191, 0.6618687, 0.6919741, 0.7136238, 0.7292901, 0.7314631, 0.7131701,
                        0.6773048, 0.6302681, 0.5738088, 0.5133060, 0.4535525, 0.4108878, 0.3908512, 0.3808001,
                        0.3752591, 0.3727644, 0.3801365, 0.3976869, 0.4237110, 0.4332685, 0.4433292]])

test_cons = np.array(
    [[0, 0.8014, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 0.1491, 0, 0, 0, 0.2241, 0]])

test_reflectance=recipe_reflectance(test_cons, model='km')
test_diff=color_diff(test_samps,test_reflectance)
print(test_diff)
