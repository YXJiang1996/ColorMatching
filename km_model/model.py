import numpy as np
import data


def main():
    # set up simulation parameters
    r = 3  # the grid dimension for the output tests
    test_split = r * r  # number of testing samples to use
    optical_model = 'km'  # the optical model to use
    ydim = 31  # number of data samples
    bound = [0.1, 0.9, 0.1, 0.9]
    seed = 1  # seed for generating data

    # 生成训练数据
    concentrations, reflectance, x, info = data.generate(
        model=optical_model,
        total_dataset_size=2 ** 20 * 20,
        ydim=ydim,
        prior_bound=bound,
        seed=seed
    )

    colors=np.arange(0,concentrations.shape[-1],1)

    c_test=concentrations[-test_split]
    r_test=reflectance[-test_split]
main()
