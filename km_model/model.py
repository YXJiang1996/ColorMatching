import torch
import torch.optim
from torch.utils import data

import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import rev_multiplicative_layer, F_fully_connected, permute_layer

import pandas as pd
from sklearn.manifold import TSNE

import data

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def MMD_multiscale(x, y):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * zz

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    for a in [0.2, 0.5, 0.9, 1.3]:
        XX += a ** 2 * (a ** 2 + dxx) ** -1
        YY += a ** 2 * (a ** 2 + dyy) ** -1
        XY += a ** 2 * (a ** 2 + dxy) ** -1

    return torch.mean(XX + YY - 2. * XY)


def fit(input, target):
    return torch.mean((input - target) ** 2)


def non_nagative_attachment(base, lamb, x):
    return 1. / torch.clamp(torch.pow(base, lamb * x[x < 0]), min=0.001)


def running_mean(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def plot_losses(losses, logscale=False, legend=None):
    # 正向传播损失
    fig = plt.figure(figsize=(6, 6))
    losses = np.array(losses)
    ax1 = fig.add_subplot(211)
    ax1.plot(losses[0], 'b')
    ax1.set_xlabel(r'epoch')
    ax1.set_ylabel(r'loss')
    if legend is not None:
        ax1.legend('forward pass', loc='upper left')
    ax1.plot(running_mean(losses[0], 50), 'g')

    # 逆向传播损失
    ax2 = fig.add_subplot(212)
    ax2.plot(losses[1], 'r')

    # rescale axis using a logistic function so that we see more detail
    # close to 0 and close 1
    ax2.set_xlabel(r'epoch')
    ax2.set_ylabel(r'loss')
    if legend is not None:
        ax2.legend('backward pass', loc='upper left')

    ax2.plot(running_mean(losses[1], 50), 'y')

    if logscale:
        ax1.set_xscale("log", nonposx='clip')
        ax1.set_yscale("log", nonposy='clip')
        ax2.set_xscale("log", nonposx='clip')
        ax2.set_yscale("log", nonposy='clip')
    plt.savefig('losses.png')
    plt.close()


# 网络训练过程
def train(model, train_loader, n_its_per_epoch, zeros_noise_scale, batch_size, ndim_tot, ndim_x, ndim_y, ndim_z,
          y_noise_scale, optimizer, lambd_predict, loss_fit, lambd_latent, loss_latent, lambd_rev, loss_backward,
          i_epoch=0):
    model.train()

    l_tot = 0
    batch_idx = 0

    # 训练轮数相关的权重 4-1
    loss_factor = 600 ** (float(i_epoch) / 300) / 600
    if loss_factor > 1:
        loss_factor = 1

    # zeros_noise_scale *= (1 - loss_factor)

    for x, y in train_loader:
        batch_idx += 1
        if batch_idx > n_its_per_epoch:
            break

        x, y = x.to(device), y.to(device)

        y_clean = y.clone()

        # 对x进行向量补齐
        pad_x = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                ndim_x, device=device)
        # 对yz进行向量补齐
        pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                 ndim_y - ndim_z, device=device)

        y += y_noise_scale * torch.randn(batch_size, ndim_y, dtype=torch.float, device=device)
        # add_info += y_noise_scale * torch.randn(batch_size, info_dim, dtype=torch.float, device=device)

        x, y = (torch.cat((x, pad_x), dim=1),
                torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz, y),
                          dim=1))

        optimizer.zero_grad()

        # 前向训练：
        output = model(x)

        # Shorten output, and remove gradients wrt y, for latent loss
        y_short = torch.cat((y[:, :ndim_z], y[:, -ndim_y:]), dim=1)

        l = lambd_predict * loss_fit(output[:, ndim_z:], y[:, ndim_z:])

        output_block_grad = torch.cat((output[:, :ndim_z],
                                       output[:, -ndim_y:].data), dim=1)

        l += lambd_latent * loss_latent(output_block_grad, y_short)
        l_tot += l.data.item()

        l.backward()

        # Backward step:
        pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                 ndim_y - ndim_z, device=device)
        y = y_clean + y_noise_scale * torch.randn(batch_size, ndim_y, device=device)

        orig_z_perturbed = (output.data[:, :ndim_z] + y_noise_scale *
                            torch.randn(batch_size, ndim_z, device=device))
        y_rev = torch.cat((orig_z_perturbed, pad_yz, y), dim=1)
        y_rev_rand = torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz, y), dim=1)

        output_rev = model(y_rev, rev=True)
        output_rev_rand = model(y_rev_rand, rev=True)

        l_rev = (
                lambd_rev
                * loss_factor
                * (loss_backward(output_rev_rand[:, :ndim_x], x[:, :ndim_x])
                   # + loss_fit(output_rev_rand[:, -info_dim:], x[:, -info_dim:])
                   )
        )

        l_rev += (0.50 * lambd_predict * loss_fit(output_rev, x)
                  + loss_factor * non_nagative_attachment(10, 2, output_rev[:, :ndim_x]).sum())

        l_tot += l_rev.data.item()
        l_rev.backward()

        for p in model.parameters():
            p.grad.data.clamp_(-5.00, 5.00)

        optimizer.step()

    return l_tot / batch_idx, l / batch_idx, l_rev / batch_idx


def main():
    # ---------------------------------------生成数据------------------------------------------
    t_generate_start = time()
    # 设置模拟数据参数
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
    print("\n\nGenerating data took %.2f minutes\n" % ((time() - t_generate_start) / 60))
    colors = np.arange(0, concentrations.shape[-1], 1)

    # 选取几个不参与训练，用作最后的测试样本
    c_test = concentrations[-test_split:]
    r_test = reflectance[-test_split:]

    # 测试样本分光反射率图，用于观察，与模型无关
    plt.figure(figsize=(6, 6))
    fig, axes = plt.subplots(r, r, figsize=(6, 6))
    cnt = 0
    for i in range(r):
        for j in range(r):
            axes[i, j].plot(x, np.array(r_test[cnt, :]), '-')
            cnt += 1
            axes[i, j].axis([400, 700, 0, 1])
    plt.savefig('test_target_reflectance.png', dpi=360)
    plt.close()
    print("\n\nGenerating data took %.2f minutes\n" % ((time() - t_generate_start) / 60))

    # ---------------------------------------构建网络------------------------------------------
    # 设置模型参数值
    ndim_x = concentrations.shape[-1]  # 配方的维度，即待选色浆的种类数
    ndim_y = ydim  # 反射率的维度 31
    ndim_z = 13  # 潜在空间的维度
    ndim_tot = max(ndim_x, ndim_y + ndim_z)

    # 定义神经网络的不同部分
    # 定义输入层节点
    inp = InputNode(ndim_tot, name='input')

    # 定义隐藏层节点
    t1 = Node([inp.out0], rev_multiplicative_layer,
              {'F_class': F_fully_connected, 'clamp': 2.0,
               'F_args': {'dropout': 0.2}})

    p1 = Node([t1.out0], permute_layer, {'seed': 1})

    t2 = Node([p1.out0], rev_multiplicative_layer,
              {'F_class': F_fully_connected, 'clamp': 2.0,
               'F_args': {'dropout': 0.2}})

    p2 = Node([t2.out0], permute_layer, {'seed': 2})

    t3 = Node([p2.out0], rev_multiplicative_layer,
              {'F_class': F_fully_connected, 'clamp': 2.0,
               'F_args': {'dropout': 0.2}})

    p3 = Node([t3.out0], permute_layer, {'seed': 1})

    t4 = Node([p3.out0], rev_multiplicative_layer,
              {'F_class': F_fully_connected, 'clamp': 2.0,
               'F_args': {'dropout': 0.2}})

    p4 = Node([t4.out0], permute_layer, {'seed': 2})

    t5 = Node([p4.out0], rev_multiplicative_layer,
              {'F_class': F_fully_connected, 'clamp': 2.0,
               'F_args': {'dropout': 0.2}})

    # 定义输出层节点
    outp = OutputNode([t5.out0], name='output')

    # 构建网络
    nodes = [inp, t1, p1, t2, p2, t3, p3, t4, p4, t5, outp]
    model = ReversibleGraphNet(nodes)

    # ---------------------------------------训练网络------------------------------------------
    # 超参数
    n_epochs = 3000  # 训练轮数
    plot_cadence = 50  # 每50步画一次损失函数图
    meta_epoch = 12  # 调整学习率的步长
    n_its_per_epoch = 12  # 每次训练12批数据
    batch_size = 1600  # 每批1600个样本
    lr = 1.5e-3  # 初始学习率
    gamma = 0.004 ** (1. / 1333)  # 学习率下降的乘数因子
    l2_reg = 2e-5  # 权重衰减（L2惩罚）
    # 为了让输入和输出维度相同，对维度进行补齐，不使用0，而是使用一些很小的值
    y_noise_scale = 3e-2
    zeros_noise_scale = 3e-2

    # 损失的权重
    lambd_predict = 300.  # forward pass
    lambd_latent = 300.  # laten space
    lambd_rev = 400.  # backwards pass

    # 定义优化器
    # params：待优化参数，lr：学习率，betas:用于计算梯度以及梯度平方的运行平均值的系数
    # eps:为了增加数值计算的稳定性而加到分母里的项
    # weight_decay:权重衰减
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.8, 0.8),
                                 eps=1e-04, weight_decay=l2_reg)
    # 学习率调整
    # optimizer:优化器
    # step_size:调整学习率的步长
    # gamma:学习率下降的乘数因子
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=meta_epoch,
                                                gamma=gamma)
    # 损失函数设置
    # x，z无监督：MMD，y有监督：平方误差
    loss_backward = MMD_multiscale
    loss_latent = MMD_multiscale
    loss_fit = fit

    # 训练集数据加载
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(concentrations[test_split:], reflectance[test_split:]),
        batch_size=batch_size, shuffle=True, drop_last=True)

    # 初始化网络权重
    for mod_list in model.children():
        for block in mod_list.children():
            for coeff in block.children():
                coeff.fc3.weight.data = 0.01 * torch.randn(coeff.fc3.weight.shape)
    model.to(device)

    # 初始化测试结果图表
    fig, axes = plt.subplots(r, r, figsize=(6, 6))

    # 测试用例数量
    N_samp = 256

    # ---------------------------------------开始训练------------------------------------------
    try:
        t_start = time()  # 训练开始时间
        loss_for_list = []  # 记录前向训练的损失
        loss_rev_list = []  # 记录反向训练的损失

        tsne = TSNE(n_components=2, init='pca')
        # 颜色编号
        color_names = ['07H', '08', '08S', '09', '09B', '09S', '10B', '12', '13',
                       '14', '15', '16', '17A', '18A', '19A', '20A-2',
                       '23A', '2704', '2803', '2804', '2807']

        # loop over number of epochs
        for i_epoch in tqdm(range(n_epochs), ascii=True, ncols=80):

            scheduler.step()

            # Initially, the l2 reg. on x and z can give huge gradients, set
            # the lr lower for this
            if i_epoch < 0:
                print('inside this iepoch<0 thing')
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * 1e-2

            # train the model
            avg_loss, loss_for, loss_rev = train(model, train_loader, n_its_per_epoch, zeros_noise_scale,
                                                 batch_size,
                                                 ndim_tot, ndim_x, ndim_y, ndim_z, y_noise_scale, optimizer,
                                                 lambd_predict,
                                                 loss_fit, lambd_latent, loss_latent, lambd_rev, loss_backward, i_epoch)

            loss_for_list.append(loss_for.item())
            loss_rev_list.append(loss_rev.item())
            inn_losses = [loss_for_list, loss_rev_list]

            if (i_epoch % plot_cadence == 0) & (i_epoch > 0):
                plot_losses(inn_losses, legend=['PE-GEN'])

        # TODO
        # model = torch.load('model_dir/km_impl_model')
        torch.save(model, 'model_dir/km_impl_model')

        fig, axes = plt.subplots(1, 1, figsize=(2, 2))

        # 真实样本对应的反射率信息
        test_samps = np.array([[0.2673378, 0.3132285, 0.3183329, 0.3234908, 0.3318701, 0.3409707, 0.3604081, 0.4168356,
                                0.5351773, 0.6202191, 0.6618687, 0.6919741, 0.7136238, 0.7292901, 0.7314631, 0.7131701,
                                0.6773048, 0.6302681, 0.5738088, 0.5133060, 0.4535525, 0.4108878, 0.3908512, 0.3808001,
                                0.3752591, 0.3727644, 0.3801365, 0.3976869, 0.4237110, 0.4332685, 0.4433292]])
        # 真实样本对应的配方
        test_cons = np.array(
            [[0, 0.8014, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0,
              0, 0.1491, 0, 0, 0, 0.2241, 0]])
        for cnt in range(test_samps.shape[0]):
            test_samp = np.tile(np.array(test_samps[cnt, :]), N_samp).reshape(N_samp, ydim)
            test_samp = torch.tensor(test_samp, dtype=torch.float)
            test_samp += y_noise_scale * torch.randn(N_samp, ydim)

            test_samp = torch.cat([torch.randn(N_samp, ndim_z),  # zeros_noise_scale *
                                   torch.zeros(N_samp, ndim_tot - ndim_y - ndim_z),
                                   test_samp], dim=1)
            test_samp = test_samp.to(device)

            # use the network to predict parameters
            test_rev = model(test_samp, rev=True)[:, :colors.size]
            test_rev = test_rev.cpu().data.numpy()
            # 假设涂料浓度小于一定值，就不需要这种涂料
            test_rev = np.where(test_rev < 0.1, 0, test_rev)

            # 计算预测配方的反射率信息
            recipe_ref = data.recipe_reflectance(test_rev, optical_model)
            print("######## Test Sample %d ########" % cnt)
            # 用于记录色差最小的三个配方
            top3 = [[100, 0], [100, 0], [100, 0]]
            for n in range(test_rev.shape[0]):
                # print(test_rev[n, :])
                diff = data.color_diff(test_samps[cnt, :], recipe_ref[n, :])
                if diff < top3[2][0]:
                    top3[2][0] = diff
                    top3[2][1] = n
                    top3.sort()
            # 将色差最小的三个配方打印出来
            for n in range(3):
                print(test_rev[top3[n][1], :])
                print("color diff: %.2f \n" % top3[n][0])
            print("\n\n")

            # draw
            # feature scaling
            test_x = test_cons[cnt, :].reshape(1, test_cons[cnt, :].shape[-1])
            plot_x = np.concatenate((test_rev, test_x), axis=0)

            # use tsne to decrease dimensionality
            x_norm = pd.DataFrame(plot_x, columns=color_names)

            # 根据需要的涂料种类（需要为1，不需要为0）将配方分类
            classes = np.zeros(N_samp).reshape(N_samp, 1)
            paint_needed = np.where(test_rev == 0, 0, 1)
            for paint_no in colors:
                classes[:, 0] += paint_needed[:, paint_no] * 2 ** paint_no
            class_norm = pd.DataFrame(np.concatenate((classes, np.zeros(1).reshape(1, 1)), axis=0),
                                      columns=['class'])

            data_plot = pd.concat([pd.DataFrame(tsne.fit_transform(x_norm)), class_norm], axis=1)
            class_data = data_plot['class']

            axes.clear()
            recipe_classes = np.array(class_norm[:-1].drop_duplicates()).reshape(1, -1).tolist()[0]
            for recipe_class in recipe_classes:
                axes.scatter(data_plot[class_data == recipe_class][0], data_plot[class_data == recipe_class][1],
                             s=2, alpha=0.5)
            axes.scatter(data_plot[class_data == 0][0], data_plot[class_data == 0][1], marker='+', s=10)
            fig.canvas.draw()
            plt.savefig('test_result%d.png' % cnt, dpi=360)

        # loop over a few cases and plot results in a grid
        cnt = 0
        for i in range(r):
            for j in range(r):
                # convert data into correct format
                y_samps = np.tile(np.array(r_test[cnt, :]), N_samp).reshape(N_samp, ydim)
                y_samps = torch.tensor(y_samps, dtype=torch.float)
                y_samps += y_noise_scale * torch.randn(N_samp, ydim)

                y_samps = torch.cat([torch.randn(N_samp, ndim_z),  # zeros_noise_scale *
                                     torch.zeros(N_samp, ndim_tot - ndim_y - ndim_z),
                                     y_samps], dim=1)
                y_samps = y_samps.to(device)

                # use the network to predict parameters
                rev_x = model(y_samps, rev=True)[:, :colors.size]
                rev_x = rev_x.cpu().data.numpy()

                # 假设涂料浓度小于一定值，就不需要这种涂料
                rev_x = np.where(rev_x < 0.1, 0, rev_x)

                # feature scaling
                test_x = c_test[cnt, :].reshape(1, c_test[cnt, :].shape[-1])
                plot_x = np.concatenate((rev_x, test_x), axis=0)

                # use pca to decrease dimensionality
                x_norm = pd.DataFrame(plot_x, columns=color_names)

                # 根据需要的涂料种类（需要为1，不需要为0）将配方分类
                classes = np.zeros(N_samp).reshape(N_samp, 1)
                paint_needed = np.where(rev_x == 0, 0, 1)
                for paint_no in colors:
                    classes[:, 0] += paint_needed[:, paint_no] * 2 ** paint_no
                class_norm = pd.DataFrame(np.concatenate((classes, np.zeros(1).reshape(1, 1)), axis=0),
                                          columns=['class'])

                data_plot = pd.concat([pd.DataFrame(tsne.fit_transform(x_norm)), class_norm], axis=1)

                class_data = data_plot['class']

                # plot the predicted and the true recipe
                axes.clear()
                recipe_classes = np.array(class_norm[:-1].drop_duplicates()).reshape(1, -1).tolist()[0]
                for recipe_class in recipe_classes:
                    axes.scatter(data_plot[class_data == recipe_class][0],
                                 data_plot[class_data == recipe_class][1],
                                 s=2, alpha=0.5)
                axes.scatter(data_plot[class_data == 0][0], data_plot[class_data == 0][1], marker='+',
                             s=10)

                fig.canvas.draw()
                plt.savefig('training_result%d.png' % cnt, dpi=360)

                recipe_ref = data.recipe_reflectance(rev_x, optical_model)
                print("######## Test %d ########" % cnt)
                print(c_test[cnt])
                print("################")
                # 用于记录色差最小的三个配方
                top3 = [[100, 0], [100, 0], [100, 0]]
                for n in range(rev_x.shape[0]):
                    # print(rev_x[n, :])
                    diff = data.color_diff(r_test[cnt].numpy(), recipe_ref[n, :])
                    if diff < top3[2][0]:
                        top3[2][0] = diff
                        top3[2][1] = n
                        top3.sort()
                # 将色差最小的三个配方打印出来
                for n in range(3):
                    print(test_rev[top3[n][1], :])
                    print("color diff: %.2f \n" % top3[n][0])
                print("\n\n")

                cnt += 1

    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nTraining took %.2f minutes\n" % ((time() - t_start) / 60))


main()
