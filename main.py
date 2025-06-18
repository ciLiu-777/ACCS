from config import opt
import models
from data.dataset import IndoorGroup
from torch.utils.tensorboard import SummaryWriter
import torch as t
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm  # 显示进度条
import numpy as np
from matplotlib import pyplot as plt
from utils.visualize import Visualizer, write_csv, plot_cdf, ConfusionMatrix,plot4_cdf
import time


@t.no_grad()
def val_CAE(model, dataloader, criterion):
    """
    计算模型在验证集上的准确率等信息
    """
    model.eval()
    val_loss = []  # 用于存储每个验证批次的损失值
    for ii, (val_input, _) in enumerate(dataloader):
        val_input = val_input.to(opt.device)
        score = model(val_input)

        loss = criterion(score, val_input)
        val_loss.append(loss.item())

    model.train()

    return val_loss


def train_CAE(**kwargs):
    opt._parse(kwargs)  # 更新参数

    # step1: configure model
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    # step2: data
    train_data = IndoorGroup(opt.train_data_root, train=True)
    val_data = IndoorGroup(opt.train_data_root, train=False)
    train_dataloader = DataLoader(
        train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers
    )
    val_dataloader = DataLoader(
        val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers
    )

    # step3: criterion and optimizer
    criterion = t.nn.MSELoss()
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)

    # step4: meters
    loss_meter = meter.AverageValueMeter()  # 用于计算和存储平均损失值的 平均计量器
    previous_loss = 1e10  # 用于存储上一次的损失值。

    # train
    train_loss_epochs = []
    val_loss_epochs = []
    for epoch in tqdm(range(opt.max_epoch)):

        ### start trianning
        trn_loss = []  # 用于存储每个训练批次的损失值

        loss_meter.reset()  # 重置损失度量器

        for ii, (data, _) in tqdm(enumerate(train_dataloader)):

            # train model
            input = data.to(opt.device)
            target = input  # CAE

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            # meters update
            loss_meter.add(loss.item())

            # visualize
            trn_loss.append(loss.item())
        train_loss_epochs.append(np.average(trn_loss))  # 计算每轮平均训练损失

        model.save()

        # validate
        val_loss = val_CAE(model, val_dataloader, criterion)

        val_loss_epochs.append(np.average(val_loss))

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        previous_loss = loss_meter.value()[0]

    ### 可视化
    epochs = np.arange(opt.max_epoch) + 1
    plt.plot(epochs, train_loss_epochs, "bo", label="Training loss")
    plt.plot(epochs, val_loss_epochs, "r-", label="Test loss")
    plt.title("Training and Test loss over increasing epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid("off")
    plt.show()




def train_ResNet(**kwargs):
    opt._parse(kwargs)  # 更新参数
    vis = Visualizer(opt.env, port=opt.vis_port)

    # step1: configure model
    pre_model = getattr(models, "CAE")()
    if opt.load_model_path:
        pre_model.load(opt.load_model_path)
    pre_model.to(opt.device)
    # 冻结编码器参数
    for param in pre_model.encoder.parameters():
        param.requires_grad = False  # 禁止梯度计算
    pre_model.encoder.eval()  # 设置编码器为评估模式（固定BN统计量）

    model = getattr(models, opt.model)(use_attention=opt.use_attention)
    model.to(opt.device)

    # step2: data
    train_data = IndoorGroup(opt.train_data_root, train=True)
    val_data = IndoorGroup(opt.train_data_root, train=False)
    train_dataloader = DataLoader(
        train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers
    )
    val_dataloader = DataLoader(
        val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers
    )

    # step3: criterion and optimizer
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)

    # step4: meters
    loss_meter = meter.AverageValueMeter()  # 用于计算和存储平均损失值的 平均计量器
    confusion_matrix = meter.ConfusionMeter(8)  ### 8分类问题！
    previous_loss = 1e10  # 用于存储上一次的损失值。

    # train
    for epoch in range(opt.max_epoch):

        loss_meter.reset()  # 重置损失度量器
        confusion_matrix.reset()  # 重置混淆矩阵

        for ii, (data, label) in tqdm(enumerate(train_dataloader), desc=f"Train-eqpoch:{epoch+1}/{opt.max_epoch}", unit="batch"):

            # train model
            input = data.to(opt.device)
            target = label.to(opt.device)

            with t.no_grad(): 
                input_zip = pre_model.encoder(input)

            optimizer.zero_grad()
            score = model(input_zip)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            # meters update and visualize
            loss_meter.add(loss.item())
            # detach 一下更安全保险
            confusion_matrix.add(score.detach(), target.detach())

            if (ii + 1) % opt.print_freq == 0:
                vis.plot("loss", loss_meter.value()[0])

        model.save()

        # validate and visualize
        val_cm,val_accuracy = val_ResNet(model,pre_model,val_dataloader)

        vis.plot('val_accuracy',val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
                    epoch = epoch,loss = loss_meter.value()[0],val_cm = str(val_cm.value()),train_cm=str(confusion_matrix.value()),lr=lr))
        
        # update learning rate
        if loss_meter.value()[0] > previous_loss:          
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]

@t.no_grad()
def val_ResNet(model,premodel,dataloader):
    """
    计算模型在验证集上的准确率等信息
    """

    model.eval()
    confusion_matrix = meter.ConfusionMeter(8)### 8分类问题！

    for ii, (val_input, label) in tqdm(enumerate(dataloader),  desc="Validation"):
        val_input = val_input.to(opt.device)
        with t.no_grad(): 
            # 使用预训练模型的编码器提取特征
            input_zip = premodel.encoder(val_input)
            score = model(input_zip)

        #更新混淆矩阵，记录预测结果与真实标签的对应关系
        confusion_matrix.add(score.detach().squeeze(), label.type(t.LongTensor))



    model.train()

    # 获取混淆矩阵的值，用于后续计算准确率。
    cm_value = confusion_matrix.value()
    # 计算准确率
    correct = cm_value.diagonal().sum()
    total = cm_value.sum()
    accuracy = 100. * correct / total

    return confusion_matrix, accuracy


@t.no_grad() # pytorch>=0.5
def test(cae_model_path, resnet_model_path, resnet_model_cp1_path, **kwargs):
    opt._parse(kwargs)

    # configure model
    premodel = getattr(models, "CAE")()
    premodel.load(cae_model_path)

    model = getattr(models, "ResNet18")(use_attention=True)
    model.load(resnet_model_path)
    cp1model = getattr(models, "ResNet18")(use_attention=False)
    cp1model.load(resnet_model_cp1_path)

    premodel.to(opt.device)
    model.to(opt.device)
    cp1model.to(opt.device)

    premodel.eval()
    model.eval()
    cp1model.eval()


    # data
    test_dataset = IndoorGroup(opt.test_data_root, test=True)
    test_dataloader = DataLoader(test_dataset,batch_size=opt.batch_size,shuffle=False)

    # 初始化统计变量
    num_classes = 15
    normalize = False  # normalize：True-百分比; False-个数
    labels = ["0","1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]
    confusion = ConfusionMatrix(num_classes=num_classes, labels=labels, normalize=normalize)
    confusion_cp = ConfusionMatrix(num_classes=num_classes, labels=labels, normalize=normalize)

    total_samples = 0
    total_error = 0.0
    total_error_cp1 = 0.0
    results = []
    results_cp1 = []
    for ii,(data,target) in tqdm(enumerate(test_dataloader)):

        # start = time.time()

        input = premodel.encoder(data.to(opt.device))
        score = model(input)
        # end = time.time()
        # print(f"执行时间: {end - start:.6f}秒")

        score_cp1 = cp1model(input)

        # 计算预测结果
        probabilities = t.nn.functional.softmax(score, dim=1)  # 获取概率分布
        pred_labels = t.argmax(probabilities,dim=1)  # 取最大概率对应的标签
        probabilities_cp1 = t.nn.functional.softmax(score_cp1, dim=1)  # 获取概率分布
        pred_labels_cp1 = t.argmax(probabilities_cp1,dim=1)  # 取最大概率对应的标签

        confusion.update(pred_labels.to("cpu").numpy()*2, target.to("cpu").numpy())
        confusion_cp.update(pred_labels_cp1.to("cpu").numpy()*2, target.to("cpu").numpy())

        for i in range(pred_labels.shape[0]):
            pred_label = pred_labels[i].item()
            pred_label_cp1 = pred_labels_cp1[i].item()
            true_label = target[i]

            # 定位误差
            error = abs(pred_label*2 - true_label) * 0.4
            error_cp1 = abs(pred_label_cp1*2 - true_label) * 0.4
            total_error += error
            total_error_cp1 += error_cp1
            total_samples += 1
            accuracy = total_error / total_samples
            accuracy_cp1 = total_error_cp1 / total_samples

            results.append(error.item())
            results_cp1.append(error_cp1.item())
        print(f"定位误差率: {accuracy:.4f}")

    print(f"定位误差率: {accuracy:.4f}")
    print(f"对比无attention定位误差率: {accuracy_cp1:.4f}")
    confusion.plot()
    confusion_cp.plot()
    confusion.summary()

    return results, results_cp1


def test_compare(**kwargs):
    opt._parse(kwargs)

    ### 改这里
    results, results_cp1 = test(cae_model_path="checkpoints\good_results_400_100\CAE_0326_32FIFAL.pth", 
                                resnet_model_path="checkpoints\good_results_400_100\Resnet18_0328_ACAT_GREAT.pth",
                                resnet_model_cp1_path="checkpoints\good_results_400_100\Resnet18_0326_ACnAT.pth",
                                test_data_root="./data/test_ac_all/"
                                )

    results_cp2, results_cp3 = test(cae_model_path="checkpoints\good_results_400_100\CAE_0326_32FIFAL.pth", 
                                    resnet_model_path="checkpoints\good_results_400_100\Resnet18_0328_nACnAT2.pth",
                                    resnet_model_cp1_path="checkpoints\good_results_400_100\Resnet18_0328_nACAT2.pth",
                                    test_data_root="./data/test_fft_all/"
                                    )
    
    plot4_cdf(results, results_cp1, results_cp2, results_cp3)







if __name__ == "__main__":
    print(t.__version__) # 2.2.2

    ### ①训练并验证CAE
    # train_CAE(model="CAE")

    ### ②训练并验证resnet --  python -m visdom.server
    # train_ResNet(model="ResNet18", 
    #             load_model_path="checkpoints\do_results\CAE_0402_FINAL.pth",
    #             use_attention=True,
    #             train_data_root = "./data/test_ac_all/")

    ### ③测试
    test_compare()



