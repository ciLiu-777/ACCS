# coding:utf8
import visdom
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from matplotlib.ticker import FuncFormatter
import numpy as np
from prettytable import PrettyTable
import itertools
import seaborn as sns  # 解决样式问题
from datetime import datetime

def write_csv(results, file_name):
    import csv

    with open(file_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        writer.writerows(results)


class Visualizer(object):
    """
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    """

    def __init__(self, env="default", **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)

        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ""

    def reinit(self, env="default", **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(
            Y=np.array([y]),
            X=np.array([x]),
            win=name,
            opts=dict(title=name),
            update=None if x == 0 else "append",
            **kwargs,
        )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)

        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        """
        self.vis.images(img_.cpu().numpy(), win=name, opts=dict(title=name), **kwargs)

    def log(self, info, win="log_text"):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += "[{time}] {info} <br>".format(
            time=time.strftime("%m%d_%H%M%S"), info=info
        )
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)


def plot_cdf(errors, errors_cp1):
    # 将误差数组排序
    sorted_errors = np.sort(errors)
    sorted_errors_cp1 = np.sort(errors_cp1)
    n = len(sorted_errors)
    n_cp1 = len(sorted_errors_cp1)

    # 生成密集的x值点
    max_error = sorted_errors[-1]
    max_error_cp1 = sorted_errors_cp1[-1]
    x = np.linspace(0, max_error, 1000)
    x_cp1 = np.linspace(0, max_error_cp1, 1000)

    # 计算每个x对应的累积概率
    y = np.searchsorted(sorted_errors, x, side="left") / n
    y_cp1 = np.searchsorted(sorted_errors_cp1, x, side="left") / n_cp1

    # 创建图形
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_cp1, label="without attention", linewidth=3, color="green")
    plt.plot(x, y, label="with attention", linewidth=3, color="red")
    plt.xlim(-0.05, 6.0)
    plt.ylim(0, 1.05)
    plt.xlabel("Error Distance (m)")
    plt.ylabel("Accuracy")
    plt.title("Cumulative Distribution Function of Errors")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.show()


def plot4_cdf(errors, errors_cp1, errors_cp2, errors_cp3):
    # 字体配置
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.linewidth"] = 1.0  # 坐标轴线宽

    # 创建紧凑画布 (适合屏幕显示的黄金比例)
    fig = plt.figure(figsize=(6.4, 4.8), dpi=100)  # 默认显示尺寸
    ax = fig.add_subplot(111)

    # 生成数据
    base_x = np.linspace(0, 6, 1000)

    # 优化后的线型配置
    line_config = {
        "-AC, -Attn": {  # 基准线1
            "color": "#757575",    # 中性灰 (降低视觉权重)
            "linestyle": (0, (5, 2)),  # 长虚线
            "linewidth": 1.6
        },
        "-AC, +Attn": {  # 基准线2
            "color": "#E69F00",    # 橙色 (高对比且色盲安全)
            "linestyle": (0, (3, 1, 1, 1)),  # 点划线
            "linewidth": 1.6
        },
        "+AC, -Attn": {  # 基准线3
            "color": "#009E73",    # 深青绿 (与橙色形成互补)
            "linestyle": "--",     # 标准短虚线
            "linewidth": 1.6
        },
        "+AC, +Attn": {  # 主算法
            "color": "#0072B2",    # IEEE标准深蓝 (最高视觉优先级)
            "linestyle": "-",      # 实线
            "linewidth": 2.0
        }
    }

    # 绘制曲线
    for data, label in zip(
        [errors_cp3, errors_cp2, errors_cp1, errors],
        ["-AC, -Attn", "-AC, +Attn", "+AC, -Attn", "+AC, +Attn"],
    ):
        sorted_data = np.sort(data)
        # 关键修复步骤：正确插入原点
        # --------------------------------------------------
        # 1. 在基础x轴前插入x=0
        x_aug = np.insert(base_x, 0, 0)  # 现在x_aug长度是1001

        # 2. 计算对应y值（包含x=0）
        y = np.searchsorted(sorted_data, x_aug, side="right") / len(data)

        # 3. 强制设置y[0]=0（原点起始）
        y[0] = 0.0  # 对应x=0的位置

        # 4. 确保维度对齐
        assert len(x_aug) == len(y), "x和y维度必须一致"
        ax.plot(
            x_aug,
            y,
            **line_config[label],
            label=label,
            solid_capstyle="round",
            dash_capstyle="round",
        )  # 端点圆角处理

    # 坐标轴优化
    ax.set_xlim(0, 6.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Localization Error (m)", fontsize=17, labelpad=5)
    ax.set_ylabel("Cumulative Probability", fontsize=17, labelpad=5)

    # 刻度设置
    ax.tick_params(axis="both", which="major", labelsize=9, length=4, width=1)
    ax.tick_params(axis="both", which="minor", length=2, width=0.8)
    ax.set_xticks(np.arange(0, 6.5, 1.0))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.xaxis.get_major_ticks()[0].label1.set_fontweight("bold")  # x=0加粗
    ax.yaxis.get_major_ticks()[0].label1.set_fontweight("bold")  # y=0加粗
    # 绘制原点标记
    # ax.plot(0, 0, 'ko', markersize=4, zorder=10)  # 黑色原点标记

    # 紧凑图例
    leg = ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        edgecolor="#2F2F2F",
        fontsize=9,
        handlelength=2.5,
        borderpad=0.6,
        borderaxespad=0.8,
    )
    leg.get_frame().set_linewidth(1.0)  # 加粗图例边框

    # 网格设置（更精细的虚线）
    ax.grid(True, linestyle=(0, (3, 3)), linewidth=0.6, alpha=0.7, which="major")

    # 自动调整布局
    plt.tight_layout(pad=1.8)

    # 生成时间戳文件名
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"./tmp/CDF_Comparison_{timestamp}.png"
    # 保存为PNG（保留PDF版本可按需取消注释）
    # plt.savefig(filename, format='png', 
    #         bbox_inches='tight', pad_inches=0.05, dpi=300)
    # 保存高分辨率图片
    # plt.savefig('CDF_Compact.pdf', format='pdf',
    #             bbox_inches='tight', dpi=300)

    plt.show()


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """

    def __init__(self, num_classes: int, labels: list, normalize: bool):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        self.normalize = normalize

    def update(self, preds, labels):
        """
        更新混淆矩阵（重要修复：行列顺序）
        - preds: 预测标签数组
        - labels: 真实标签数组
        """
        # 确保输入为整数类型
        preds = preds.astype(int)
        labels = labels.astype(int)

        for p, t in zip(preds, labels):
            self.matrix[t, p] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = self.matrix[i, :].sum() - TP
            FN = self.matrix[:, i].sum() - TP
            TN = self.matrix.sum() - TP - FP - FN
            
            precision = TP / (TP + FP) if (TP + FP) else 0
            recall = TP / (TP + FN) if (TP + FN) else 0
            specificity = TN / (TN + FP) if (TN + FP) else 0
            
            table.add_row([
                self.labels[i],
                f"{precision:.3f}",
                f"{recall:.3f}",
                f"{specificity:.3f}"
            ])
        print(table)

    def plot(self):

        # print(matrix)
        # self.plot_confusion_matrix()
        self.plot_confusion_matrix_great()

    def plot_confusion_matrix(self):
        matrix = self.matrix
        classes = self.labels
        normalize = self.normalize
        title = "Confusion matrix"
        cmap = plt.cm.Blues  # 绘制的颜色

        plt.figure(figsize=(9, 9))  # 增大画布尺寸
        # 设置学术字体规范
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 12,
            'mathtext.fontset': 'stix',
        })
        # print("normalize: ", normalize)

        """
        - matrix : 计算出的混淆矩阵的值
        - classes : 混淆矩阵中每一行每一列对应的列
        - normalize : True:显示百分比, False:显示个数
        """
        if normalize:
            matrix = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
        else:
            fmt = ".0f"
        plt.imshow(matrix, interpolation="nearest", cmap=cmap)
        plt.title(title, fontsize=16)
        cbar = plt.colorbar(fraction=0.046, pad=0.04)  # 调整颜色条比例
        cbar.set_ticklabels(['0.00', '0.20', '0.40', '0.60', '0.80', '1.00'])
        cbar.outline.set_visible(False)  # 移除颜色条边框
        cbar.ax.tick_params(labelsize=12)

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, ha="right")
        plt.yticks(tick_marks, classes, fontsize=12)
        # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
        plt.ylim(len(classes) - 0.5, -0.5)

        fmt = ".2f" if normalize else ".0f"
        thresh = matrix.max() / 2.0
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(
                j,
                i,
                format(matrix[i, j], fmt),
                fontsize=12,  # 调整字体大小
                horizontalalignment="center",
                verticalalignment="center",  # 添加垂直居中
                color="white" if matrix[i, j] > thresh else "black",
            )
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.subplots_adjust(bottom=0.15, left=0.15)  # 调整边距

        # 保存选项
        #plt.savefig("confusion_matrix.pdf", bbox_inches="tight", dpi=300)
        plt.show()

    def plot_confusion_matrix_great(self):
        matrix = self.matrix
        classes = self.labels
        normalize = self.normalize
        title = "Confusion matrix"
        cmap = plt.cm.Blues  # 绘制的颜色
        # 创建画布和坐标轴
        fig, ax = plt.subplots(figsize=(9, 9))
    
        # 设置学术字体规范
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 12,
            'mathtext.fontset': 'stix',
        })
        
        # 标准化处理
        if normalize:
            matrix = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
            vmax = 1.0  # 标准化时设置最大值为1
        else:
            fmt = ".0f"
            vmax = matrix.max()  # 非标准化时使用最大值

        # 绘制热力图
        im = ax.imshow(matrix, interpolation="nearest", cmap=cmap, 
                    vmin=0, vmax=vmax, aspect='auto')
    
        # 移除所有外围边框
        for spine in ax.spines.values():
            spine.set_visible(False)


        plt.imshow(matrix, interpolation="nearest", cmap=cmap)
        # plt.title(title, fontsize=16)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticklabels(['0.00', '0.20', '0.40', '0.60', '0.80', '1.00'])
        cbar.outline.set_visible(False)  # 移除颜色条边框
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.tick_params(labelsize=10, width=0.5)  # 调整刻度粗细
        
        # 坐标轴设置
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_yticklabels(classes, fontsize=10)

        # 添加数值文本
        thresh = matrix.max() / 2.
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            ax.text(j, i, format(matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black",
                    fontsize=10)
        
        # 标签设置（符合IEEE图表规范）
        ax.set_xlabel('Predicted Label', fontsize=24, labelpad=10)
        ax.set_ylabel('True Label', fontsize=24, labelpad=10)
        
        # 调整布局和边距
        plt.tight_layout()
        fig.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)

        
        # 标签设置
        ax.set_xlabel('Predicted Label', fontsize=24, labelpad=10)
        ax.set_ylabel('True Label', fontsize=24, labelpad=10)

        # # 生成时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"./tmp/confusion_matrix_{timestamp}.png"
        # 保存为PNG（保留PDF版本可按需取消注释）
        # plt.savefig(filename, format='png', 
        #         bbox_inches='tight', pad_inches=0.05, dpi=300)
        # # 保存为矢量图（推荐IEEE使用PDF格式）
        # plt.savefig("./tmp/confusion_matrix.pdf", format="pdf", bbox_inches="tight", 
        #             pad_inches=0.05, dpi=300)

        plt.show()