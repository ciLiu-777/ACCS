import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
# from CAE_Pretrain import CAE
from torchvision import transforms
from PIL import Image
from scipy.ndimage import zoom
import models


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output)
        self.model.zero_grad()
        class_score = output[:, class_idx]
        class_score.backward(retain_graph=True)

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = zoom(cam, 28 / cam.shape[0], order=1)
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        return cam

    def plot_cam(self, img, cam, save_path=None):
        plt.figure(figsize=(10, 10))
        plt.imshow(img, alpha=0.8)
        plt.imshow(
            cam, cmap="jet", interpolation="sinc", alpha=0.5
        )  # 使用matplotlib叠加热图
        plt.axis("off")
        # 添加自动保存功能
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.1)
            plt.close()
        else:
            plt.show()
        plt.show()


# 在可视化对比部分添加保存逻辑
# -------------------- 可视化对比 --------------------
# 创建保存目录
save_dir = "./tmp/gradcam/"
os.makedirs(save_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

# 配置学术绘图参数
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 300,
    }
)


# 使用示例
device = torch.device("cpu")  # 如果有GPU支持，可以使用 torch.device('cuda')
transform = transforms.Compose(
    [
        # transforms.Grayscale(num_output_channels=1),  # 转换为灰度图像
        # transforms.Resize((224, 224)),  # ResNet18 expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)
img_path = (
    "./data/test_ac_all/group04.22.jpg"  # 不具有可解释性，使注意力注意到机器关注特征处
)
image = Image.open(img_path)
# 应用转换操作
input_tensor = transform(image)

# 增加批次维度，与DataLoader中的输出一致
input_tensor = input_tensor.unsqueeze(0).to(device)


# configure model
premodel = getattr(models, "CAE")()
premodel.load("checkpoints\good_results_400_100\CAE_0326_32FIFAL.pth")
model = getattr(models, "ResNet18")(use_attention=True)
model.load("checkpoints\good_results_400_100\Resnet18_0328_ACAT_GREAT.pth")
cp1model = getattr(models, "ResNet18")(use_attention=False)
cp1model.load("checkpoints\good_results_400_100\Resnet18_0326_ACnAT.pth")

device = torch.device("cpu")
premodel.to(device)
model.to(device)
cp1model.to(device)

premodel.eval()
model.eval()
cp1model.eval()


# 获取编码器的输出
encoded_feature = premodel.encoder(input_tensor)

# -------------------- 选择目标层 --------------------
# 带注意力的模型：定位到CBAM的空间注意力层（sa）
target_layer_with_attn = model.layer4

# 不带注意力的模型：定位到最后一个卷积层（示例：layer4最后一个残差块的第二个卷积）
target_layer_without_attn = cp1model.layer4

# -------------------- 创建GradCAM对象 --------------------
# 带注意力的模型
grad_cam_attn = GradCAM(model, target_layer_with_attn)

# 不带注意力的对比模型
grad_cam_cp1 = GradCAM(cp1model, target_layer_without_attn)

# -------------------- 生成对比热力图 --------------------
# 注意：需确保输入经过正确的前处理
with torch.no_grad():
    encoded_feature = premodel.encoder(input_tensor)  # 编码器处理

cam_attn = grad_cam_attn.generate_cam(encoded_feature)
cam_cp1 = grad_cam_cp1.generate_cam(encoded_feature)

# 可视化
# 将 encoded_feature 转换为 numpy 数组以便可视化
image_data = encoded_feature[0].detach().numpy().transpose(1, 2, 0)
img = (image_data - image_data.min()) / (image_data.max() - image_data.min())

# 确保图像为三通道
if img.shape[2] == 1:
    img = np.repeat(img, 3, axis=-1)

# -------------------- 可视化对比 --------------------
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 7))

ax0.imshow(img)
ax0.set_title("Original Feature Map")
ax0.axis("off")

# 带注意力的热图
ax1.imshow(img, alpha=0.8)
ax1.imshow(cam_attn, cmap="jet", alpha=0.5)
ax1.set_title("With CBAM Attention")
ax1.axis("off")

# 不带注意力的热图
ax2.imshow(img, alpha=0.8)
ax2.imshow(cam_cp1, cmap="jet", alpha=0.5)
ax2.set_title("Without Attention")
ax2.axis("off")

plt.show()

# 保存对比图
fig.savefig(
    f"{save_dir}GradCAM_Comparison_{timestamp}.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.05,
)
# fig.savefig(
#     f"{save_dir}GradCAM_Comparison_{timestamp}.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )

# 单独保存每个子图
for i, (ax, suffix) in enumerate(
    zip([ax0, ax1, ax2], ["Original", "With_Attention", "Without_Attention"])
):
    # 提取子图数据
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    # 保存子图
    fig.savefig(
        f"{save_dir}GradCAM_{suffix}_{timestamp}.png",
        dpi=600,
        bbox_inches=extent,
        pad_inches=0.05,
    )
    # fig.savefig(
    #     f"{save_dir}GradCAM_{suffix}_{timestamp}.pdf",
    #     format="pdf",
    #     bbox_inches=extent,
    #     pad_inches=0.05,
    # )
