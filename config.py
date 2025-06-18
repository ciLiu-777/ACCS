# coding:utf8
import warnings
import torch as t


class DefaultConfig(object):
    env = "default"  # visdom 环境
    vis_port = 8097  # visdom 端口 python -m visdom.server
    tensorboard_port = 6006  # tensorboard 端口tensorboard --logdir=tmp/logs --port=6006
    model = "CAE"  # 使用的模型，名字必须与models/__init__.py中的名字一致

    train_data_root = "./data/train/"  # 训练集存放路径
    test_data_root = "./data/test/"  # 测试集存放路径
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载
    use_attention = True    # 是否使用注意力

    batch_size = 16  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 7  # how many workers for loading data
    print_freq = 16*4  # print info every N batch

    debug_file = "/tmp/debug"  # if os.path.exists(debug_file): enter ipdb,
    result_file = "/tmp/result.csv"

    max_epoch = 10
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = t.device("cuda") if opt.use_gpu else t.device("cpu")

        print("user config:")
        for k, v in self.__class__.__dict__.items():
            if not k.startswith("_"):
                print(k, getattr(self, k))


opt = DefaultConfig()
