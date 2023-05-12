from torch.utils.data import Dataset


class DcclRecDataset(Dataset):
    """
    功能描述：继承自torch.utils.data.Dataset，它是代表这一数据的抽象类。通过继承和重写此
    类，实现模型的小批量优化，即每次都会从原数据集中取出一小批量进行训练，完成一次权重更新后
    ，再从原数据集中取下一个小批量数据，然后再训练再更新。
    接口：（1）__init__()：初始化参数；
         （2）__len__()：提供数据的大小；
         （3）__getitem__()：通过给定索引获取数据与标签。
    修改记录：
    """

    def __init__(self,
                 features,
                 labels,
                 pop_ratios
                 ):
        """
        功能描述：初始化参数。
        参数：
        修改记录：
        """
        super(DcclRecDataset, self).__init__()
        self.features = features
        self.labels = labels
        self.pop_ratios = pop_ratios

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        功能描述：
        参数：（1）idx：
        返回值：
        修改记录：
        """
        idx = idx
        if self.pop_ratios is None:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx], self.labels[idx], self.pop_ratios[idx]
