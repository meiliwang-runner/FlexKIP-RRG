# from lightning.pytorch import LightningDataModule
# from torch.utils.data import DataLoader

# # 导入两种 dataset 创建函数
# from dataset.data_module_withregion import create_datasets as create_sip_datasets, custom_collate_fn
# from dataset.data_helper import create_datasets as create_default_datasets


# class DataModule(LightningDataModule):

#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.collate_fn = None  # 默认先置空

#     def prepare_data(self):
#         """
#         可选：下载、预处理（此处略）
#         """
#         pass

#     def setup(self, stage: str):
#         """
#         初始化训练/验证/测试数据集
#         """
#         if getattr(self.args, "use_sip", False):
#             train_dataset, dev_dataset, test_dataset = create_sip_datasets(self.args)
#             self.collate_fn = custom_collate_fn
#         else:
#             train_dataset, dev_dataset, test_dataset = create_default_datasets(self.args)
#             self.collate_fn = None  # 默认 PyTorch collation

#         self.dataset = {
#             "train": train_dataset,
#             "validation": dev_dataset,
#             "test": test_dataset
#         }

#     def train_dataloader(self):
#         return DataLoader(
#             self.dataset["train"],
#             batch_size=self.args.batch_size,
#             drop_last=True,
#             pin_memory=True,
#             num_workers=self.args.num_workers,
#             prefetch_factor=self.args.prefetch_factor,
#             collate_fn=self.collate_fn
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.dataset["validation"],
#             batch_size=self.args.val_batch_size,
#             drop_last=False,
#             pin_memory=True,
#             num_workers=self.args.num_workers,
#             prefetch_factor=self.args.prefetch_factor,
#             collate_fn=self.collate_fn
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             self.dataset["test"],
#             batch_size=self.args.test_batch_size,
#             drop_last=False,
#             pin_memory=False,
#             num_workers=self.args.num_workers,
#             prefetch_factor=self.args.prefetch_factor,
#             collate_fn=self.collate_fn
#         )

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from dataset.data_helper import create_datasets as create_default_datasets


class DataModule(LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.collate_fn = None  # 默认先置空

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if getattr(self.args, "use_sip", False):
            # ✅ 延迟导入，避免循环引用
            from dataset.data_helper_withregion import create_datasets as create_sip_datasets, custom_collate_fn
            train_dataset, dev_dataset, test_dataset = create_sip_datasets(self.args)
            self.collate_fn = custom_collate_fn
        else:
            train_dataset, dev_dataset, test_dataset = create_default_datasets(self.args)
            self.collate_fn = None

        self.dataset = {
            "train": train_dataset,
            "validation": dev_dataset,
            "test": test_dataset
        }

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.args.batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=self.args.num_workers,
            prefetch_factor=self.args.prefetch_factor,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.args.val_batch_size,
            drop_last=False,
            pin_memory=True,
            num_workers=self.args.num_workers,
            prefetch_factor=self.args.prefetch_factor,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.args.test_batch_size,
            drop_last=False,
            pin_memory=False,
            num_workers=self.args.num_workers,
            prefetch_factor=self.args.prefetch_factor,
            collate_fn=self.collate_fn
        )
