# 数据放置说明

请将教师提供的数据整理为以下结构后再运行代码：

```text
data/
├── train/
├── test/
└── train_labels.csv
```

其中：

- `train/` 中存放训练集 EEG `.npy` 文件
- `test/` 中存放测试集 EEG `.npy` 文件
- `train_labels.csv` 中至少包含两列：`eeg_file` 和 `label`

`label` 取值为：

- `background`
- `target`
