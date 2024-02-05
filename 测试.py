import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 构建FCN模型
def create_fcn_model():
    model = keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(320, 640, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    return model


# 计算评价指标
def compute_metrics(true_labels, pred_labels):
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    iou = jaccard_score(true_labels, pred_labels, average='weighted')
    return precision, recall, f1, iou


# 加载数据集
x_train = np.load("./train/x_train.npy", allow_pickle=True, encoding="latin1")
y_train = np.load("./train/y_train.npy", allow_pickle=True, encoding="latin1")
x_test = np.load("./test/x_test.npy", allow_pickle=True, encoding="latin1")
y_test = np.load("./test/y_test.npy", allow_pickle=True, encoding="latin1")
x_validation = np.load("./validation/x_validation.npy", allow_pickle=True, encoding="latin1")
y_validation = np.load("./validation/y_validation.npy", allow_pickle=True, encoding="latin1")
h, w = 320, 640  # 图片的高度和宽度
x_train = x_train.reshape(-1, h, w, 1)
x_test = x_test.reshape(-1, h, w, 1)
x_validation = x_validation.reshape(-1, h, w, 1)

# 定义超参数
epochs = 10

# 创建模型
model = create_fcn_model()

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 定义记录指标变化的列表
precisions = []
recalls = []
f1_scores = []
ious = []

# 训练模型
for epoch in range(epochs):
    # 在每个epoch中迭代训练集，并调用model.fit()方法进行训练
    model.fit(x_train, y_train, batch_size=32, epochs=1, validation_data=(x_validation, y_validation))
    # 使用验证集评估模型
    y_pred = model.predict(x_validation)
    y_pred = np.argmax(y_pred, axis=1)
    precision, recall, f1, iou = compute_metrics(y_validation, y_pred)

    # 将评价指标添加到列表中
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    ious.append(iou)

# 画出评价指标的变化曲线
plt.plot(range(epochs), precisions, label='Precision')
plt.plot(range(epochs), recalls, label='Recall')
plt.plot(range(epochs), f1_scores, label='F1-Score')
plt.plot(range(epochs), ious, label='IOU')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()

# 导出模型参数
# 创建一个DataFrame对象
data = pd.DataFrame({'Precision': precisions, 'Recall': recalls,'F1_score': f1_scores,'IOU': ious})
# 将DataFrame对象导出到Excel文件
data.to_excel('模型相关参数.xlsx',index = False)
# 根据准确度大小，找到验证数据集上性能最好的模型
best_model_index = np.argmax(precisions)
best_model = model

# 使用最优模型在测试数据集上做预测
y_pred_test = best_model.predict(x_test)
y_pred_test = np.argmax(y_pred_test, axis=1)

# 计算指标
precision_test, recall_test, f1_test, iou_test = compute_metrics(y_test, y_pred_test)

# 打印测试数据集上的性能指标
print("测试数据评价指标:")
print("准确度:", precision_test)
print("召回率:", recall_test)
print("F1得分:", f1_test)
print("相关度:", iou_test)