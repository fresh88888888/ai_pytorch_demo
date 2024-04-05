import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from transformers import TrainingArguments, Trainer

housing = fetch_california_housing(as_frame=True)
housing = housing['data'].join(housing['target']).sample(1000, random_state=52)
print(housing.head())

np.set_printoptions(precision=2, suppress=True)
# print(np.corrcoef(housing, rowvar=False))


def corr(x, y): return np.corrcoef(x, y)[0][1]


def corr_d(eval_pred): return {'pearson': corr(*eval_pred)}


print(corr(housing.MedInc, housing.MedHouseVal))


def show_corr(df, a, b):
    x, y = df[a], df[b]
    plt.scatter(x, y, alpha=0.5, s=4)
    plt.title(f'{a} vs {b}; r: {corr(x, y):.2f}')


# show_corr(housing, 'MedInc', 'MedHouseVal')
# show_corr(housing, 'MedInc', 'AveRooms')
# subset = housing[housing.AveRooms < 15]
# show_corr(subset, 'MedInc', 'AveRooms')
subset = housing[housing.AveRooms < 15]
# show_corr(subset, 'MedHouseVal', 'AveRooms')
# show_corr(subset, 'HouseAge', 'AveRooms')

# plt.show()

# Training

# bs = 128
# epochs = 4
# lr = 8e-5
# args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
#                          evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
#                          num_train_epochs=epochs, weight_decay=0.01, report_to='none')
# model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)
# trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
#                   tokenizer=tokz, compute_metrics=corr_d)
# preds = trainer.predict(eval_ds).predictions.astype(float)
# preds = np.clip(preds, 0, 1)

# submission = datasets.Dataset.from_dict({
#     'id': eval_ds['id'],
#     'score': preds
# })

# submission.to_csv('submission.csv', index=False)
