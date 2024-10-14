from sklearn.datasets import fetch_openml

from automl.schema.descriptors import DatasetDescriptor

# 1461,31,29
x = fetch_openml(data_id=1461, as_frame=True, parser="pandas")
dataset = x["frame"]
print(f"dataset Shape {dataset.shape}")
dataset.head()


DatasetDescriptor.build_from_dataset(dataset, target_columns=["Class"])
