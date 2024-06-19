import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer

class LabelEncoder:
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        self.classes_ = None
    
    def encode(self, labels):
        if self.classes_ is None:
            binary_labels = self.mlb.fit_transform(labels).tolist()
            self.classes_ = self.mlb.classes_
        else:
            binary_labels = self.mlb.transform(labels).tolist()
        return binary_labels
    
    def decode(self, binary_labels):
        try:
            if isinstance(binary_labels, torch.Tensor):
                binary_labels = binary_labels.numpy()
            
            if len(binary_labels.shape) == 1:
                binary_labels = binary_labels.reshape(1, -1)
            
            labels_tuple_list = self.mlb.inverse_transform(binary_labels)
            labels = []
            for tup in labels_tuple_list:
                lst = list(tup)
                str_lst = list(map(str, lst))
                labels.append(" ".join(str_lst))
            labels = np.array(labels)
            return labels[0] if len(labels) == 1 else labels
        except AttributeError:
            raise ValueError("Unsupported data type for binary_labels.")
    
    def save(self, file_path: str = "label_encoder.npy"):
        try:
            np.save(file_path, self.classes_)
        except Exception as e:
            raise IOError(f"An error occurred while saving to {file_path}: {str(e)}")
    
    @classmethod
    def from_pretrained(cls, file_path: str = "label_encoder.npy"):
        try:
            instance = cls()
            classes_ = np.load(file_path, allow_pickle=True)
            if classes_.size == 0:
                raise ValueError("Loaded classes are empty.")
            instance.classes_ = classes_
            instance.mlb.fit([classes_])
            return instance
        except FileNotFoundError:
            raise FileNotFoundError(f"No file found at {file_path}")