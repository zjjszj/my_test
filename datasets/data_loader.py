from PIL import Image
from torch.utils.data import Dataset


def read_img(img_path):
    img = Image.open(img_path).convert('RGB')
    return img


class ImageData(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img_path, c_id = self.dataset[index]
        img = read_img(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, c_id

    def __len__(self):
        return len(self.dataset)