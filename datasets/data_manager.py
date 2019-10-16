import os

#train:val:test=65650:15150:20200
class Food101:
    def __init__(self, img_path=r'/kaggle/input/food-101/food-101/food-101/images'):
        self.train, self.val, self.test = self._process_dir(img_path)
        print("=> Food101 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   |  images")
        print("  ------------------------------")
        print("  train    |  {:8d}".format(len(self.train)))
        print("  val      |  {:8d}".format(len(self.val)))
        print("  test     |  {:8d}".format(len(self.test)))
        print("  ------------------------------")

    def _process_dir(self, img_path):
        c_id, count = 0, 1
        train, val, test = [], [], []
        for c_name in os.listdir(img_path):
            if not c_name == '.DS_Store':
                for rootpath, _, filenames in os.walk(os.path.join(img_path, c_name)):
                    for filename in filenames:
                        if count <= 650:
                            train.append((os.path.join(rootpath, filename), c_id))
                            count += 1
                        elif count <= 800:
                            val.append((os.path.join(rootpath, filename), c_id))
                            count += 1
                        else:
                            test.append((os.path.join(rootpath, filename), c_id))
                            count += 1
                c_id += 1
                count = 1
        return train, val, test

# datasets=Food101()