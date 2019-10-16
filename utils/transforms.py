from torchvision import transforms as T

class TrainTransform:
    def __call__(self,x):
        '''
        x:Image类型
        '''
        w,h=x.size
        if h<512:
            x = T.Resize((512,int(512 * (512 / h))))(x)
        if w<512:
            x=T.Resize((int(512*(512/w)),512))(x)
        x=T.RandomCrop((448,448))(x)
        x = T.RandomHorizontalFlip()(x)
        x = T.ToTensor()(x)
        x = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
        return x

class TestTransform:
    def __call__(self,x):
        w,h=x.size
        if h<512:
            x = T.Resize((512,int(512 * (512 / h))))(x)
        if w<512:
            x=T.Resize((int(512*(512/w)),512))(x)
        x=T.RandomCrop((448,448))(x)
        x = T.ToTensor()(x)
        x = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
        return x