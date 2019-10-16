class DefaultConfig:
    start_epoch = 0
    max_epoch = 15
    seed = 0
    dataset = 'food_101'
    workers = 1
    train_batch = 16  ##32 cuda out of memory
    val_batch = 32
    pretrained_model = True
    loss_fnc = 'CrossEntropy'
    optim = 'adma'
    lr = 0.0001
    weight_decay = 0.1
    print_freq = 30
    adjust_lr = False
    best_rank = -np.inf
    eval_step = 50  ##节省训练时间
    mode = ""
    save_dir = ''  # 当前目录下
    trained_path='cache/classmodel/.....'        #训练好的分类模型



    def _parse(self, wkargs):
        for k, v in wkargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in DefaultConfig.__dict__.items() if not k.startswith('_')}

opt = DefaultConfig()