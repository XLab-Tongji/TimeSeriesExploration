class Config_bay():
    def __init__(self):
        self.lr=1e-4
        self.dataset_kwargs={
            "window":24,
            "horizon":24,
            "delay":0,
            "stride":1
        }
        self.split_kwargs={
            "in_sample":False,
            "val_len":0.1,
            "test_len":0.2,
        }
        
    def get_config(self):
        params={
            'lr':self.lr,
            'dataset':self.dataset_kwargs,
        }

        return params
