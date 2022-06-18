import os

from dataset.bay_dataset import PemsBay,MissingValuesPemsBay
from model.utils.metrics import *

from dataset.bay_dataset import PemsBay,MissingValuesPemsBay

from utils.parser_utils import str_to_bool
from utils import parser_utils, numpy_metrics, ensure_list, prediction_dataframe,get_args_from_json
from main import parse_args,dataset_loader,model_loader

checkpoint_path='logs//gril//pems-bay'
def test(dataset,datamodule,network,args):
    #testing
    logdir=os.path.join(args.logdir,args.model_name,args.dataset_name)
    modelstate_path='epoch=292-step=42778.ckpt'
    bestmodel_path=os.path.join(logdir,modelstate_path)
    # checkpoint_callback = ModelCheckpoint(dirpath=logdir, save_top_k=1, monitor='val_mae', mode='min')
    network.load_state_dict(torch.load(bestmodel_path,
                                      lambda storage, loc: storage)['state_dict'])
    network.freeze()
    #trainer.test()
    network.eval()

    if torch.cuda.is_available():
        network.cuda()

    with torch.no_grad():
        y_true, y_hat, mask = network.predict_loader(datamodule.test_dataloader(), return_mask=True)
    y_hat = y_hat.detach().cpu().numpy().reshape(y_hat.shape[:3])  # reshape to (eventually) squeeze node channels

    # Test imputations in whole series
    eval_mask = dataset.eval_mask[datamodule.test_slice]
    df_true = dataset.df.iloc[datamodule.test_slice]
    metrics = {
        'mae': numpy_metrics.masked_mae,
        'mse': numpy_metrics.masked_mse,
        'mre': numpy_metrics.masked_mre,
        'mape': numpy_metrics.masked_mape
    }
    # Aggregate predictions in dataframes
    index = datamodule.torch_dataset.data_timestamps(datamodule.testset.indices, flatten=False)['horizon']
    aggr_methods = ensure_list(args.aggregate_by)
    df_hats = prediction_dataframe(y_hat, index, dataset.df.columns, aggregate_by=aggr_methods)
    df_hats = dict(zip(aggr_methods, df_hats))
    for aggr_by, df_hat in df_hats.items():
        # Compute error
        print(f'- AGGREGATE BY {aggr_by.upper()}')
        for metric_name, metric_fn in metrics.items():
            error = metric_fn(df_hat.values, df_true.values, eval_mask).item()
            print(f' {metric_name}: {error:.4f}')

    return y_true, y_hat, mask

if __name__=='__main__':
    
    args=parse_args()
    #load dataset after wrapping 
    dataset_module,adj=dataset_loader(args)
    dataset=MissingValuesPemsBay()
    network=model_loader(dataset_module,adj,args)
    test(dataset,dataset_module,network,args)

