import os
import argparse
import datetime

from torch.utils.data import DataLoader
import wandb

from src.read_data import *
from src.vit_new import ViT, train
from src.he2rna import HE2RNA, fit
from src.tformer_lin import ViS 

def custom_collate_fn(batch):
    """Remove bad entries from the dataloader
    Args:
        batch (torch.Tensor): batch of tensors from the dataaset
    Returns:
        collate: Default collage for the dataloader
    """

    try:
        batch = list(filter(lambda x: x[0] is not None, batch))
    except:
        batch['image'] = []
    return torch.utils.data.dataloader.default_collate(batch)

def filter_no_features(df, feature_path = "examples/features"):
    no_features = []
    for i, row in df.iterrows():
        row = row.to_dict()
        wsi = row['wsi_file_name']
        project = row['tcga_project']
        path = os.path.join(feature_path, project, wsi, wsi+'.h5')
        if not os.path.exists(path):
            no_features.append(wsi)
    df = df[~df['wsi_file_name'].isin(no_features)]
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Getting features')
    parser.add_argument('--save_dir', type=str, default="/examples/pretrained_model",  help='save directory')
    parser.add_argument('--path_csv', type=str, default="/examples/ref_file.csv", help='path to reference file with gene expression data')
    parser.add_argument('--feature_path', type=str, default="/examples/features", help='path to resnet and clustered features')
    parser.add_argument('--exp_name', type=str, default="exp", help='Experiment name used to create saved model name')
    parser.add_argument('--log', type=int, default=0, help='whether to log the loss')
    parser.add_argument('--model', type=str, default='vit', help='model to pretrain, "he2rna" for MLP aggregation, "vit" for transformer aggregation or "vis" for linearized transformer aggregation')
    parser.add_argument('--seed', type=int, default=99, help='Seed for random generation')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size to train')
    parser.add_argument('--n_workers', type=int, default=8, help='number of workers to train')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint from trained model')
    parser.add_argument('--quick', type=int, default=0, help='Whether to run a quick exp for debugging')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ############################################## logging and save dir ##############################################
    if args.exp_name == "":
        args.exp_name = '{date:%Y-%m-%d}'.format(date=datetime.datetime.now())
    else:
        args.exp_name = '{date:%Y-%m-%d}'.format(date=datetime.datetime.now()) + "_" + args.exp_name

    save_dir = os.path.join(args.save_dir, args.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    run = None
    if args.log:
        run = wandb.init(project="visgene", entity='account_name', config=args, name=args.exp_name)

    ############################################## prepare data ##############################################
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)

    df = pd.read_csv(args.path_csv)
    df = filter_no_features(df)

    if args.quick:
        df = df.iloc[0:20, :]
        args.num_epochs = 5

    dataset = SuperTileRNADataset(df, args.feature_path)

    dataloader = DataLoader(dataset,
                num_workers=args.n_workers, pin_memory=True,
                shuffle=True, batch_size=args.batch_size,
                collate_fn=custom_collate_fn)

    ############################################## model ##############################################
    
    if args.model == 'vis':
        model = ViS(num_outputs=dataset.num_genes, input_dim=dataset.feature_dim,  
                            depth=6, nheads=16, 
                            dimensions_f=64, dimensions_c=64, dimensions_s=64, device=device)
    elif args.model == 'vit':
        model = ViT(num_outputs=dataset.num_genes,
                    dim=dataset.feature_dim, depth=6, heads=16, mlp_dim=2048, dim_head = 64,
                    device=device)

    elif args.model == 'he2rna':
        model = HE2RNA(input_dim=dataset.feature_dim, layers=[256,256],
                ks=[1,2,5,10,20,50,100],
                output_dim=dataset.num_genes, device=device)
    else:
        print('please specify correct model name, "vit" or "he2rna"')
        exit()

    if args.checkpoint != None:
        model.load_state_dict(torch.load(args.checkpoint))
    model = model.to(device)

    ############################################## training ##############################################
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=3e-3,weight_decay=0.)
    dataloaders = {'train': dataloader,}

    if args.model == 'vit':
        model = train(model, dataloaders, optimizer, num_epochs=args.num_epochs, phases=['train'], save_dir=save_dir, run=run)
    else:
        model = fit(model=model, lr=3e-3, train_loader=dataloaders['train'], valid_loader=None, test_loader=None,
                            params={}, fold=None, optimizer=None, path=save_dir)

    print('Finished pre-training')
