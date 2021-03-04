import model
import dataset
import argparse
import os

#bool type for arguments
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#Define arguments
parser = argparse.ArgumentParser(description='Download dataset')
parser.add_argument("--data_path", type=str,default='./data')
parser.add_argument("--image_size", type=int,default=256)
parser.add_argument("--split_components", type=str2bool,default=True)
parser.add_argument("--patch_size", type=int,default=10)
parser.add_argument("--zip_weights", type=str2bool,default=False)
parser.add_argument("--network", type=str,default='melnyk',choices=['VGG16','inception-resnet','mobilenet','efficient-net','melnyk'])
parser.add_argument("--fc_link", type=str,default='',choices=['', 'GAP','GWAP','GWOAP'])
parser.add_argument("--iterative_refinement", type=str2bool,default=False)
parser.add_argument("--refinement_t", type=int,default=4)
parser.add_argument("--data_augmentation", type=str2bool,default=False)
parser.add_argument("--fit_discriminator", type=str2bool,default=True)
parser.add_argument("--adversarial_learning", type=str2bool,default=False)
parser.add_argument("--adversarial_ratio", type=float,default=0.15)
parser.add_argument("--silent_mode", type=str2bool,default=False)

parser.add_argument("--log_tensorboard", type=str2bool,default=True)
parser.add_argument("--log_wandb", type=str2bool,default=False)
parser.add_argument("--setup_wandb", type=str2bool,default=False)

parser.add_argument("--optimizer", type=str,default='adabound',choices=['sgd', 'adam','adabound'])
parser.add_argument("--direct_map", type=str2bool,default=False)
parser.add_argument("--batch_size", type=int,default=32)
parser.add_argument("--epochs", type=int,default=50)
parser.add_argument("--weights", type=str,default='')
parser.add_argument("--learning_rate", type=float,default=0.001)

if __name__=='__main__':
    args = parser.parse_args()

    KoOCR=model.KoOCR(split_components=args.split_components,weight_path=args.weights,fc_link=args.fc_link,iterative_refinement=args.iterative_refinement,\
        network_type=args.network,image_size=args.image_size,direct_map=args.direct_map,refinement_t=args.refinement_t,data_augmentation=args.data_augmentation,
        adversarial_learning=args.adversarial_learning)
    KoOCR.train(epochs=args.epochs,lr=args.learning_rate,data_path=args.data_path,patch_size=args.patch_size,batch_size=args.batch_size,optimizer=args.optimizer,
        zip_weights=args.zip_weights,adversarial_ratio=args.adversarial_ratio,log_tensorboard=args.log_tensorboard,log_wandb=args.log_wandb,
        setup_wandb=args.setup_wandb,fit_discriminator=args.fit_discriminator,silent_mode=args.silent_mode)