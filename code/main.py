# ==== Argparse ====
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, required=True)
parser.add_argument('--seed', type=int, default=17)
parser.add_argument('--cuda', type=int, default=3)
parser.add_argument('--epochs', type=int, default=101)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--update_every', type=int, default=10)
parser.add_argument('--drop_ratio', type=float, default=0.2)
parser.add_argument('--random_mode', action='store_true', help="Enable random mode")
parser.add_argument('--lambda_value', type=float, default=0.0004)
parser.add_argument('--md_num_steps', type=int, default=400)
parser.add_argument('--md_lr', type=float, default=1)
parser.add_argument('--log_dir', type=str, default="../logs")
args = parser.parse_args()

# ==== Set seed ====
from utils.set_seed import set_seed
set_seed(args.seed)

# ==== Device ====
import torch
DEVICE = torch.device(f"cuda:{args.cuda}")
print(f"Using device: {torch.cuda.get_device_name(DEVICE)}")

# ==== Data ====
from data.data import get_cifar10_data
trainloader, testloader = get_cifar10_data(
    train_batch_size=args.train_batch_size,
    test_batch_size=args.test_batch_size,
)

# ==== Initialize ====   
from models.resnet import ResNet18
model = ResNet18().to(DEVICE)

from utils.impacts import initialize_impacts
impacts = initialize_impacts(model, ones=True, device=DEVICE)

from torch.optim import SGD
optimizer = SGD(model.parameters(), lr=args.lr)

from torch.nn import CrossEntropyLoss
criterion = CrossEntropyLoss()

from log.log import DataLogger
logger = DataLogger(
    name=args.run_name,
    hyperparams=vars(args),
    log_dir=args.log_dir,
)

from utils.running_average import RunningAverage
loss_running_average = RunningAverage(alpha=0.7)


# ==== Train ====

from train.step import step
from md.md import mirror_descent
from utils.accuracy import update_accuracy

from tqdm import tqdm

for epoch in tqdm(range(args.epochs), desc=f"Run {args.run_name}", ncols=100, leave=False):
    model.train()
    with tqdm(total=len(trainloader), desc=f"Epoch {epoch}/{args.epochs}", leave=False, ncols=100) as pbar:
        for X_batch, y_batch in trainloader:
            loss_value = step(
                X_batch=X_batch,
                y_batch=y_batch,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                impacts=impacts,
                drop_ratio=args.drop_ratio,
                random_mode=args.random_mode,
                drop_connections=True,
                device=DEVICE,
            )
            optimizer.step()

            loss_running_average.update(loss_value)
            # TODO: add here running average of train accuracy

            pbar.set_postfix({
                'Loss': f'{loss_running_average.average():.4f}'
            })
            pbar.update(1)
    
        logger.update(
            epoch=epoch, 
            loss=loss_running_average.average(),
            loss_std_dev=loss_running_average.std_dev(),
        )

        if epoch % args.update_every == 0:
            train_accuracy, test_accuracy = update_accuracy(model, trainloader, testloader, device=DEVICE)
            logger.update(
                epoch=epoch, 
                train_accuracy=train_accuracy, 
                test_accuracy=test_accuracy,
            )
            tqdm.write("Epoch {}/{} \t Loss: {:.4f} \t Train accuracy: {:.4f} \t Test accuracy: {:.4f}".format(epoch, args.epochs, loss_running_average.average(), train_accuracy, test_accuracy))

            _ = step(
                X_batch=X_batch,
                y_batch=y_batch,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                impacts=impacts,
                drop_ratio=1,
                random_mode=args.random_mode,
                drop_connections=False,
                device=DEVICE,
            )


            impacts = initialize_impacts(model, ones=False, device=DEVICE)

            for name in tqdm(impacts.keys(), desc="Mirror Descent", leave=False, ncols=100):
                impacts[name] = mirror_descent(
                    model=model,
                    data_loader=trainloader,
                    param_name=name,
                    impact=impacts[name],
                    model_lr=args.lr,
                    md_lr=args.md_lr,
                    md_lambda=args.lambda_value,
                    md_num_steps=args.md_num_steps,
                    criterion=criterion,
                    device=DEVICE,
                )
            logger.update(
                epoch=epoch, 
                impacts=impacts,
            )

# ==== Log ====
logger.dump()
