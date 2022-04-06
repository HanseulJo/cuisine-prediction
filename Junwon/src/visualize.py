import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import wandb

def visualize_confusion_matrix(model, dataloader, class_names, args, device='cpu', data_dir=None):
    was_training = model.training
    model.eval()

    labels_stack = None
    preds_stack = None

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            if labels_stack is not None:
                labels_stack = np.hstack([labels_stack, labels.clone().detach().cpu().numpy()])
            else:
                labels_stack = labels.clone().detach().cpu().numpy()
            if preds_stack is not None:
                preds_stack = np.hstack([preds_stack, preds.clone().detach().cpu().numpy()])
            else:
                preds_stack = preds.clone().detach().cpu().numpy()

        model.train(mode=was_training)

        confusion_matrix = confusion_matrix(labels_stack, preds_stack)
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                      display_labels=class_names)
        disp.plot()
        plt.xticks(rotation=90)

        fname = ['confusionMtrx',
                 'batch', str(args.batch_size), 'n_epochs', str(args.n_epochs),
                 'lr', str(args.lr), 'step_size', str(args.step_size), 'seed', str(args.seed)]
        fname = '_'.join(fname) + '.png'

        if not os.path.isdir('examples/'):
            os.mkdir('examples/')
        plt.savefig(os.path.join('examples/', fname))
        if args.wandb_log:
            wandb.log({"confusion_matrix": plt.gcf()})