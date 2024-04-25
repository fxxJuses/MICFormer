import os
import numpy as np
import seaborn as sns
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact
from tqdm import tqdm

def visualize_positional_encodings(pos_encoding, figsize=(15,10), title="Positional Encodings"):
    '''
    shows a visualization of a positional encoding in a tensor
    Parameters:
        pos_encoding (torch.tensor): positional encoding tensor (C, D, H, W)
        figsize (Tuple[int]): size of the figure
        title (str): title of the plot
    '''

    @interact
    def visualise_4d_image(d=(0,pos_encoding.shape[1]-1), h=(0,pos_encoding.shape[2]-1), w=(0,pos_encoding.shape[3]-1)): 
        #get image
        img = pos_encoding[:, d, h, w].reshape(1, -1)

        #plot slice
        plt.figure(figsize=figsize)
        plt.imshow(img, aspect=pos_encoding.shape[0]/6)
        plt.xlabel('channels')
        plt.clim(-1,1)

        #show plot
        plt.show()

def visualize_attention(attention_weights_avg, figsize=(15,15), fraction=0.02, pad=0.05, title='Average Attention Weights'):
    '''
    shows a visualization of a positional encoding in a tensor
    Parameters:
        pos_encoding (torch.tensor): positional encoding tensor (C, D, H, W)
        figsize (Tuple[int]): size of the figure
        fraction (float): colorbar parameter
        pad (float): colorbar padding from the figure
        title (str): title of the plot
    '''

    @interact
    def visualise_4d_image(
        image=(0,attention_weights_avg.shape[0]-1),
        depth_decoder=(0,attention_weights_avg.shape[-3]-1),
        height_decoder=(0,attention_weights_avg.shape[-2]-1),
        width_decoder=(0,attention_weights_avg.shape[-1]-1),
        depth_skip=(0,attention_weights_avg.shape[1]-1),
        ): 
        #get image
        img = attention_weights_avg[image, depth_skip, :, :, depth_decoder, height_decoder, width_decoder]

        #plot slice
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(img)
        fig.colorbar(im, fraction=fraction, pad=pad)
        plt.title(title)

        #show plot
        plt.show()


def visualize_dataloaders_overlap(dataloader, alpha=0.3, figsize=(8, 8)):
    '''
    shows a visualization of a slice of an image and its label from a dataloader

    Args:
        dataloader: torch.utils.data.DataLoader
            dataloader to visualize
        alpha: float
            % of  oppacity for the label
        figsize: tuple
            size of the figure
    '''

    @interact
    def plot_slice(image=(1, len(dataloader.dataset)),
                   slice=(1, dataloader.dataset[0][0][0].numpy().shape[0]),
                   ):
        # get image and label
        im = dataloader.dataset[image][0][0].numpy()
        label = dataloader.dataset[image][1][0].numpy()
        if slice > im.shape[0] - 1:
            slice = im.shape[0] - 1

        # plot slice
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(im[slice], cmap="gray")
        ax.imshow(label[slice], cmap="jet", alpha=alpha)
        ax.set_title(f"Image and label")

        # show plot
        plt.show()


def visualize_infered_labels(test_dataloader, labels_path, alpha=0.3, figsize=(8, 8)):
    '''
    shows a visualization of a slice of an image and its infered label from the test dataloader

    Args:
        test_dataloader: torch.utils.data.DataLoader
            test_dataloader to visualize images from
        labels_path: str
            path to the infered labels
        alpha: float
            % of  oppacity for the label
        figsize: tuple
            size of the figure
    '''
    filenames = sorted(os.listdir(labels_path))
    max_depth = 0
    # determine max depth of labels
    for filename in tqdm(filenames):
        if filename.endswith(".nii.gz"):
            label = nib.load(os.path.join(labels_path, filename)).get_fdata()
            if label.shape[-1] > max_depth:
                max_depth = label.shape[-1]
            
    @interact
    def plot_slice(image=(1, len(test_dataloader.dataset)),
                   slice=(1, max_depth),
                   ):
        # get image and label
        im = test_dataloader.dataset[image-1][0].numpy()
        label = nib.load(os.path.join(labels_path, filenames[image-1])).get_fdata()
        label = np.transpose(label, (2, 0, 1))
        print("image shape:", im.shape)
        print("label shape:", label.shape)
        if slice > im.shape[0] - 1:
            slice = im.shape[0] - 1

        # plot slice
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(im[slice], cmap="gray")
        ax.imshow(label[slice], cmap="jet", alpha=alpha)
        ax.set_title(f"Image and label")

        # show plot
        plt.show()


def plot_learning_curves(dfs, metric, model_names, y_axis, figsize=(10, 5), show=False, save_path=None):
    """
    Plot the learning curves of a model for k-fold cross-validation training.

    Args:
        dfs list(list(pandas.DataFrame)):
            The logged data as a list of list of pandas DataFrame.
            i.e. [model1[fold1_df, fold2_df, ...], model2[fold1_df, fold2_df, ...], ...]
        metric (str):
            The metric to plot.
        model_names list(str):
            List of names of the models used for training.
        y_axis (str):
            The y-axis name on the plot.
        figsize (tuple):
            The figure size.
        show (bool):
            Whether to show the plot.
        save_path (str):
            The path to save the plot.
            If None, the plot is not saved.

    Returns:
        matplotlib.pyplot:
            The plot object.
    """
    assert isinstance(dfs, list)
    assert isinstance(dfs[0], list)
    assert isinstance(model_names, list)

    colors = ["#377eb8", "#ff7f00", "#4daf4a", "#e41c1c", ]
    linestyles = ["solid", "dashed", "dashdot", "dotted", "(5, (10, 3))"]
    plt.figure(figsize=figsize)
    for i, df in enumerate(dfs):
        mean = np.mean([fold_df[metric] for fold_df in df], axis=0)
        std = np.std([fold_df[metric] for fold_df in df], axis=0)
        plt.plot(mean, label=model_names[i], color=colors[i], linestyle=linestyles[i])
        plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.10, color=colors[i])
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(y_axis)
    plt.title(f"{y_axis} on {len(df)} folds")
    # plt.rc('font', size=14)
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    
    return plt


def plot_seaborn_fold_wise(plot_type, df_baseline, dfs_compared, metric, model_names, y_axis, figsize=(10, 5), show=False, save_path=None):
    """
    Plot the box plots of a model for k-fold cross-validation training.

    Args:
        plot_type (str):
            The type of plot to use. Can be "box", "violin" or "bar".
        df_baseline (pandas.DataFrame):
            The logged data of the baseline model as a pandas DataFrame.
        dfs_compared list(list(pandas.DataFrame)):
            The logged data of the compared models as a list of list of pandas DataFrame.
            i.e. [model1[fold1_df, fold2_df, ...], model2[fold1_df, fold2_df, ...], ...]
        dfs list(list(pandas.DataFrame)):
        metric (str):
            The metric to plot.
        model_names list(str):
            List of names of the models used for training.
        y_axis (str):
            The y-axis name on the plot.
        figsize (tuple):
            The figure size.
        show (bool):
            Whether to show the plot.
        save_path (str):
            The path to save the plot.
            If None, the plot is not saved.

    Returns:
        seaborn.boxplot:
            The boxplot object.
    """
    assert isinstance(dfs_compared[0], list)
    assert isinstance(model_names, list)

    max_metric_list_baseline = []
    for baseline_fold in df_baseline:
        max_metric_list_baseline.append(np.max(baseline_fold[metric].values))

    plt.figure(figsize=figsize)
    plt.axhline(y=0, color='black', linestyle='--')
    dic = {}
    for i, model_df in enumerate(dfs_compared):
        max_metric_list_compared = []
        for fold_df in model_df:
            max_metric_list_compared.append(np.max(fold_df[metric].values))
        difference_list = [max_metric_list_compared[i] - max_metric_list_baseline[i] for i in range(len(max_metric_list_baseline))]
        dic = dic | {model_names[i]: difference_list}
    df = pd.DataFrame(dic)
    if plot_type == "box":
        sns.boxplot(data=df)
    elif plot_type == "violin":
        sns.violinplot(data=df)
    elif plot_type == "bar":
        sns.barplot(data=df)
    plt.ylabel(y_axis)
    plt.title(f"Fold-wise difference with UNet baseline on {y_axis}")
    # plt.rc('font', size=14)
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    
    return plt


def plot_scatter_relations(df, variable, metric, xlabel, ylabel, figsize=(10, 5), show=False, save_path=None):
    """
    Plot the scatter plot of a variable against a metric.

    Args:
        df (pandas.DataFrame):
            The dataframe containing the metric and the variable.
        variable (str):
            The variable to plot.
        metric (str):
            The metric to plot.
        figsize (tuple):
            The figure size.
        show (bool):
            Whether to show the plot.
        save_path (str):
            The path to save the plot.
            If None, the plot is not saved.

    Returns:
        matplotlib.pyplot:
            The plot object.
    """
    assert len(df[metric].values) == len(df[variable].values)

    markers_list = ["x", "s", "v", "*", "o", "+", "D", "p", "2", "<", ">"]
    plt.figure(figsize=figsize)
    # df = df.sort_values(by=[variable])
    for i, txt in enumerate(df.index):
        plt.scatter(df[variable].values[i], df[metric].values[i], marker=markers_list[i], label=txt, linewidths=2)
        # plt.scatter(df[variable].values[i], df[metric].values[i], marker=markers_list[i])
        # plt.annotate(txt, (df[variable].values[i], df[metric].values[i]))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Relations between {xlabel} and {ylabel}")
    plt.xscale('log')
    # plt.rc('font', size=14)
    plt.legend()
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    
    return plt