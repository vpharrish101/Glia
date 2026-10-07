import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional,Tuple


def overlay_mask_on_image(image: np.ndarray,
                          mask: np.ndarray,
                          color: Tuple[int,int,int]=(0,255,0),
                          alpha: float=0.4,)->np.ndarray:
    """
    Overlay a binary mask on a grayscale or RGB image.
    For 2.5D multi-channel images,pass the center channel as a grayscale array.

    Returns:
        RGB overlay image,shape (H,W,3),uint8.
    """

    if image.ndim==2:
        image_rgb=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    else:
        image_rgb=image.copy()

    mask_binary=(mask>127 if mask.max()>1 else mask>0.5).astype(np.uint8)
    overlay=image_rgb.copy()
    overlay[mask_binary==1]=color

    return cv2.addWeighted(image_rgb,1-alpha,overlay,alpha,0)


def save_overlay(image: np.ndarray,
                 mask: np.ndarray,
                 save_path: str,
                 color: Tuple[int,int,int]=(0,255,0),
                 alpha: float=0.4,)->None:
    """
    Create and save an overlay image.
    """

    overlay=overlay_mask_on_image(image,mask,color,alpha)
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    cv2.imwrite(save_path,cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))


def plot_prediction(image: np.ndarray,
                    gt_mask: Optional[np.ndarray],
                    pred_mask: np.ndarray,
                    save_path: Optional[str]=None,
                    title: str="",)->None:
    """
    Plot image,ground truth,prediction,and overlay side by side.
    """

    ncols=4 if gt_mask is not None else 3
    fig,axes=plt.subplots(1,ncols,figsize=(5*ncols,5))

    axes[0].imshow(image,cmap="gray")
    axes[0].set_title("FLAIR Input")
    axes[0].axis("off")

    col=1
    if gt_mask is not None:
        axes[col].imshow(gt_mask,cmap="gray")
        axes[col].set_title("Ground Truth")
        axes[col].axis("off")
        col+=1

    axes[col].imshow(pred_mask,cmap="gray")
    axes[col].set_title("Prediction")
    axes[col].axis("off")
    col+=1

    overlay=overlay_mask_on_image(image,pred_mask)
    axes[col].imshow(overlay)
    axes[col].set_title("Overlay")
    axes[col].axis("off")

    if title:
        fig.suptitle(title,fontsize=14,fontweight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        plt.savefig(save_path,dpi=150,bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_comparison_grid(images: list,
                           masks_gt: list,
                           masks_pred: list,
                           save_path: str,
                           max_samples: int=8,)->None:
    """
    Create a grid comparing multiple predictions.
    """

    n=min(len(images),max_samples)
    fig,axes=plt.subplots(n,4,figsize=(20,5*n))

    if n==1:
        axes=axes[np.newaxis,:]

    for i in range(n):
        for ax,data,title in zip(axes[i],
                                [images[i],
                                 masks_gt[i],
                                 masks_pred[i],
                                overlay_mask_on_image(images[i],masks_pred[i])],
                                ["FLAIR","Ground Truth","Prediction","Overlay"],):
            
            ax.imshow(data,cmap="gray" if data.ndim==2 else None)
            ax.set_title(title)
            ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    plt.savefig(save_path,dpi=150,bbox_inches="tight")
    plt.close()
