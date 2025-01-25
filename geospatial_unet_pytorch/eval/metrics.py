from typing import Optional, List

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from segmentation_models_pytorch.losses.constants import *
from segmentation_models_pytorch.losses._functional import soft_dice_score, to_tensor

class MaskedDice(torch.nn.Module):
    '''
    Compute the average dice loss per valid pixel. 
    Most code copied from segmentation models torch. We tried to use their DiceLoss fn but couldn_t
    because it would have reduction=mean by defeault and we need reduction=None to weight loss equally
    per valid pixel 
    Src: https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/losses/dice.py

    Args:
        smooth: If we have a lot of images with almost no positive class, increasing this can help 
            the training stability. See online for more.
        ignore_index: Value that will not occur in prediction or target. We are using this value
         to keep track of nan values that should be ignored in the loss computation
        reduction: loss reduction to apply to the batch of loss values either 'none', 'mean', 'sum'.
        see smp documentations for other arguments
    '''
    def __init__(self, 
                mode: str,
                classes: Optional[List[int]] = None,
                log_loss: bool = False,
                from_logits: bool = True,
                smooth: float = 0.0,
                ignore_index: Optional[int] = -999,
                eps: float = 1e-7,
                reduction: str = 'mean'
                ):
        super(MaskedDice, self).__init__()
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, (
                "Masking classes is not supported with mode=binary"
            )
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target, mask):
        """
        Input:
            input: see MaskedLoss
            target: see MaskedLoss
            mask torch.tensor(): 1. for invalid pixels and 0. for valid pixels.
        Returns:
            loss: torch.Tensor(batch_size)
        """
        # Rename from pytorch standards to smp standards
        nan_mask = mask
        y_pred = input.clone() # cloning here bc the loss modifies the values
        y_true = target.clone()

        # Set pixels in the y_prediction and y_true to no_data_value whereever there's no data
        y_pred[nan_mask == 1.] = self.ignore_index
        y_true[nan_mask == 1.] = self.ignore_index

        # The code starting here is copied from smp.
        # """"""""""""""""""""""""""""""""
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        # 
        dims = (-1)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                y_true = F.one_hot(
                    (y_true * mask).to(torch.long), num_classes
                )  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # N, C, H*W
            else:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # N, C, H*W

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = soft_dice_score(
            y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims
        ) # shape: (batch_size, channel_idx)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        if self.classes is not None:
            loss = loss[:,self.classes]
        # """"""""""""""""""""""""""""""""

        # Compute diceLoss. This dice loss will compute the average loss per tile.
        loss_per_tile = loss

        # Reweight with number of valid pixels per tile to compute the avg dice loss per valid px
        num_valid_pixels = torch.count_nonzero(1-nan_mask,dim=(-2,-1)) # shape: (batch_size, channel_idx)
        num_total_pixels = nan_mask.shape[-2] * nan_mask.shape[-1]
        
        loss_per_pixel = loss_per_tile / num_total_pixels * num_valid_pixels # shape: (batch_size, channel_idx)

        # Reduce over batch and channel dimension at the end:
        if self.reduction == 'mean':
            final_loss = loss_per_pixel.mean()
        else:
            final_loss = loss_per_pixel

        return final_loss

def lookup_loss_function(loss_name: str):
    """
    Returns loss function given a string
    """
    if loss_name == 'l1':
        return nn.L1Loss
    elif loss_name == 'bceloss':
        return nn.BCELoss
    elif loss_name == 'nllloss':
        return nn.NLLLoss
    elif loss_name == 'bcewithlogits':
        return nn.BCEWithLogitsLoss
    elif loss_name == 'dice':
        return smp.losses.DiceLoss
    else:
        print(f'Warning: Did not find requested loss fn called {loss_name}. using MSE')
        return nn.MSELoss

class MaskedLoss(torch.nn.Module):
    '''
    A loss function wrapper that calculates the loss on unmasked values only.
    This module wraps around standard loss functions (such as L1 or L2) and applies them to the model predictions and
    ground truth targets, while excluding masked values from the loss calculation.
    Args:
        loss_name (str): loss functions to be used either 'l1', 'l2'
        reduction (str): loss reduction to apply to the batch of loss values either 'none', 'mean', 'sum'.
        **kwargs: miscellaneous arguments that might be required by the loss function
    '''
    def __init__(self, loss_name: str = 'l1',
                 reduction: str = 'mean',
                 **kwargs):
        super(MaskedLoss, self).__init__()
        Criterion = lookup_loss_function(loss_name)
        self.criterion = Criterion(reduction = 'none', **kwargs)
        self.reduction = reduction
    def forward(self, input, target, mask):
        """
        Input:
            input torch.Tensor(batch_size, out_channels, height, width): Model prediction
            target torch.Tensor(batch_size, out_channels, height, width): Ground-truth target
            mask torch.Tensor(batch_size, out_channels, height, width, dtype=float32):
                with 1. for masked/nan/invalid values and 0. for unmasked values
        Returns:
            loss: torch.Tensor(batch_size)
        """
        loss = self.criterion(input, target) # dims: (batch_size, out_ch, height, width)
        # First calculate the loss over all, but the batch dimension.
        all_dims_but_first = tuple(range(1, len(loss.shape)))
        # Cumulative loss over all filled, unmasked pixels.
        loss = (loss * (1.-mask)).sum(dim=all_dims_but_first) # dims: (batch_size)
        # Total number of valid, unmasked pixels
        num_valid_pixels = (1.-mask).sum(dim=all_dims_but_first) # dims: (batch_size)
        # Set to one, in case all pixels in an image are masked, to avoid division by zero
        num_valid_pixels[num_valid_pixels==0] = 1.
        # Calculate average loss per valid pixel
        loss_pixelwise = loss / num_valid_pixels # dims: (batch_size)
        # Optionally reduce loss over batch dimension
        if self.reduction == 'mean':
            loss_pixelwise = loss_pixelwise.mean() # dims: ()
        elif self.reduction == 'sum':
            loss_pixelwise = loss_pixelwise.sum()
        return loss_pixelwise