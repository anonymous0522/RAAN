import torch
import torch.nn as nn
import torch.nn.functional as F
from det3d.core.utils.center_utils import _transpose_and_gather_feat

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
      only regression gt bbox centers.
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    # output is pred bs * reg type * w * h
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.float().unsqueeze(2) 
    # only gt center pixel to calculate loss.
    loss = F.l1_loss(pred*mask, target*mask, reduction='none')
    loss = loss / (mask.sum() + 1e-4)
    loss = loss.transpose(2 ,0).sum(dim=2).sum(dim=1)
    return loss

class FastFocalLoss(nn.Module):
  '''
  Reimplemented focal loss, exactly the same as the CornerNet version.
  Faster and costs much less memory.
  '''
  def __init__(self):
    super(FastFocalLoss, self).__init__()

  def forward(self, out, target, ind, mask, cat):
    '''
    Arguments:
      out, target: B x C x H x W target is heatmap
      ind, mask: B x M
      cat (category id for peaks): B x M
      Postive sample are same with gt number.
    '''
    mask = mask.float()
    gt = torch.pow(1 - target, 4) # give a weight for neg loss, if heatmap val higher, weight smaller.
    # gt when 1.0 in target, gt will be 0, other pixel are calculated. so we only need to make up 1.0 pos loss in target.
    neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt
    neg_loss = neg_loss.sum()
    pos_pred_pix = _transpose_and_gather_feat(out, ind) # B x M x C
    pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M gather by cls.
    num_pos = mask.sum()
    # positive sample are only focus on center point per gt
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
               mask.unsqueeze(2)
    pos_loss = pos_loss.sum()
    if num_pos == 0:
      return - neg_loss
    return - (pos_loss + neg_loss) / num_pos

class FocalLoss(nn.Module):
  '''
  Implemented focal loss.
  https://medium.com/visionwizard/understanding-focal-loss-a-quick-read-b914422913e7
  '''
  def __init__(self, num_classes, gamma=2.0, alpha=0.25):
    super(FocalLoss, self).__init__()
    self._alpha = alpha
    self._gamma = gamma
    self._num_classes = num_classes

  def forward(self, out, target, ind, mask):
    '''
    Arguments:
      out : B x C x H x W target is score map
      target, ind, mask: B x M, which target starts from 1 (i.e. 1,2,3...)
      Postive sample are same with gt number. only use focal loss on those positive samples.
    '''
    one_hot_targets = F.one_hot(
        target, num_classes=(self._num_classes + 1)) # N * 500 * num_classes+1
    one_hot_targets = one_hot_targets[..., 1:].float() # remove background 0.
    mask = mask.float()
    num_pos = mask.sum()
    pos_pred_pix = _transpose_and_gather_feat(out, ind) # B x M x C
    pt = torch.abs(one_hot_targets - pos_pred_pix.sigmoid()) # equal to 1 - prob
    loss = F.binary_cross_entropy_with_logits( # include Sigmoid layer
        pos_pred_pix, one_hot_targets, reduction='none') * pt.pow(self._gamma)
    # where pos, alpha, where neg, 1 - alpha
    '''
    alpha_weight_factor = torch.where(one_hot_targets == 1, 
        torch.tensor(self._alpha).type_as(pos_pred_pix), 
        torch.tensor(1 - self._alpha).type_as(pos_pred_pix))
    '''
    # to enlarge loss.
    loss = loss * mask.unsqueeze(2) # alpha_weight_factor * (loss * mask.unsqueeze(2))
    return loss.sum() / num_pos
