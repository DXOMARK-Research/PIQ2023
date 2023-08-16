import torch
import torch.nn as nn
from torch.nn.functional import one_hot

from .sem_hyperiqa_util import HyperNet, RescaleNet, TargetNet, SceneClassNet
from .arch_util import load_pretrained_network


defaultHyperNetWeights = {
    # Only for original hyperIQA model with input size 224x224
    'koniq': None #link: https://drive.google.com/file/d/1OOUmnbvpGea0LIGpIWEbOyxfWx6UCiiE/view
}

defaultWeights = {}


class SemHyperIQA(nn.Module):
    def __init__(self, patchSize,  
                 hyperNetPretrained=None, 
                 pretrained=None,
                 classify=None, 
                 rescale=None,
                  **kwargs):
        
        super().__init__()
        patchRate = patchSize // 224
        self.classify = classify
        self.rescale = rescale
        self.classFeaturesOut = None
        self.nbPatchesIn = None
        self.classKey = kwargs.get('classKey', 'class')
        self.qualityKey = kwargs.get('qualityKey', 'quality')
        if self.classify is not None:
            self.classFeaturesOut = self.classify.get('numClasses', None)
            self.nbPatchesIn = self.classify.get('nbPatchesIn', 1) # Assume one patch if we do not want to concatenate patches
        self.hyperNet = HyperNet(16, 
                                 112 * patchRate, 
                                 224 * patchRate, 
                                 112 * patchRate, 
                                 56 * patchRate, 
                                 28 * patchRate, 
                                 14 * patchRate, 
                                 7 * patchRate, 
                                 patchRate, 
                                 classFeaturesOut=self.classFeaturesOut)#.cuda()
        
        if hyperNetPretrained is not None:
            load_pretrained_network(self.hyperNet, defaultHyperNetWeights.get(hyperNetPretrained, hyperNetPretrained))
        if pretrained is not None:
            load_pretrained_network(self, defaultWeights.get(pretrained, pretrained))
        
        if self.classify is not None:
            self.sceneClassNet = SceneClassNet(featureInSize=self.nbPatchesIn * 112 * patchRate * self.classFeaturesOut,
                                               **self.classify)
            self.sceneclassnet_params = self.sceneClassNet.parameters()
        if self.rescale is not None:
            if 'featureInSize' in self.rescale:
                self.classFeedback = False
            else:
                self.classFeedback = True
                self.rescale.update({'featureInSize': self.classify.get('numClasses')}) # intentionally throw error if rescale is defined with infeatures and no classification parameter
            
            self.rescaleNet = RescaleNet(**self.rescale)
            self.rescalenet_params = self.rescaleNet.parameters()

        backbone_params = list(map(id, self.hyperNet.res.parameters()))
        self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.hyperNet.parameters())
        self.resnet_params = filter(lambda p: id(p) in backbone_params, self.hyperNet.parameters())

    def forward(self, x, index=None, *args):
        # Generate weights for target network
        output = self.hyperNet(x)

        # Check if hyperNet returns hnFeatures
        if isinstance(output, tuple):
            paras, hnFeatures = output
            hnFeatures = self._consolidate_patches(hnFeatures, self.nbPatchesIn)
        else:
            paras = output

        if isinstance(paras, list):
            paras = self._stack_dicts(paras)

        # Building target network
        modelTarget = TargetNet(paras)
        for param in modelTarget.parameters():
            param.requires_grad = False
            
        # Quality score prediction
        inputTargetNet = paras['target_in_vec']
        predictionsQuality = modelTarget(inputTargetNet)

        predScene = None

        if hasattr(self, 'sceneClassNet') and isinstance(output, tuple):
            predScene = self.sceneClassNet(hnFeatures)
            predictionsQuality = predictionsQuality.reshape(self.nbPatchesIn, -1).mean(dim=0)

        if hasattr(self, 'rescaleNet') and hasattr(self, 'classFeedback'):
            if not self.classFeedback and index is not None:
                index_ = one_hot(index, num_classes=self.rescale['featureInSize']).to(torch.float32)
                scoreWeights = self.rescaleNet(index_)
            elif hasattr(self, 'sceneClassNet') and isinstance(output, tuple):
                scoreWeights = self.rescaleNet(predScene.softmax(dim=1))
            else:
                raise ValueError("Class feedback needs class prediction, which is not defined in this configuration")

            # FIXME: Fit with any polynomial degree instead of only manual linear fit.
            # Re-scale the score prediction with alpha/beta
            predictionsQuality = scoreWeights[:,0] * predictionsQuality + scoreWeights[:,1]
        
        if predScene is not None:
            # FIXME: In case of rescaling return the quality before rescaling too
            return {self.qualityKey: predictionsQuality.unsqueeze(1),
                    self.classKey: predScene}

        return predictionsQuality
    
    @staticmethod    
    def _consolidate_patches(hnFeatures, patches_per_image):
        # Check if hnFeatures is a list
        if isinstance(hnFeatures, list):
            # If first element of the list is a tensor and is 2D, stack along 1st dimension
            if hnFeatures[0].dim() >= 2:
                hnFeatures = torch.cat(hnFeatures, dim=0)
            # If first element of the list is a tensor and is 1D, convert to 2D and stack along 1st dimension
            elif hnFeatures[0].dim() == 1:
                hnFeatures = torch.stack(hnFeatures, dim=0)

        # Ensure that the total number of features is a multiple of patches_per_image
        if hnFeatures.shape[0] % patches_per_image != 0:
            raise ValueError("Total number of features is not a multiple of patches_per_image")

        # Reshape the tensor
        consolidated_features = hnFeatures.reshape(-1, hnFeatures.shape[1] * patches_per_image)

        return consolidated_features
    
    @staticmethod
    def _stack_dicts(dict_list):
        # Ensure dict_list is not empty
        if not dict_list:
            return {}

        # Create a new dictionary where each key is a stack of the corresponding values from the dictionaries in dict_list
        stacked_dict = {key: torch.stack([d[key] for d in dict_list], dim=0) for key in dict_list[0].keys()}

        return stacked_dict