import math
import torch
import torch.nn as nn
import torch.nn.functional


class NeRF(nn.Module):
    def __init__(self, numLayerPosition=8, numLayerColor=1,
                 positionHiddenDim=256, colorHiddenDim=128,
                 positionFeatureDim=256,
                 requiresPositionEmbedding=(5,)):
        super(NeRF, self).__init__()

        positionEmbeddingDim = 60
        directionEmbeddingDim = 60

        self.positionEmbedding = None
        self.directionEmbedding = None

        # position network
        positionLayers = []
        for i in range(numLayerPosition):
            if i == 0:
                inputDimension = positionEmbeddingDim
                outputDimension = positionHiddenDim
                requireAuxPositionEmbedding = False
            elif i in requiresPositionEmbedding:
                inputDimension = positionEmbeddingDim + positionHiddenDim
                outputDimension = positionHiddenDim
                requireAuxPositionEmbedding = True
            else:
                inputDimension = positionHiddenDim
                outputDimension = positionHiddenDim
                requireAuxPositionEmbedding = False

            layer = nn.Linear(inputDimension, outputDimension)
            layer.requireAuxPositionEmbedding = requireAuxPositionEmbedding
            positionLayers.append(layer)

        self.positionLayers = nn.ModuleList(positionLayers)

        # density layer
        self.densityLayer = nn.Linear(positionHiddenDim, 1)

        # position feature layer
        self.positionFeatureLayer = nn.Linear(positionHiddenDim, positionFeatureDim)

        # color network
        colorLayers = []
        for i in range(numLayerColor):
            if i == 0:
                inputDimension = directionEmbeddingDim + positionFeatureDim
                outputDimension = colorHiddenDim
            else:
                inputDimension = colorHiddenDim
                outputDimension = colorHiddenDim

            layer = nn.Linear(inputDimension, outputDimension)
            colorLayers.append(layer)

        self.colorLayers = nn.ModuleList(colorLayers)

        # RGB layer
        self.rgbLayer = nn.Linear(colorHiddenDim, 3)
        return

    def forward(self, x, d):
        x = self.positionEmbedding(x)
        d = self.directionEmbedding(d)

        # position network forward pass
        y = x
        for layer in self.positionLayers:
            if layer.requireAuxPositionEmbedding:
                y = layer(torch.cat([x, y], dim=1))
            else:
                y = layer(y)
            y = nn.functional.relu(y)

        # density layer forward pass
        sigma = self.densityLayer(y)

        # compute position feature
        positionFeature = self.positionFeatureLayer(y)

        # color network forward pass
        c = torch.cat([positionFeature, d], dim=1)
        for layer in self.colorLayers:
            c = nn.functional.relu(layer(c))

        # compute rgb for final output
        rgb = self.rgbLayer(c)

        out = torch.cat([rgb, sigma], dim=-1)
        return out
