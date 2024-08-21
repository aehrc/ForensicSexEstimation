import torch
import torch.nn as nn
from networks.modules import ResidualConv
import torch.nn.functional as F

class ResNet_two_output(nn.Module):
    def __init__(self, channel=1, filters=[32, 64, 128, 256], num_metrics=7, num_classes=1):
        super(ResNet_two_output, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU(),
            nn.Conv3d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.output_layer1 = nn.Linear(filters[3], num_metrics)
        self.output_layer2 = nn.Linear(num_metrics, num_classes)

   
    def forward(self, x):
      
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)
        
        x5 = self.avgpool(x4)  
        x5 = x5.flatten(1)
        x6 = self.output_layer1(x5)
        
        output1 = F.relu(x6)
        
        x7 = self.output_layer2(x6)
        
        output2 = torch.sigmoid(x7)

        return output1, output2
    
class ResNet_test(nn.Module):
    def __init__(self, channel=1, filters=[32, 64, 128, 256], num_metrics=7, num_classes=1, which_output=1):
        super(ResNet_test, self).__init__()
        self.which_output = which_output
        
        self.input_layer = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU(),
            nn.Conv3d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.output_layer1 = nn.Linear(filters[3], num_metrics)
        self.output_layer2 = nn.Linear(num_metrics, num_classes)

   
    def forward(self, x):
      
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)
        
        x5 = self.avgpool(x4)  
        x5 = x5.flatten(1)
        x6 = self.output_layer1(x5)
        
        output1 = F.relu(x6)
        
        x7 = self.output_layer2(x6)
        
        output2 = torch.sigmoid(x7)
        
        if self.which_output!=0:

            return output2
        
        else:
            
            return output1

    
class ResNet_single_output(nn.Module):
    def __init__(self, channel=1, filters=[32, 64, 128, 256], num_metrics=7, num_classes=1):
        super(ResNet_single_output, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU(),
            nn.Conv3d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.output_layer1 = nn.Linear(filters[3], num_metrics)
        self.output_layer2 = nn.Linear(num_metrics, num_classes)

   
    def forward(self, x):
      
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)
        
        x5 = self.avgpool(x4)
        x5 = x5.flatten(1)
        x6 = self.output_layer1(x5)
        
        x7 = self.output_layer2(x6)
        
        output2 = torch.sigmoid(x7)

        return output2
    
class ResNet_auxiliary(nn.Module):
    def __init__(self, channel=1, filters=[32, 64, 128, 256], num_metrics=7, num_classes=1):
        super(ResNet_auxiliary, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU(),
            nn.Conv3d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.avgpool1 = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.output_layer1 = nn.Linear(filters[3], num_metrics)
        self.output_layer2 = nn.Linear(filters[3], num_classes)

   
    def forward(self, x):
      
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)
        
        x5_1 = self.avgpool1(x4)
        x5_1 = x5_1.flatten(1)
        output1 = self.output_layer1(x5_1)
        output1 = torch.relu(output1)
        
        x5_2 = self.avgpool2(x4)
        x5_2 = x5_2.flatten(1)
        output2 = self.output_layer2(x5_2)   
        output2 = torch.sigmoid(output2)

        return output1, output2
    
    
class ResNet_auxiliary_test(nn.Module):
    '''
    This is ResNet with two seperate outputs for gradcam generation
    '''
    def __init__(self, channel=1, filters=[32, 64, 128, 256], num_metrics=7, num_classes=1, which_output=1):
        super(ResNet_auxiliary_test, self).__init__()
        self.which_output = which_output
        
        self.input_layer = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU(),
            nn.Conv3d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.avgpool1 = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.relu = 
        
        self.output_layer1 = nn.Linear(filters[3], num_metrics)
        self.output_layer2 = nn.Linear(filters[3], num_classes)

   
    def forward(self, x):
      
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)
        
        x5_1 = self.avgpool1(x4)
        x5_1 = x5_1.flatten(1)
        output1 = self.output_layer1(x5_1)
        output1 = torch.relu(output1)
        
        x5_2 = self.avgpool2(x4) 
        x5_2 = x5_2.flatten(1)
        output2 = self.output_layer2(x5_2)     
        output2 = torch.sigmoid(output2)
        
        if self.which_output!=0:

            return output2
        
        else:
            
            return output1

class ResNet_just_score(nn.Module):
    def __init__(self, channel=1, filters=[32, 64, 128, 256], num_metrics=7):
        super(ResNet_just_score, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU(),
            nn.Conv3d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
               
        self.output_layer1 = nn.Linear(filters[3], num_metrics)
         
    def forward(self, x):
      
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)
        
        x5 = self.avgpool(x4)  
        x5 = x5.flatten(1)
        x6 = self.output_layer1(x5)
        
        output1 = F.relu(x6)
        
        return output1