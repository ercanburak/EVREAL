import torch
from torchvision import transforms
import glob
import os
from torch import nn
import pretrainedmodels
from torch.functional import F
from PIL import Image
import tqdm


class ResNet50(nn.Module):
    def __init__(self, pretrained):
        super(ResNet50, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained=None)

        self.l0 = nn.Linear(2048, 101)
        self.dropout = nn.Dropout2d(0.4)

    def forward(self, x):
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        l0 = self.l0(x)
        return l0


dir_path = os.path.dirname(__file__)
model_path = os.path.join(dir_path, 'caltech101_resnet50.pth')
classes_list_txt = os.path.join(dir_path, 'caltech101_classes.txt')

model_class_folders_glob_pattern = os.path.join('outputs',  'NCaltech101', '*', '*')
model_class_folders = glob.glob(model_class_folders_glob_pattern)
model_class_folders_dict = {}

for model_class_folder in model_class_folders:
    folder_path_parts = model_class_folder.split(os.sep)
    class_name = folder_path_parts[-1]
    model_name = folder_path_parts[-2]
    if model_name not in model_class_folders_dict:
        model_class_folders_dict[model_name] = []
    model_class_folders_dict[model_name].append(model_class_folder)

with open(classes_list_txt, 'r') as f:
    classes_list = f.read().splitlines()

model = ResNet50(pretrained=False)
model.load_state_dict(torch.load(model_path))
model = model.cuda()
model.eval()

data_transforms = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

model_acc_dict = {}
for model_name in model_class_folders_dict:
    print("Evaluating model: ", model_name)
    correct_preds = 0
    total_preds = 0
    for ncaltech_class_dir in tqdm.tqdm(model_class_folders_dict[model_name]):
        imgs_list = glob.glob(os.path.join(ncaltech_class_dir,  '*'))
        class_id = os.path.basename(ncaltech_class_dir)
        for img_path in imgs_list:
            img = Image.open(img_path).convert('RGB')
            data = data_transforms(img).unsqueeze(0).cuda()
            outputs = model(data)
            _, _id = torch.max(outputs.data, 1)
            _id = _id.cpu().tolist()[0]
            total_preds += 1
            if _id == classes_list.index(os.path.basename(ncaltech_class_dir)):
                correct_preds += 1

    model_acc = (100 * correct_preds / total_preds)
    model_acc_dict[model_name] = model_acc

for model_name in model_acc_dict:
    model_acc = model_acc_dict[model_name]
    print(f'{model_name} accuracy: {model_acc:.2f}%')
