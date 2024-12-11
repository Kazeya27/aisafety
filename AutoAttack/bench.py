import pdb
import torch
import torchvision.transforms as transforms


def save_image(img, path):
    toPIL = transforms.ToPILImage()
    img = img.cpu()
    img = img.squeeze(0)
    img = toPIL(img)
    img.save(path)

device = 'cuda:3'

# image range: [0, 1]

x_test = torch.load('images.pth', map_location=device)
y_test = torch.load('labels.pth', map_location=device)
# shape = ([50, 3, 32, 32]) ([50])

model_ids = ['Wang2023Better_WRN-70-16', 'Rebuffi2021Fixing_70_16_cutmix_extra', 'Gowal2020Uncovering_extra', 'Rebuffi2021Fixing_70_16_cutmix_ddpm', 'Augustin2020Adversarial_34_10_extra', 'Rade2021Helper_R18_ddpm']

from robustbench.utils import load_model

model = load_model(model_name=model_ids[0], dataset='cifar10', threat_model='L2')
model.to(torch.device(device))

from autoattack import AutoAttack
adversary = AutoAttack(model, norm='L2', eps=2.75, version='plus', device=device)


x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=64) # default 250
for i in range(500):
    save_image(x_adv[i], f'images/{i}.png')
