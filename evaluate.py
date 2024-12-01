import os
import torch
import argparse

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchmetrics.functional import structural_similarity_index_measure as ssim
from dataset import Cifar10Clean500
from models import CnnBench, ResnetBench
from PIL import Image
from pdb import set_trace
import numpy as np


class Evaluator:
    def __init__(self, exp_id, data, models, checkpoints, device):
        self.exp_id = exp_id
        self.data = data
        self.models = models
        self.checkpoints = checkpoints
        self.model_dict = {
            "cnn": CnnBench,
            "resnet": ResnetBench
        }
        self.device = device

    def load_model(self, model_name, ckp_path):
        model = self.model_dict[model_name](10, model_name)
        ckp_path = os.path.join("checkpoints", model_name, f"epoch{ckp_path}.pth")
        model.load(ckp_path)
        model = model.model()
        model.to(self.device)
        return model

    def compute_asr(self, model_name, ckp_path):
        attack_success = 0
        model = self.load_model(model_name, ckp_path)
        model.eval()
        for img, label in self.data:
            img = img.to(self.device)
            label = label.to(self.device)
            with torch.no_grad():
                pred = model(img).argmax(dim=1).item()
                label = label.argmax(dim=1).item()
            if pred != label:
                attack_success += 1
        return attack_success / len(self.data)


    def compute_ssim(self, attacked_path, original_path):
        ssim_values = []
        transform = transforms.Compose([transforms.ToTensor()])
        for i in range(500):
            attacked_image_path = os.path.join(attacked_path, f"{i}.png")
            attacked_image = Image.open(attacked_image_path).convert('RGB')
            attacked_image = transform(attacked_image)
            attacked_image = attacked_image.unsqueeze(0)
            original_image_path = os.path.join(original_path, f"{i}.png")
            original_image = Image.open(original_image_path).convert('RGB')
            original_image = transform(original_image)
            original_image = original_image.unsqueeze(0)
            ssim_value = ssim(attacked_image, original_image, data_range=1.0)
            ssim_values.append(ssim_value)
        return np.mean(ssim_values)


    def evaluate_objective_score(self):
        asrs = []
        for model_name, ckp_path in zip(self.models, self.checkpoints):
            asr = self.compute_asr(model_name, ckp_path)
            asrs.append(asr)
        asrs = np.mean(asrs)
        attacked_path = os.path.join("data", 'attack_{}'.format(self.exp_id))
        original_path = os.path.join("data", 'original', 'images')
        ssim_score = self.compute_ssim(attacked_path, original_path)

        # Compute final score
        objective_score = 100 * asrs * ssim_score
        return asrs, ssim_score, objective_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=int, default=0)
    parser.add_argument("--models", type=str, nargs="+", default=["cnn"])
    parser.add_argument("--checkpoints", type=str, nargs="+", default=["115"])
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    dataset = Cifar10Clean500(root="data", attack_id=args.exp_id)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    evaluator = Evaluator(args.exp_id, dataloader, args.models, args.checkpoints, device)

    asr, ssim, objective = evaluator.evaluate_objective_score()

    print(f"attack_{args.exp_id}: ASR = {asr}, SSIM = {ssim}, Objective = {objective}")

