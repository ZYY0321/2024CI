import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
from models.vehicle_model import VehicleDetectionModel
import torch.nn.functional as F
import os

class CarPredictor:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型结构
        self.model = VehicleDetectionModel()
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        
        # 加载模型权重
        model_dict = self.model.state_dict()
        if 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
        else:
            pretrained_dict = checkpoint
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict, strict=False)
        
        self.model.to(self.device)
        self.model.eval()
        
        # 从训练数据目录获取类别信息
        train_dir = '/data/jj/datasets/scripts/Temporarily_useless/carid/train'
        self.idx_to_class = {}
        class_names = sorted([d for d in os.listdir(train_dir) 
                            if os.path.isdir(os.path.join(train_dir, d))])
        for idx, class_name in enumerate(class_names):
            self.idx_to_class[idx] = class_name
        
        # 设置图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path, top_k=5):
        # 加载并预处理图像
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        image = image.to(self.device)
        
        # 进行预测
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = F.softmax(outputs, dim=1)
            top_prob, top_class = torch.topk(probabilities, top_k)
            
        # 获取预测结果
        predictions = []
        for i in range(top_k):
            class_idx = top_class[0][i].item()
            prob = top_prob[0][i].item()
            class_name = self.idx_to_class[class_idx]
            predictions.append((class_name, prob))
            
        return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--model', type=str, default='/data/jj/datasets/scripts/Temporarily_useless/carid/checkpoints/best_model_20241117_233010_acc69.48.pth', help='Path to the model file')
    args = parser.parse_args()
    
    predictor = CarPredictor(args.model)
    predictions = predictor.predict(args.image)
    
    print(f"\n预测结果 - 图片: {args.image}\n")
    print("Top 5 预测结果:")
    print("-" * 50)
    for i, (class_name, probability) in enumerate(predictions, 1):
        print(f"{i}. {class_name}")
        print(f"   置信度: {probability:.2%}")
        print("-" * 50)

if __name__ == '__main__':
    main() 