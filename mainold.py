import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import argparse
from models.vehicle_model import VehicleDetectionModel
from data.stanford_cars import StanfordCarsDataset
from utils.trainer import Trainer
import matplotlib.pyplot as plt
import json
from datetime import datetime
import logging
from tqdm import tqdm
import math
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 设置 PyTorch 镜像源
os.environ['TORCH_HOME'] = '/data/jj/.cache/torch/hub/checkpoints'
os.environ['TORCH_MODEL_ZOO'] = 'https://mirrors.tuna.tsinghua.edu.cn/pytorch/whl/torch_stable.html'

def setup_distributed():
    # 初始化分布式训练
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()

def setup_logging(local_rank):
    """设置日志"""
    # 创建日志目录
    log_dir = 'logs'
    if local_rank == 0 and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}_rank{local_rank}.log')
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return timestamp

def plot_metrics(history, timestamp):
    """绘制训练指标图"""
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['test_losses'], label='Test Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accs'], label='Train Accuracy')
    plt.plot(history['test_accs'], label='Test Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/training_curves_{timestamp}.png')
    plt.close()

def save_training_history(history, timestamp):
    """保存训练历史"""
    if not os.path.exists('history'):
        os.makedirs('history')
    
    with open(f'history/training_history_{timestamp}.json', 'w') as f:
        json.dump(history, f, indent=4)

def update_history(history, train_loss, train_acc, test_loss, test_acc):
    """更新训练历史"""
    history['train_losses'].append(train_loss)
    history['train_accs'].append(train_acc)
    history['test_losses'].append(test_loss)
    history['test_accs'].append(test_acc)

def save_checkpoint(model, optimizer, epoch, acc, timestamp, is_best=False):
    """保存检查点"""
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 如果是最佳模型，删除之前的最佳模型
    if is_best:
        for f in os.listdir(checkpoint_dir):
            if f.startswith('best_model_'):
                os.remove(os.path.join(checkpoint_dir, f))
        
        checkpoint_path = os.path.join(checkpoint_dir, f'best_model_{timestamp}_acc{acc:.2f}.pth')
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f'model_{timestamp}_epoch{epoch}_acc{acc:.2f}.pth')
    
    checkpoint = {
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'accuracy': acc,
        'timestamp': timestamp
    }
    torch.save(checkpoint, checkpoint_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--pretrained", action="store_true", help="是否使用预训练模型")
    parser.add_argument("--pretrained_path", type=str, default="", help="预训练模型路径")
    args = parser.parse_args()
    
    # 设置分布式训练
    local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    # 设置日志
    timestamp = setup_logging(local_rank)
    
    # 记录训练历史
    history = {
        'train_losses': [],
        'train_accs': [],
        'test_losses': [],
        'test_accs': [],
        'best_acc': 0.0,
        'best_epoch': 0
    }
    
    # 修改训练参数
    NUM_EPOCHS = 200  # 增加最大训练轮数
    BATCH_SIZE = 32   # 减小批量大小以提高泛化能力
    LEARNING_RATE = 0.0001  # 降低学习率
    WEIGHT_DECAY = 0.0005   # 添加权重衰减
    PATIENCE = 30    # 增加早停耐心值
    
    # 设置混合精度训练
    torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler('cuda')
    
    # 加载数据集
    train_dataset = StanfordCarsDataset(
        root_dir='/data/jj/datasets/scripts/Temporarily_useless/carid/train',
        mat_file=None,
        is_train=True
    )
    
    test_dataset = StanfordCarsDataset(
        root_dir='/data/jj/datasets/scripts/Temporarily_useless/carid/test',
        mat_file=None,
        is_train=False
    )
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        sampler=test_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 初始化模型
    num_classes = len(train_dataset.car_types)
    model = VehicleDetectionModel(num_classes=num_classes)
    
    # 加载预训练模型
    if args.pretrained and args.pretrained_path:
        if local_rank == 0:
            logging.info(f"Loading pretrained model from {args.pretrained_path}")
        try:
            checkpoint = torch.load(args.pretrained_path, map_location=device)
            # 加载模型权重
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                if local_rank == 0:
                    logging.info("Successfully loaded model weights")
                # 可选：加载优化器状态
                if 'optimizer_state_dict' in checkpoint and local_rank == 0:
                    logging.info("Found optimizer state, but skipping for fresh start")
            else:
                model.load_state_dict(checkpoint, strict=False)
                if local_rank == 0:
                    logging.info("Successfully loaded model weights (direct state dict)")
        except Exception as e:
            if local_rank == 0:
                logging.error(f"Error loading pretrained model: {str(e)}")
            raise e
    
    model = model.to(device, memory_format=torch.channels_last)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    initial_lr = 5e-4  # 提高初始学习率
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=0.05  # 增加权重衰减
    )
    
    # 使用带预热的余退火学习率调度器
    warmup_epochs = 3  # 减少预热轮数
    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # 更快的预热
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # 更缓慢的衰减
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))  # 最小学习率提高到初始值的10%
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 添加学习率监控
    if local_rank == 0:
        writer = SummaryWriter(f'runs/train_{timestamp}')
    
    try:
        best_acc = 0
        save_interval = 10
        patience = 30  # 增加耐心值
        min_delta = 0.001  # 添加最小改善阈值
        patience_counter = 0
        
        for epoch in range(NUM_EPOCHS):
            train_sampler.set_epoch(epoch)
            model.train()
            train_loss = 0.0
            train_correct = 0
            total = 0
            
            # 只在rank 0上打印进度
            if local_rank == 0:
                pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
            else:
                pbar = train_loader
            
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.cuda(), labels.cuda()
                
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                if local_rank == 0:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100.*train_correct/total:.2f}%'
                    })
            
            # 计算平均损失和准确率
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / total
            
            # 验证阶段
            test_loss, test_acc = evaluate(model, test_loader, criterion)
            
            # 记录学习率
            if local_rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Learning Rate', current_lr, epoch)
                logging.info(f'Current learning rate: {current_lr:.6f}')
            
            # 更新学习率
            scheduler.step()
            
            # 早停检查
            if test_acc > best_acc + min_delta:  # 只有显著改善才重置计数器
                best_acc = test_acc
                patience_counter = 0
                if local_rank == 0:
                    save_checkpoint(model, optimizer, epoch, test_acc, timestamp, is_best=True)
                    logging.info(f'New best accuracy: {test_acc:.2f}%')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logging.info(f'Early stopping triggered after {epoch+1} epochs')
                break
                
            if local_rank == 0:
                logging.info(f'Epoch {epoch+1}/{NUM_EPOCHS}:')
                logging.info(f'Learning Rate: {current_lr:.6f}')
                logging.info(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
                logging.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
                logging.info(f'Best Accuracy: {best_acc:.2f}%')
                logging.info(f'Patience Counter: {patience_counter}/{patience}')
                
                # 更新并保存训练历史
                update_history(history, train_loss, train_acc, test_loss, test_acc)
                plot_metrics(history, timestamp)
                
                # 保存模型逻辑
                is_best = test_acc > best_acc
                if is_best:
                    best_acc = test_acc
                    history['best_acc'] = best_acc
                    history['best_epoch'] = epoch
                    save_checkpoint(model, optimizer, epoch, test_acc, timestamp, is_best=True)
                    logging.info(f'Saved best model with accuracy: {test_acc:.2f}%')
                elif (epoch + 1) % save_interval == 0:  # 每save_interval个epoch保存一次常规检查点
                    save_checkpoint(model, optimizer, epoch, test_acc, timestamp, is_best=False)
                    logging.info(f'Saved checkpoint at epoch {epoch+1}')
    
    except Exception as e:
        logging.error(f"Error in training: {str(e)}")
        raise e
    
    finally:
        cleanup_distributed()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if local_rank == 0:
            writer.close()

def evaluate(model, loader, criterion):
    """评估函数"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return test_loss / len(loader), 100. * correct / total

if __name__ == '__main__':
    main()