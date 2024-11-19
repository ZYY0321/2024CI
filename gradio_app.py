import gradio as gr
from inference import CarPredictor
import os
import tempfile
from pathlib import Path

# 设置临时文件目录
TEMP_DIR = Path.home() / '.gradio_cache'
TEMP_DIR.mkdir(exist_ok=True)
os.environ['GRADIO_TEMP_DIR'] = str(TEMP_DIR)

# 初始化预测器
MODEL_PATH = '/data/jj/datasets/scripts/Temporarily_useless/carid/checkpoints/best_model_20241117_233010_acc69.48.pth'
predictor = CarPredictor(MODEL_PATH)

def predict_car(image):
    """
    处理上传的图片并返回预测结果
    """
    # 在用户目录下创建临时文件
    temp_path = TEMP_DIR / "temp_upload.jpg"
    try:
        image.save(str(temp_path))
        predictions = predictor.predict(str(temp_path))
        
        # 格式化输出结果
        result = "🚗 预测结果:\n\n"
        for i, (class_name, probability) in enumerate(predictions, 1):
            result += f"#{i} {class_name}\n"
            result += f"   信心指数: {'▓' * int(probability * 20)}{'░' * (20 - int(probability * 20))} {probability:.2%}\n\n"
        
        return result
    
    except Exception as e:
        return f"预测出错: {str(e)}"
    
    finally:
        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()

# 创建Gradio界面
iface = gr.Interface(
    fn=predict_car,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="预测结果"),
    title="汽车品牌型号识别分类",
    description="上传一张汽车图片，系统将识别其品牌和型号。",
    theme=gr.themes.Soft()
)

# 启动服务
if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    ) 
