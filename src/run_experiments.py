import os
import subprocess
import sys
import time
import json
import shutil
from datetime import datetime

def run_experiment(script_name, experiment_name, args=None):
    """运行单个实验并只保存关键结果"""
    print(f"\n{'='*50}")
    print(f"开始运行实验: {experiment_name}")
    print(f"{'='*50}")
    
    # 构建命令
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    
    # 添加结果文件参数
    results_dir = "experiment_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建结果文件路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"{experiment_name}_{timestamp}.json")
    cmd.extend(["--results_file", results_file])
    
    # 运行实验
    start_time = time.time()
    
    try:
        # 运行命令并捕获输出
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时打印输出到控制台，但不保存到文件
        for line in process.stdout:
            print(line, end='')
        
        # 等待进程完成
        process.wait()
        
        # 计算耗时
        end_time = time.time()
        duration = end_time - start_time
        
        # 检查结果文件是否存在
        if os.path.exists(results_file):
            print(f"\n实验结果已保存到: {results_file}")
            return results_file, process.returncode == 0, duration
        else:
            print(f"\n警告: 结果文件 {results_file} 不存在，实验可能失败")
            return None, False, duration
        
    except Exception as e:
        print(f"运行实验时出错: {str(e)}")
        return None, False, 0

def copy_visualization_files(experiment_name, results_file):
    """复制可视化文件到实验结果目录"""
    # 源目录 - 训练脚本保存可视化文件的地方
    src_results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    
    # 目标目录 - 实验结果目录
    dst_results_dir = os.path.dirname(results_file)
    
    # 创建可视化文件子目录
    vis_dir = os.path.join(dst_results_dir, f"{experiment_name}_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 要复制的可视化文件列表
    vis_files = [
        'loss_curve_lightweight.png',
        'translation_metrics.png',
        'bleu_distribution.png',
        'length_distribution.png',
        'reference_wordcloud.png',
        'hypothesis_wordcloud.png'
    ]
    
    copied_files = {}
    
    for vis_file in vis_files:
        src_path = os.path.join(src_results_dir, vis_file)
        if os.path.exists(src_path):
            # 为每个实验创建唯一的文件名
            dst_filename = f"{experiment_name}_{vis_file}"
            dst_path = os.path.join(vis_dir, dst_filename)
            
            try:
                shutil.copy2(src_path, dst_path)
                copied_files[vis_file] = dst_path
                print(f"已复制可视化文件: {vis_file} -> {dst_path}")
            except Exception as e:
                print(f"复制可视化文件失败 {vis_file}: {str(e)}")
        else:
            print(f"可视化文件不存在: {src_path}")
    
    return copied_files

def extract_key_info(results_file):
    """从结果文件中提取关键信息"""
    if not os.path.exists(results_file):
        return None
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取关键信息
        key_info = {
            "model_config": data.get("model_config", {}),
            "training_config": data.get("training_config", {}),
            "best_epoch": data.get("best_epoch"),
            "best_val_loss": data.get("best_val_loss"),
            "final_test_loss": data.get("final_test_loss"),
            "translation_metrics": data.get("translation_metrics", {}),
            "visualization_files": {}  # 添加可视化文件路径记录
        }
        
        return key_info
    except Exception as e:
        print(f"提取关键信息时出错: {str(e)}")
        return None

def main():
    """运行所有实验"""
    print("开始运行所有实验...")
    
    # 创建结果目录
    results_dir = "experiment_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建实验汇总文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(results_dir, f"experiment_summary_{timestamp}.json")
    
    # 实验列表
    experiments = [
        # 基准实验 - 轻量级模型默认配置
        {
            "name": "baseline_lightweight",
            "script": "train_lightweight.py",
            "args": ["--use_preprocessed"],
            "description": "轻量级模型基准实验"
        },
        
        # 消融实验 - 去掉相对位置编码
        {
            "name": "ablation_no_relative_pos",
            "script": "train_lightweight_no_relative_pos.py",
            "args": ["--use_preprocessed"],
            "description": "消融实验: 去掉相对位置编码"
        },
        
        # 消融实验 - 去掉标签平滑
        {
            "name": "ablation_no_label_smoothing",
            "script": "train_lightweight_no_label_smoothing.py",
            "args": ["--use_preprocessed"],
            "description": "消融实验: 去掉标签平滑"
        },
        
        # 消融实验 - 去掉权重衰减
        {
            "name": "ablation_no_weight_decay",
            "script": "train_lightweight_no_weight_decay.py",
            "args": ["--use_preprocessed"],
            "description": "消融实验: 去掉权重衰减"
        },
        
        # 对比实验 - 不同模型规模
        {
            "name": "comparison_small_model",
            "script": "train_lightweight_small.py",
            "args": ["--use_preprocessed"],
            "description": "对比实验: 小模型"
        },
        
        {
            "name": "comparison_medium_model",
            "script": "train_lightweight_medium.py",
            "args": ["--use_preprocessed"],
            "description": "对比实验: 中等模型"
        },
        
        {
            "name": "comparison_large_model",
            "script": "train_lightweight_large.py",
            "args": ["--use_preprocessed"],
            "description": "对比实验: 大模型"
        },
        
        # 对比实验 - 不同学习率
        {
            "name": "comparison_lr_1e3",
            "script": "train_lightweight_lr_1e3.py",
            "args": ["--use_preprocessed"],
            "description": "对比实验: 学习率1e-3"
        },
        
        {
            "name": "comparison_lr_1e4",
            "script": "train_lightweight_lr_1e4.py",
            "args": ["--use_preprocessed"],
            "description": "对比实验: 学习率1e-4"
        },
        
        {
            "name": "comparison_lr_1e5",
            "script": "train_lightweight_lr_1e5.py",
            "args": ["--use_preprocessed"],
            "description": "对比实验: 学习率1e-5"
        },
        
        # 对比实验 - 不同批次大小
        {
            "name": "comparison_batch_8",
            "script": "train_lightweight_batch_8.py",
            "args": ["--use_preprocessed"],
            "description": "对比实验: 批次大小8"
        },
        
        {
            "name": "comparison_batch_16",
            "script": "train_lightweight_batch_16.py",
            "args": ["--use_preprocessed"],
            "description": "对比实验: 批次大小16"
        },
        
        {
            "name": "comparison_batch_32",
            "script": "train_lightweight_batch_32.py",
            "args": ["--use_preprocessed"],
            "description": "对比实验: 批次大小32"
        }
    ]
    
    # 运行每个实验
    all_results = {
        "experiment_summary": {
            "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_experiments": len(experiments)
        },
        "experiments": []
    }
    
    for exp in experiments:
        print(f"\n准备运行实验: {exp['name']} - {exp['description']}")
        
        # 检查脚本是否存在
        script_path = exp['script']
        if not os.path.exists(script_path):
            print(f"跳过实验 (脚本不存在: {script_path})")
            all_results["experiments"].append({
                "name": exp['name'],
                "description": exp['description'],
                "status": "跳过",
                "reason": "脚本不存在"
            })
            continue
        
        # 运行实验
        results_file, success, duration = run_experiment(script_path, exp['name'], exp['args'])
        
        # 记录实验结果
        exp_result = {
            "name": exp['name'],
            "description": exp['description'],
            "status": "成功" if success else "失败",
            "duration_seconds": duration,
            "results_file": results_file if results_file else ""
        }
        
        # 如果实验成功，提取关键信息
        if success and results_file:
            key_info = extract_key_info(results_file)
            if key_info:
                exp_result["key_info"] = key_info
                
                # 复制可视化文件并记录路径
                print(f"\n正在复制可视化文件...")
                copied_files = copy_visualization_files(exp['name'], results_file)
                if copied_files:
                    # 更新key_info中的可视化文件路径
                    exp_result["key_info"]["visualization_files"] = copied_files
        
        all_results["experiments"].append(exp_result)
    
    # 记录结束时间
    all_results["experiment_summary"]["end_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 保存汇总结果
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    # 打印汇总信息
    print(f"\n{'='*50}")
    print("实验汇总:")
    print(f"{'='*50}")
    
    successful = sum(1 for exp in all_results["experiments"] if exp["status"] == "成功")
    failed = sum(1 for exp in all_results["experiments"] if exp["status"] == "失败")
    skipped = sum(1 for exp in all_results["experiments"] if exp["status"] == "跳过")
    
    print(f"成功: {successful}, 失败: {failed}, 跳过: {skipped}")
    
    # 打印成功实验的关键指标
    print("\n成功实验的关键指标:")
    for exp in all_results["experiments"]:
        if exp["status"] == "成功" and "key_info" in exp:
            key_info = exp["key_info"]
            print(f"\n{exp['name']}:")
            print(f"  最佳验证损失: {key_info.get('best_val_loss', 'N/A')}")
            print(f"  测试损失: {key_info.get('final_test_loss', 'N/A')}")
            
            # 打印翻译指标
            metrics = key_info.get('translation_metrics', {})
            if metrics:
                print("  翻译指标:")
                for metric, value in metrics.items():
                    print(f"    {metric}: {value:.4f}")
            
            # 打印可视化文件信息
            vis_files = key_info.get('visualization_files', {})
            if vis_files:
                print("  可视化文件:")
                for vis_type, vis_path in vis_files.items():
                    print(f"    {vis_type}: {vis_path}")
    
    print(f"\n所有实验完成! 汇总报告已保存到: {summary_file}")

if __name__ == "__main__":
    main()