# -*- coding: utf-8 -*-
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

# 全局变量，用于跟踪NLTK数据是否已下载
NLTK_DATA_DOWNLOADED = False

# 尝试导入METEOR和ROUGE，如果不可用则跳过相关评估
try:
    from nltk.translate.meteor_score import meteor_score
    HAS_METEOR = True
except ImportError:
    HAS_METEOR = False
    print("警告: nltk.translate.meteor_score 不可用，将跳过METEOR评估")

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("警告: rouge_score 不可用，将跳过ROUGE评估")

import pandas as pd
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
    print("警告: wordcloud 不可用，将跳过词云生成")

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def safe_print(text):
    """安全打印函数，处理Unicode编码问题"""
    try:
        print(text)
    except UnicodeEncodeError:
        # 如果出现编码错误，尝试使用不同的编码方式
        try:
            # 尝试使用UTF-8编码
            print(text.encode('utf-8').decode('utf-8'))
        except:
            # 如果仍然失败，使用ASCII编码并替换非ASCII字符
            try:
                print(text.encode('ascii', 'replace').decode('ascii'))
            except:
                # 最后的备选方案：逐个字符处理
                safe_text = ''.join(c if ord(c) < 128 else f'\\u{ord(c):04x}' for c in text)
                print(safe_text)

def download_nltk_data():
    """下载必要的NLTK数据"""
    global NLTK_DATA_DOWNLOADED
    if NLTK_DATA_DOWNLOADED:
        return
    
    try:
        import nltk
        # 检查wordnet是否已下载
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("正在下载NLTK wordnet数据...")
            nltk.download('wordnet', quiet=True)
            print("NLTK wordnet数据下载完成")
        
        # 检查punkt是否已下载（用于分词）
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("正在下载NLTK punkt数据...")
            nltk.download('punkt', quiet=True)
            print("NLTK punkt数据下载完成")
        
        NLTK_DATA_DOWNLOADED = True
    except Exception as e:
        print(f"下载NLTK数据时出错: {e}")

class TranslationEvaluator:
    """翻译模型评估器"""
    def __init__(self, model, src_vocab, tgt_vocab, device, config=None):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.model.eval()
        
        # 解码策略配置
        if config is None:
            from config import lightweight_config
            config = lightweight_config
            
        self.decoding_strategy = getattr(config, 'decoding_strategy', 'greedy')
        self.beam_size = getattr(config, 'beam_size', 5)
        self.length_penalty = getattr(config, 'length_penalty', 0.6)
        self.repetition_penalty = getattr(config, 'repetition_penalty', 1.2)
        self.min_length = getattr(config, 'min_length', 4)
        self.max_decode_length = getattr(config, 'max_decode_length', 50)
        
        # 确保NLTK数据已下载
        if HAS_METEOR:
            download_nltk_data()
    
    def evaluate(self, test_loader, num_samples=100):
        """评估翻译模型性能"""
        print("开始评估翻译模型...")
        print(f"使用解码策略: {self.decoding_strategy}")
        if self.decoding_strategy == "beam_search":
            print(f"束大小: {self.beam_size}, 长度惩罚: {self.length_penalty}, 重复惩罚: {self.repetition_penalty}")
        
        # 收集预测和参考翻译
        references = []
        hypotheses = []
        src_sentences = []
        
        sample_count = 0
        with torch.no_grad():
            for batch in test_loader:
                src = batch['de_sequences'].to(self.device)
                tgt = batch['en_sequences'].to(self.device)
                
                batch_size = src.size(0)
                for i in range(batch_size):
                    if sample_count >= num_samples:
                        break
                    
                    # 获取源句子
                    src_tokens = [self.src_vocab.idx2word[idx.item()] for idx in src[i] 
                                 if idx.item() != self.src_vocab.word2idx[self.src_vocab.pad_token]]
                    src_sentence = ' '.join(src_tokens)
                    
                    # 获取参考翻译
                    tgt_tokens = [self.tgt_vocab.idx2word[idx.item()] for idx in tgt[i] 
                                 if idx.item() != self.tgt_vocab.word2idx[self.tgt_vocab.pad_token]
                                 and idx.item() != self.tgt_vocab.word2idx[self.tgt_vocab.bos_token]
                                 and idx.item() != self.tgt_vocab.word2idx[self.tgt_vocab.eos_token]]
                    ref_sentence = ' '.join(tgt_tokens)
                    
                    # 生成预测翻译
                    start_symbol = self.tgt_vocab.word2idx[self.tgt_vocab.bos_token]
                    end_symbol = self.tgt_vocab.word2idx[self.tgt_vocab.eos_token]
                    
                    # 根据配置选择解码策略
                    if self.decoding_strategy == "beam_search":
                        output = self.model.beam_search_decode(
                            src[i:i+1], 
                            None, 
                            max_len=self.max_decode_length, 
                            start_symbol=start_symbol, 
                            end_symbol=end_symbol,
                            beam_size=self.beam_size,
                            length_penalty=self.length_penalty,
                            repetition_penalty=self.repetition_penalty,
                            min_length=self.min_length
                        )
                    else:  # 默认使用贪心解码
                        output = self.model.greedy_decode(
                            src[i:i+1], 
                            None, 
                            max_len=self.max_decode_length, 
                            start_symbol=start_symbol, 
                            end_symbol=end_symbol,
                            repetition_penalty=self.repetition_penalty,
                            min_length=self.min_length
                        )
                    
                    hyp_tokens = [self.tgt_vocab.idx2word[idx.item()] for idx in output[0] 
                                 if idx.item() != self.tgt_vocab.word2idx[self.tgt_vocab.pad_token]
                                 and idx.item() != self.tgt_vocab.word2idx[self.tgt_vocab.bos_token]
                                 and idx.item() != self.tgt_vocab.word2idx[self.tgt_vocab.eos_token]]
                    hyp_sentence = ' '.join(hyp_tokens)
                    
                    src_sentences.append(src_sentence)
                    references.append(ref_sentence)
                    hypotheses.append(hyp_sentence)
                    
                    sample_count += 1
                
                if sample_count >= num_samples:
                    break
        
        # 计算各种评估指标
        metrics = self._calculate_metrics(references, hypotheses)
        
        # 可视化结果
        self._visualize_results(src_sentences, references, hypotheses, metrics)
        
        # 显示翻译对比示例
        self._show_translation_examples(src_sentences, references, hypotheses, n=5)
        
        return metrics
    
    def _calculate_metrics(self, references, hypotheses):
        """计算各种翻译评估指标"""
        print("计算评估指标...")
        
        # 准备数据
        ref_tokens_list = [ref.split() for ref in references]
        hyp_tokens_list = [hyp.split() for hyp in hypotheses]
        
        # 计算BLEU分数
        smooth_func = SmoothingFunction().method4
        bleu_scores = []
        for ref_tokens, hyp_tokens in zip(ref_tokens_list, hyp_tokens_list):
            if len(hyp_tokens) > 0:
                bleu1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth_func)
                bleu2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_func)
                bleu3 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth_func)
                bleu4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_func)
                bleu_scores.append((bleu1, bleu2, bleu3, bleu4))
        
        # 计算平均BLEU分数
        if bleu_scores:
            avg_bleu1 = sum(b[0] for b in bleu_scores) / len(bleu_scores)
            avg_bleu2 = sum(b[1] for b in bleu_scores) / len(bleu_scores)
            avg_bleu3 = sum(b[2] for b in bleu_scores) / len(bleu_scores)
            avg_bleu4 = sum(b[3] for b in bleu_scores) / len(bleu_scores)
        else:
            avg_bleu1 = avg_bleu2 = avg_bleu3 = avg_bleu4 = 0
        
        # 计算Corpus BLEU分数
        corpus_bleu_score = corpus_bleu([[ref] for ref in ref_tokens_list], hyp_tokens_list)
        
        # 计算METEOR分数（如果可用）
        avg_meteor = 0
        if HAS_METEOR:
            try:
                meteor_scores = []
                for ref, hyp in zip(ref_tokens_list, hyp_tokens_list):
                    if len(hyp) > 0:
                        meteor_scores.append(meteor_score([ref], hyp))
                avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
            except Exception as e:
                print(f"计算METEOR分数时出错: {e}")
                avg_meteor = 0
        
        # 计算ROUGE分数（如果可用）
        avg_rouge1 = avg_rouge2 = avg_rougeL = 0
        if HAS_ROUGE:
            try:
                scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                rouge1_scores = []
                rouge2_scores = []
                rougeL_scores = []
                
                for ref, hyp in zip(references, hypotheses):
                    if hyp.strip():  # 确保预测翻译不为空
                        scores = scorer.score(ref, hyp)
                        rouge1_scores.append(scores['rouge1'].fmeasure)
                        rouge2_scores.append(scores['rouge2'].fmeasure)
                        rougeL_scores.append(scores['rougeL'].fmeasure)
                
                avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
                avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
                avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
            except Exception as e:
                print(f"计算ROUGE分数时出错: {e}")
                avg_rouge1 = avg_rouge2 = avg_rougeL = 0
        
        # 计算词重叠率
        exact_matches = sum(1 for ref, hyp in zip(references, hypotheses) if ref == hyp)
        exact_match_rate = exact_matches / len(references) if references else 0
        
        # 收集所有指标
        metrics = {
            'BLEU-1': avg_bleu1,
            'BLEU-2': avg_bleu2,
            'BLEU-3': avg_bleu3,
            'BLEU-4': avg_bleu4,
            'Corpus BLEU': corpus_bleu_score,
        }
        
        if HAS_METEOR:
            metrics['METEOR'] = avg_meteor
        
        if HAS_ROUGE:
            metrics['ROUGE-1'] = avg_rouge1
            metrics['ROUGE-2'] = avg_rouge2
            metrics['ROUGE-L'] = avg_rougeL
        
        metrics['Exact Match'] = exact_match_rate
        
        return metrics
    
    def _visualize_results(self, src_sentences, references, hypotheses, metrics):
        """可视化评估结果"""
        print("可视化评估结果...")
        
        # 创建结果目录
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # 1. 绘制评估指标柱状图
        plt.figure(figsize=(12, 6))
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = plt.bar(metric_names, metric_values, color='skyblue')
        
        # 在柱状图上添加数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        plt.title('Translation Evaluation Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'translation_metrics.png'))
        plt.close()
        
        # 2. 绘制BLEU分数分布
        plt.figure(figsize=(10, 6))
        bleu_scores = []
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = ref.split()
            hyp_tokens = hyp.split()
            if len(hyp_tokens) > 0:
                bleu = sentence_bleu([ref_tokens], hyp_tokens, 
                                    weights=(0.25, 0.25, 0.25, 0.25),
                                    smoothing_function=SmoothingFunction().method4)
                bleu_scores.append(bleu)
        
        if bleu_scores:
            plt.hist(bleu_scores, bins=20, color='green', alpha=0.7)
            plt.title('Distribution of BLEU Scores')
            plt.xlabel('BLEU Score')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(results_dir, 'bleu_distribution.png'))
            plt.close()
        
        # 3. 绘制翻译长度分布
        plt.figure(figsize=(12, 6))
        ref_lengths = [len(ref.split()) for ref in references]
        hyp_lengths = [len(hyp.split()) for hyp in hypotheses]
        
        plt.hist(ref_lengths, bins=20, alpha=0.5, label='Reference', color='blue')
        plt.hist(hyp_lengths, bins=20, alpha=0.5, label='Hypothesis', color='red')
        plt.title('Distribution of Translation Lengths')
        plt.xlabel('Length (words)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(results_dir, 'length_distribution.png'))
        plt.close()
        
        # 4. 绘制词云（如果wordcloud可用）
        if HAS_WORDCLOUD:
            try:
                all_ref_words = ' '.join(references).split()
                all_hyp_words = ' '.join(hypotheses).split()
                
                ref_word_counts = Counter(all_ref_words)
                hyp_word_counts = Counter(all_hyp_words)
                
                # 参考翻译词云
                plt.figure(figsize=(12, 6))
                ref_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(ref_word_counts)
                plt.imshow(ref_wordcloud, interpolation='bilinear')
                plt.title('Reference Translation Word Cloud')
                plt.axis('off')
                plt.savefig(os.path.join(results_dir, 'reference_wordcloud.png'))
                plt.close()
                
                # 预测翻译词云
                plt.figure(figsize=(12, 6))
                hyp_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(hyp_word_counts)
                plt.imshow(hyp_wordcloud, interpolation='bilinear')
                plt.title('Hypothesis Translation Word Cloud')
                plt.axis('off')
                plt.savefig(os.path.join(results_dir, 'hypothesis_wordcloud.png'))
                plt.close()
            except Exception as e:
                print(f"生成词云时出错: {e}")
        
        print(f"所有可视化结果已保存到 {results_dir}")
    
    def _show_translation_examples(self, src_sentences, references, hypotheses, n=5):
        """显示翻译对比示例"""
        safe_print("\n===== 翻译对比示例 =====")
        
        # 随机选择n个示例
        indices = np.random.choice(len(src_sentences), size=min(n, len(src_sentences)), replace=False)
        
        for i, idx in enumerate(indices):
            safe_print(f"\n示例 {i+1}:")
            safe_print(f"源语言 (德语): {src_sentences[idx]}")
            safe_print(f"参考翻译 (英语): {references[idx]}")
            safe_print(f"模型翻译 (英语): {hypotheses[idx]}")
            
            # 计算BLEU分数
            ref_tokens = references[idx].split()
            hyp_tokens = hypotheses[idx].split()
            if len(hyp_tokens) > 0:
                bleu = sentence_bleu([ref_tokens], hyp_tokens, 
                                   weights=(0.25, 0.25, 0.25, 0.25),
                                   smoothing_function=SmoothingFunction().method4)
                safe_print(f"BLEU分数: {bleu:.4f}")
            safe_print("-" * 50)
    
    def analyze_attention(self, test_loader, num_samples=3):
        """分析注意力机制 - 已禁用"""
        print("注意力分析功能已禁用，跳过...")
        return
    
    def _visualize_attention(self, src_tokens, tgt_tokens, attention_weights, sample_idx, results_dir):
        """可视化注意力权重"""
        try:
            # 处理注意力权重，可能是元组或张量
            if isinstance(attention_weights, tuple):
                # 如果是元组，获取最后一个元素
                attention = attention_weights[-1]
            else:
                # 如果是张量列表，获取最后一个张量
                attention = attention_weights[-1]
            
            # 确保是张量
            if not isinstance(attention, torch.Tensor):
                attention = torch.tensor(attention)
            
            # 检查张量形状并适当处理
            if attention.dim() == 4:  # [batch_size, num_heads, seq_len, seq_len]
                # 取第一个样本和第一个头的注意力权重
                attention = attention[0, 0]  # [seq_len, seq_len]
            elif attention.dim() == 3:  # [batch_size, seq_len, seq_len]
                # 取第一个样本的注意力权重
                attention = attention[0]  # [seq_len, seq_len]
            elif attention.dim() == 2:  # [seq_len, seq_len]
                # 已经是正确的形状，不需要处理
                pass
            else:
                print(f"不支持的注意力权重形状: {attention.shape}")
                return
            
            # 转换为numpy数组
            attention = attention.cpu().numpy()
            
            # 确保注意力矩阵的形状与源和目标标记匹配
            if attention.shape[0] != len(tgt_tokens) or attention.shape[1] != len(src_tokens):
                print(f"注意力矩阵形状 ({attention.shape}) 与标记数量不匹配 (目标: {len(tgt_tokens)}, 源: {len(src_tokens)})")
                # 尝试调整大小
                min_rows = min(attention.shape[0], len(tgt_tokens))
                min_cols = min(attention.shape[1], len(src_tokens))
                attention = attention[:min_rows, :min_cols]
                tgt_tokens = tgt_tokens[:min_rows]
                src_tokens = src_tokens[:min_cols]
            
            # 绘制注意力热图
            plt.figure(figsize=(12, 10))
            sns.heatmap(attention, 
                       xticklabels=src_tokens, 
                       yticklabels=tgt_tokens,
                       cmap='YlGnBu')
            plt.title(f'Attention Heatmap - Sample {sample_idx + 1}')
            plt.xlabel('Source Tokens')
            plt.ylabel('Target Tokens')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'attention_heatmap_{sample_idx + 1}.png'))
            plt.close()
            print(f"注意力热图已保存: attention_heatmap_{sample_idx + 1}.png")
        except Exception as e:
            print(f"可视化注意力权重时出错: {e}")
            import traceback
            traceback.print_exc()