"""
多模态情感识别系统 - 主入口

基于 StructBERT + emotion2vec + ViT 的中文多模态情感识别与可视化系统。

使用的预训练模型:
- 文本: StructBERT 中文情感分类 (阿里达摩院, ModelScope)
- 语音: emotion2vec+ large (Ma et al., 2024, FunASR)
- 面部: ViT Face Expression (trpakov, HuggingFace, FER2013)

增量贡献:
1. 跨模态统一情感标签映射体系
2. 注意力加权晚期融合模块
3. 完整可视化流水线（雷达图、时间线、注意力热力图、桑基图）
4. 端到端多模态集成系统
5. 实时面部表情识别
6. 基于 Gradio 的交互式 Web 系统
"""

import gradio as gr

from config import GRADIO_CONFIG


def create_about_tab() -> gr.Tab:
    """创建关于页面"""
    with gr.Tab("ℹ️ 关于") as tab:
        gr.Markdown("""
# 🎭 多模态情感识别系统

## 系统架构

```
                    ┌──────────────┐
   文本输入 ──────▶│  StructBERT   │──▶ 文本情感概率
                    │  (中文情感)   │        │
                    └──────────────┘        │
                                            │     ┌──────────────┐
                    ┌──────────────┐        ├────▶│  注意力加权    │──▶ 融合结果
   语音输入 ──────▶│  emotion2vec  │──▶ 语音情感概率 │  晚期融合模块  │
                    │  (2024 SOTA)  │        ├────▶│  (增量贡献)    │
                    └──────────────┘        │     └──────────────┘
                                            │
                    ┌──────────────┐        │
   面部图片 ──────▶│   ViT-base    │──▶ 面部情感概率
                    │  (FER2013)    │
                    └──────────────┘
```

## 使用的预训练模型（参考出处）

| 模态 | 模型 | 来源 | 参考文献 |
|------|------|------|----------|
| 文本 | StructBERT 中文情感分类 | 阿里达摩院 / ModelScope | Wang et al., "StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding", ICLR 2020 |
| 语音 | emotion2vec+ large | FunASR / ModelScope | Ma et al., "emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation", ACL 2024 |
| 面部 | ViT Face Expression | HuggingFace (trpakov) | Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021; 训练数据: FER2013 |

## 本项目增量贡献

1. **跨模态统一情感标签映射** — 将三个不同模型的输出统一映射到 Ekman 6+1 情感体系
2. **注意力加权晚期融合模块** — 基于 MLP 的动态权重融合，自动根据模态置信度分配权重
3. **丰富的可视化流水线** — Plotly 交互雷达图、注意力热力图、情感时间线、融合桑基图
4. **端到端多模态集成** — 支持文本+语音+面部的任意组合输入
5. **视频情感分析** — 自动提取帧和音频，生成双轨情感时间线
6. **实时表情识别** — 摄像头流式处理与实时可视化

## 情感标签体系

| 情感 | 英文 | Emoji |
|------|------|-------|
| 高兴 | Happy | 😊 |
| 悲伤 | Sad | 😢 |
| 愤怒 | Angry | 😠 |
| 恐惧 | Fearful | 😨 |
| 厌恶 | Disgusted | 🤢 |
| 惊讶 | Surprised | 😲 |
| 中性 | Neutral | 😐 |

## 技术栈

- **深度学习**: PyTorch, Transformers, FunASR, ModelScope
- **视觉处理**: OpenCV, MediaPipe
- **音频处理**: librosa, soundfile
- **可视化**: Plotly, Matplotlib
- **Web 框架**: Gradio
        """)
    return tab


def build_app() -> gr.Blocks:
    """构建完整的 Gradio 应用"""
    # 加载自定义 CSS
    css_path = "assets/custom.css"
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            custom_css = f.read()
    except FileNotFoundError:
        custom_css = ""

    with gr.Blocks(
        title=GRADIO_CONFIG["title"],
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="cyan",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Noto Sans SC"),
        ),
        css=custom_css,
    ) as app:
        gr.Markdown(
            f"# {GRADIO_CONFIG['title']}\n"
            f"*{GRADIO_CONFIG['description']}*"
        )

        # 延迟导入避免模型提前加载
        from ui.text_tab import create_text_tab
        from ui.audio_tab import create_audio_tab
        from ui.face_tab import create_face_tab
        from ui.multimodal_tab import create_multimodal_tab
        from ui.video_tab import create_video_tab
        from ui.realtime_tab import create_realtime_tab

        create_text_tab()
        create_audio_tab()
        create_face_tab()
        create_multimodal_tab()
        create_video_tab()
        create_realtime_tab()
        create_about_tab()

        gr.Markdown(
            "---\n"
            "*多模态情感识别系统 | "
            "文本(StructBERT) + 语音(emotion2vec) + 面部(ViT) | "
            "注意力加权融合*"
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_port=GRADIO_CONFIG["server_port"],
        share=GRADIO_CONFIG["share"],
        inbrowser=True,
    )
