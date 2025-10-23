import gradio as gr
import time
import numpy as np

# 模拟的模型选项
END_TO_END_MODELS = [
    "Qwen2.5-Omni-3B",
    "Qwen2.5-Omni-7B",
    "Qwen3-Omni-30B-A3B-Instruct", 
    "Qwen3-Omni-30B-A3B-Thinking"
]

SEPARATED_MODELS = {
    "ASR": ["Whisper", "Wav2Vec2", "Conformer"],
    "LLM": ["Qwen3-4B-Instruct", "Qwen3-8B"],
    "TTS": ["Bark", "VITS", "Tacotron2"]
}

def process_audio(audio_input, conversation_mode, system_prompt, language, speed, emotion,
                  end_to_end_model, asr_model, llm_model, tts_model, text_input, image_input, video_input):
    """
    处理音频输入并返回响应
    """
    if conversation_mode == "全模态模型":
        # 检查是否有任何输入
        has_input = any([audio_input, text_input, image_input, video_input])
        if not has_input:
            return None, "", ""
    else:
        if audio_input is None:
            return None, "", ""
    
    # 模拟处理过程
    time.sleep(1)
    
    # 构建输入描述
    input_desc = []
    if audio_input:
        input_desc.append("语音")
    if text_input:
        input_desc.append("文本")
    if image_input:
        input_desc.append("图像")
    if video_input:
        input_desc.append("视频")
    
    input_types = "、".join(input_desc) if input_desc else "音频"
    
    # 模拟转录结果
    transcription = f"[模拟转录] 用户通过{input_types}输入信息"
    
    # 构建当前配置信息
    current_config = ""
    if conversation_mode == "全模态模型":
        current_config = f"- 模型: {end_to_end_model}"
    else:
        current_config = f"- ASR模型: {asr_model}\n- LLM模型: {llm_model}\n- TTS模型: {tts_model}"
    
    # 模拟AI响应
    ai_response = f"你好！我已经收到你的{input_types}信息。当前设置:\n- 模式: {conversation_mode}\n{current_config}\n- 语言: {language}\n- 语速: {speed}\n- 情感: {emotion}"
    
    # 模拟生成的响应音频
    sample_rate = 22050
    duration = 2
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 440 * t)
    
    return (sample_rate, audio_data), transcription, ai_response

def update_input_interface(mode):
    """根据对话模式更新输入界面"""
    if mode == "全模态模型":
        return [
            gr.update(visible=True),   # 多模态输入组
            gr.update(visible=False)   # 仅音频输入
        ]
    else:
        return [
            gr.update(visible=False),  # 多模态输入组
            gr.update(visible=True)    # 仅音频输入
        ]

def update_model_selection(mode):
    """根据选择的对话模式更新模型选择组件"""
    if mode == "全模态模型":
        return [
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        ]
    else:
        return [
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True)
        ]

def save_system_prompt(prompt):
    """保存系统提示词"""
    return f"系统提示词已保存: {prompt[:50]}..." if len(prompt) > 50 else f"系统提示词已保存: {prompt}"

with gr.Blocks(title="实时语音对话系统") as demo:
    gr.Markdown("# 🎙️ 实时语音对话系统")
    
    with gr.Tab("主页"):
        with gr.Row():
            # 左侧区域 - 语音交互界面
            with gr.Column(scale=2):
                # 对话模式选择
                conversation_mode = gr.Radio(
                    choices=["全模态模型", "交互式语音"],
                    value="全模态模型",
                    label="对话模式"
                )
                
                # 多模态输入区域（全模态模型时显示）
                with gr.Group(visible=True) as multimodal_input_group:
                    gr.Markdown("### 📥 多模态输入")
                    # 第一排：图像和视频输入
                    with gr.Row():
                        with gr.Column():
                            image_input = gr.Image(
                                type="filepath",
                                label="🖼️ 图像"
                            )
                        with gr.Column():
                            video_input = gr.Video(
                                label="🎬 视频"
                            )
                    
                    # 第二排：文本输入
                    with gr.Row():
                        text_input = gr.Textbox(
                            label="📝 文本",
                            placeholder="请输入文本..."
                        )
                    
                    # 第三排：音频输入
                    with gr.Row():
                        audio_input_multi = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            label="🎤 语音"
                        )
                
                # 仅音频输入区域（交互式语音时显示）
                with gr.Group(visible=False) as audio_only_group:
                    gr.Markdown("### 🎤 音频输入")
                    audio_input_single = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="点击录音或上传音频"
                    )
                
                # 输出区域
                with gr.Group():
                    gr.Markdown("### 📤 AI回应")
                    audio_output = gr.Audio(
                        label="🔊 语音回应",
                        autoplay=True
                    )
                
                # 转录和响应显示
                with gr.Row():
                    with gr.Column():
                        transcription_output = gr.Textbox(
                            label="📝 转文字结果",
                            interactive=False
                        )
                    with gr.Column():
                        response_output = gr.Textbox(
                            label="🤖 文本回应",
                            interactive=False
                        )
                
                # 提交按钮
                submit_btn = gr.Button("🚀 发送", variant="primary", elem_classes=["send-button"])
            
            # 右侧区域 - 配置界面
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 🤖 模型选择")
                    
                    # 全模态模型选择
                    end_to_end_model = gr.Dropdown(
                        choices=END_TO_END_MODELS,
                        value=END_TO_END_MODELS[0],
                        label="全模态模型",
                        visible=True
                    )
                    
                    # 分离式模型选择
                    asr_model = gr.Dropdown(
                        choices=SEPARATED_MODELS["ASR"],
                        value=SEPARATED_MODELS["ASR"][0],
                        label="ASR模型",
                        visible=False
                    )
                    
                    llm_model = gr.Dropdown(
                        choices=SEPARATED_MODELS["LLM"],
                        value=SEPARATED_MODELS["LLM"][0],
                        label="LLM模型",
                        visible=False
                    )
                    
                    tts_model = gr.Dropdown(
                        choices=SEPARATED_MODELS["TTS"],
                        value=SEPARATED_MODELS["TTS"][0],
                        label="TTS模型",
                        visible=False
                    )
                
                with gr.Group():
                    gr.Markdown("### ⚙️ 系统配置")
                    
                    system_prompt = gr.Textbox(
                        value="你是一个智能助手，请用友好和专业的语气回答问题。",
                        label="系统提示词",
                        lines=3
                    )
                    
                    # 添加保存按钮
                    with gr.Row():
                        save_prompt_btn = gr.Button("💾 保存", size="sm")
                        save_status = gr.Textbox(label="", interactive=False, visible=False)
                    
                    language = gr.Dropdown(
                        choices=["中文", "English", "日本語", "Français", "Español"],
                        value="中文",
                        label="语言"
                    )
                    
                    with gr.Row():
                        speed = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="语速"
                        )
                        
                        emotion = gr.Dropdown(
                            choices=["默认", "开心", "严肃", "温柔", "兴奋"],
                            value="默认",
                            label="情感"
                        )
    
    # 事件处理
    conversation_mode.change(
        fn=update_input_interface,
        inputs=[conversation_mode],
        outputs=[multimodal_input_group, audio_only_group]
    )
    
    conversation_mode.change(
        fn=update_model_selection,
        inputs=[conversation_mode],
        outputs=[end_to_end_model, asr_model, llm_model, tts_model]
    )
    
    def get_active_inputs(conversation_mode, audio_multi, audio_single, text_inp, image_inp, video_inp):
        """根据对话模式选择活动的输入"""
        if conversation_mode == "全模态模型":
            return audio_multi, text_inp, image_inp, video_inp
        else:
            return audio_single, "", None, None
    
    def handle_submit(mode, audio_multi, audio_single, text_inp, image_inp, video_inp, 
                      system_prompt, language, speed, emotion, end_to_end_model, 
                      asr_model, llm_model, tts_model):
        """处理提交事件的函数"""
        active_inputs = get_active_inputs(mode, audio_multi, audio_single, text_inp, image_inp, video_inp)
        return process_audio(*active_inputs, mode, system_prompt, language, speed, emotion,
                           end_to_end_model, asr_model, llm_model, tts_model)
    
    submit_btn.click(
        fn=handle_submit,
        inputs=[conversation_mode, audio_input_multi, audio_input_single, text_input, image_input, video_input,
                system_prompt, language, speed, emotion, end_to_end_model, asr_model, llm_model, tts_model],
        outputs=[audio_output, transcription_output, response_output]
    )
    
    # 保存系统提示词
    save_prompt_btn.click(
        fn=save_system_prompt,
        inputs=[system_prompt],
        outputs=[save_status]
    )

if __name__ == "__main__":
    demo.launch()