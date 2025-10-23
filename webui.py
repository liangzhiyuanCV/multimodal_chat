import time
from dataclasses import dataclass
from typing import Optional, Any

import gradio as gr
import numpy as np

@dataclass
class AudioProcessParams:
    """
    音频处理参数封装类
    
    用于封装处理音频输入所需的所有参数，避免函数参数列表过长的问题。
    """
    # 输入相关参数
    audio_input: Optional[str] = None           # 音频输入文件路径
    text_input: Optional[str] = None            # 文本输入内容
    image_input: Optional[str] = None           # 图像输入文件路径
    video_input: Optional[str] = None           # 视频输入文件路径
    
    # 配置相关参数
    conversation_mode: str = "全模态模型"        # 对话模式（全模态模型/交互式语音）
    system_prompt: str = ""                     # 系统提示词
    language: str = "中文"                      # 语言设置
    speed: float = 1.0                          # 语速控制
    emotion: str = "默认"                       # 情感设置
    
    # 模型选择相关参数
    end_to_end_model: str = ""                  # 端到端模型选择
    asr_model: str = ""                         # ASR模型选择
    llm_model: str = ""                         # LLM模型选择
    tts_model: str = ""                         # TTS模型选择

@dataclass
class InputParams:
    """
    输入参数封装类
    
    用于封装get_active_inputs函数所需的参数。
    """
    conversation_mode: str              # 对话模式
    audio_multi: Optional[str]          # 多模态音频输入
    audio_single: Optional[str]         # 单模态音频输入
    text_inp: str                       # 文本输入
    image_inp: Optional[str]            # 图像输入
    video_inp: Optional[str]            # 视频输入

@dataclass
class CreateParams:
    """
    创建参数封装类
    
    用于封装create_audio_params函数所需的参数。
    """
    # 模式和输入参数
    mode: str = ""
    audio_multi: Optional[str] = None
    audio_single: Optional[str] = None
    text_inp: str = ""
    image_inp: Optional[str] = None
    video_inp: Optional[str] = None
    
    # 配置参数
    system_prompt: str = ""
    language: str = "中文"
    speed: float = 1.0
    emotion: str = "默认"
    
    # 模型参数
    end_to_end_model: str = ""
    asr_model: str = ""
    llm_model: str = ""
    tts_model: str = ""

@dataclass
class SubmitParams:
    """
    提交参数封装类
    
    用于封装handle_submit函数所需的参数。
    """
    # 模式和输入参数
    mode: str = ""
    audio_multi: Optional[str] = None
    audio_single: Optional[str] = None
    text_inp: str = ""
    image_inp: Optional[str] = None
    video_inp: Optional[str] = None
    
    # 配置参数
    system_prompt: str = ""
    language: str = "中文"
    speed: float = 1.0
    emotion: str = "默认"
    
    # 模型参数
    end_to_end_model: str = ""
    asr_model: str = ""
    llm_model: str = ""
    tts_model: str = ""

# 模拟的端到端模型选项
END_TO_END_MODELS = [
    "Qwen2.5-Omni-3B",
    "Qwen2.5-Omni-7B",
    "Qwen3-Omni-30B-A3B-Instruct", 
    "Qwen3-Omni-30B-A3B-Thinking"
]

# 模拟的分离式模型选项
SEPARATED_MODELS = {
    "ASR": ["Whisper", "Wav2Vec2", "Conformer"],      # 自动语音识别模型
    "LLM": ["Qwen3-4B-Instruct", "Qwen3-8B"],         # 大语言模型
    "TTS": ["Bark", "VITS", "Tacotron2"]              # 文本转语音模型
}

def process_audio(params: AudioProcessParams):
    """
    处理音频输入并返回响应
    
    根据不同的对话模式处理用户输入（音频、文本、图像、视频等），并生成相应的AI响应。
    
    Args:
        params (AudioProcessParams): 封装了所有处理所需参数的对象
        
    Returns:
        tuple: 包含三个元素的元组:
            - 音频输出: (采样率, 音频数据) 或 None
            - 转录文本: 用户输入的文本表示
            - AI响应: AI生成的文本响应
    """
    # 根据对话模式检查输入有效性
    if params.conversation_mode == "全模态模型":
        # 全模态模型模式下，检查是否有任何输入
        has_input = any([params.audio_input, params.text_input, params.image_input, params.video_input])
        if not has_input:
            return None, "", ""
    else:
        # 交互式语音模式下，必须有音频输入
        if params.audio_input is None:
            return None, "", ""
    
    # 模拟处理过程（添加1秒延迟）
    time.sleep(1)
    
    # 构建输入描述，用于显示用户通过哪些方式输入了信息
    input_desc = []
    if params.audio_input:
        input_desc.append("语音")
    if params.text_input:
        input_desc.append("文本")
    if params.image_input:
        input_desc.append("图像")
    if params.video_input:
        input_desc.append("视频")
    
    # 格式化输入类型描述
    input_types = "、".join(input_desc) if input_desc else "音频"
    
    # 模拟转录结果
    transcription = f"[模拟转录] 用户通过{input_types}输入信息"
    
    # 根据对话模式构建当前配置信息
    current_config = ""
    if params.conversation_mode == "全模态模型":
        # 全模态模型模式下显示端到端模型
        current_config = f"- 模型: {params.end_to_end_model}"
    else:
        # 交互式语音模式下显示分离式模型组合
        current_config = f"- ASR模型: {params.asr_model}\n- LLM模型: {params.llm_model}\n- TTS模型: {params.tts_model}"
    
    # 构建AI响应文本，包含当前设置信息
    ai_response = f"你好！我已经收到你的{input_types}信息。"
    
    # 模拟生成响应音频（440Hz正弦波，持续2秒）
    sample_rate = 22050
    duration = 2
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 440 * t)
    
    return (sample_rate, audio_data), transcription, ai_response

def update_input_interface(mode):
    """
    根据对话模式更新输入界面的可见性
    
    在全模态模型和交互式语音模式之间切换时，控制不同输入组件的显示状态。
    
    Args:
        mode (str): 当前对话模式，可选值为"全模态模型"或"交互式语音"
        
    Returns:
        list: 包含两个gr.update对象的列表，分别控制多模态输入组和仅音频输入组的可见性
    """
    if mode == "全模态模型":
        # 全模态模型模式：显示多模态输入组，隐藏仅音频输入组
        return [
            gr.update(visible=True),   # 多模态输入组
            gr.update(visible=False)   # 仅音频输入
        ]
    else:
        # 交互式语音模式：隐藏多模态输入组，显示仅音频输入组
        return [
            gr.update(visible=False),  # 多模态输入组
            gr.update(visible=True)    # 仅音频输入
        ]

def update_model_selection(mode):
    """
    根据选择的对话模式更新模型选择组件的可见性
    
    在不同对话模式下，显示相应的模型选择组件。
    
    Args:
        mode (str): 当前对话模式，可选值为"全模态模型"或"交互式语音"
        
    Returns:
        list: 包含四个gr.update对象的列表，分别控制端到端模型和三个分离式模型组件的可见性
    """
    if mode == "全模态模型":
        # 全模态模型模式：显示端到端模型选择，隐藏分离式模型选择
        return [
            gr.update(visible=True),   # 端到端模型
            gr.update(visible=False),  # ASR模型
            gr.update(visible=False),  # LLM模型
            gr.update(visible=False)   # TTS模型
        ]
    else:
        # 交互式语音模式：隐藏端到端模型选择，显示分离式模型选择
        return [
            gr.update(visible=False),  # 端到端模型
            gr.update(visible=True),   # ASR模型
            gr.update(visible=True),   # LLM模型
            gr.update(visible=True)    # TTS模型
        ]

def get_active_inputs(params: InputParams):
    """
    根据对话模式选择活动的输入
    
    Args:
        params (InputParams): 封装了所有输入参数的对象
        
    Returns:
        tuple: 活动输入的元组 (audio_input, text_input, image_input, video_input)
    """
    if params.conversation_mode == "全模态模型":
        return params.audio_multi, params.text_inp, params.image_inp, params.video_inp
    else:
        return params.audio_single, "", None, None

def create_audio_params(params: CreateParams):
    """
    创建AudioProcessParams实例
    
    Args:
        params (CreateParams): 封装了所有创建参数的对象
        
    Returns:
        AudioProcessParams: 封装了所有参数的对象
    """
    # 获取当前活动的输入
    input_params = InputParams(
        conversation_mode=params.mode,
        audio_multi=params.audio_multi,
        audio_single=params.audio_single,
        text_inp=params.text_inp,
        image_inp=params.image_inp,
        video_inp=params.video_inp
    )
    
    audio_input, text_input, image_input, video_input = get_active_inputs(input_params)
    
    # 创建并返回参数对象
    return AudioProcessParams(
        audio_input=audio_input,
        text_input=text_input,
        image_input=image_input,
        video_input=video_input,
        conversation_mode=params.mode,
        system_prompt=params.system_prompt,
        language=params.language,
        speed=params.speed,
        emotion=params.emotion,
        end_to_end_model=params.end_to_end_model,
        asr_model=params.asr_model,
        llm_model=params.llm_model,
        tts_model=params.tts_model
    )

def handle_submit(params: SubmitParams):
    """
    处理提交按钮点击事件
    
    Args:
        params (SubmitParams): 封装了所有提交所需参数的对象
        
    Returns:
        tuple: (音频输出, 转录文本, AI响应文本)
    """
    # 创建参数对象
    create_params = CreateParams(
        mode=params.mode,
        audio_multi=params.audio_multi,
        audio_single=params.audio_single,
        text_inp=params.text_inp,
        image_inp=params.image_inp,
        video_inp=params.video_inp,
        system_prompt=params.system_prompt,
        language=params.language,
        speed=params.speed,
        emotion=params.emotion,
        end_to_end_model=params.end_to_end_model,
        asr_model=params.asr_model,
        llm_model=params.llm_model,
        tts_model=params.tts_model
    )
    
    # 处理音频并返回结果
    return process_audio(create_audio_params(create_params))

def handle_submit_adapter(mode, audio_multi, audio_single, text_inp, image_inp, video_inp,
                         system_prompt, language, speed, emotion, end_to_end_model,
                         asr_model, llm_model, tts_model):
    """
    提交处理适配器函数
    
    将Gradio的参数列表转换为SubmitParams对象，然后调用handle_submit函数。
    
    Args:
        mode (str): 对话模式
        audio_multi: 多模态音频输入
        audio_single: 单模态音频输入
        text_inp (str): 文本输入
        image_inp: 图像输入
        video_inp: 视频输入
        system_prompt (str): 系统提示词
        language (str): 语言设置
        speed (float): 语速
        emotion (str): 情感设置
        end_to_end_model (str): 端到端模型
        asr_model (str): ASR模型
        llm_model (str): LLM模型
        tts_model (str): TTS模型
        
    Returns:
        tuple: (音频输出, 转录文本, AI响应文本)
    """
    # 创建参数对象
    params = SubmitParams(
        mode=mode,
        audio_multi=audio_multi,
        audio_single=audio_single,
        text_inp=text_inp,
        image_inp=image_inp,
        video_inp=video_inp,
        system_prompt=system_prompt,
        language=language,
        speed=speed,
        emotion=emotion,
        end_to_end_model=end_to_end_model,
        asr_model=asr_model,
        llm_model=llm_model,
        tts_model=tts_model
    )
    
    # 调用处理函数
    return handle_submit(params)

def save_system_prompt(prompt):
    """
    保存系统提示词
    
    保存用户设置的系统提示词，并返回保存状态信息。
    
    Args:
        prompt (str): 用户输入的系统提示词
        
    Returns:
        str: 保存状态信息，如果提示词过长则截断显示
    """
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
    
    # 提交按钮点击事件处理
    submit_btn.click(
        fn=handle_submit_adapter,
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