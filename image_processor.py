"""
图片提取器 - 从 PDF/Word 文档中提取图片并生成描述
"""

import os
import re
import io
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from PIL import Image

# 图片提取库
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    from docx.oxml.ns import qn
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


class ImageExtractor:
    """图片提取器 - 从文档中提取图片"""

    # 支持的图片类型
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}

    def __init__(self, llm_provider=None, temp_dir: str = None):
        """
        初始化图片提取器

        Args:
            llm_provider: LLM提供商（需要支持视觉模型）
            temp_dir: 临时目录，用于保存提取的图片
        """
        self.llm_provider = llm_provider
        self.temp_dir = Path(temp_dir) if temp_dir else Path("temp_images")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Ollama 视觉模型
        self.vision_model = "llava:7b"

    def extract_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        从 PDF 提取图片

        Args:
            pdf_path: PDF 文件路径

        Returns:
            图片列表 [{'image': PIL.Image, 'page': int, 'index': int}]
        """
        if not PDFPLUMBER_AVAILABLE:
            print("警告: pdfplumber 未安装，无法从 PDF 提取图片")
            return []

        images = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    # 提取图片
                    page_images = page.images
                    for img_idx, img_info in enumerate(page_images):
                        try:
                            # 从 PDF 提取原始图片数据
                            if 'stream' in img_info:
                                # 获取图片位置和大小
                                x0 = img_info.get('x0', 0)
                                y0 = img_info.get('top', 0)
                                x1 = img_info.get('x1', 0)
                                y1 = img_info.get('bottom', 0)

                                # 尝试从页面对象获取图片
                                img_data = self._extract_image_from_pdf_page(page, img_info)
                                if img_data:
                                    img = Image.open(io.BytesIO(img_data))
                                    images.append({
                                        'image': img,
                                        'page': page_num,
                                        'index': img_idx,
                                        'source': pdf_path
                                    })
                        except Exception as e:
                            print(f"  提取第{page_num}页图片失败: {e}")
                            continue

        except Exception as e:
            print(f"  打开 PDF 失败: {e}")

        return images

    def _extract_image_from_pdf_page(self, page, img_info: Dict) -> Optional[bytes]:
        """从 PDF 页面提取图片数据"""
        try:
            # 尝试使用 pdfplumber 的方法获取图片
            for img in page.images:
                if img.get('name') == img_info.get('name'):
                    # 返回原始图片数据（如果可用）
                    return None  # pdfplumber 不直接提供原始图片数据
            return None
        except:
            return None

    def extract_from_docx(self, docx_path: str) -> List[Dict]:
        """
        从 Word 文档提取图片

        Args:
            docx_path: Word 文件路径

        Returns:
            图片列表 [{'image': PIL.Image, 'index': int}]
        """
        if not DOCX_AVAILABLE:
            print("警告: python-docx 未安装，无法从 Word 提取图片")
            return []

        images = []

        try:
            doc = DocxDocument(docx_path)

            # 遍历所有内联图片
            for rel_idx, rel in enumerate(doc.part.rels.values()):
                if "image" in rel.target_ref:
                    try:
                        # 获取图片数据
                        image_part = rel.target_part
                        img = Image.open(io.BytesIO(image_part.blob))

                        images.append({
                            'image': img,
                            'index': rel_idx,
                            'source': docx_path
                        })
                    except Exception as e:
                        print(f"  提取图片失败: {e}")
                        continue

        except Exception as e:
            print(f"  打开 Word 文档失败: {e}")

        return images

    def extract_from_file(self, file_path: str) -> List[Dict]:
        """根据文件类型提取图片"""
        path = Path(file_path)

        if path.suffix.lower() == '.pdf':
            return self.extract_from_pdf(str(path))
        elif path.suffix.lower() in ['.docx', '.doc']:
            return self.extract_from_docx(str(path))

        return []


class ImageDescriber:
    """图片描述器 - 使用视觉模型生成图片描述"""

    # 默认的图片描述提示词
    DEFAULT_PROMPT = """请详细描述这张图片的内容。
描述要求：
1. 完整描述图片中的所有元素
2. 识别图片中的文字（如有）
3. 说明图片的布局和结构
4. 如果是图表，说明数据趋势和关键信息
5. 如果是流程图，说明步骤和逻辑关系

请用中文描述。"""

    def __init__(self, llm_provider=None, vision_model: str = None, vision_provider: str = None):
        """
        初始化图片描述器

        Args:
            llm_provider: LLM提供商
            vision_model: 视觉模型名称（可选，默认从config读取）
            vision_provider: 视觉模型提供商: "ollama" | "minimax"（可选，默认从config读取）
        """
        self.llm_provider = llm_provider

        # 从配置读取默认值
        from config import OLLAMA_VISION_MODEL, MINIMAX_VISION_MODEL, IMAGE_VISION_PROVIDER

        # 视觉模型提供商
        self.vision_provider = vision_provider or IMAGE_VISION_PROVIDER

        # 视觉模型
        if vision_model:
            self.vision_model = vision_model
        elif self.vision_provider == "ollama":
            self.vision_model = OLLAMA_VISION_MODEL
        elif self.vision_provider == "minimax":
            self.vision_model = MINIMAX_VISION_MODEL
        else:
            self.vision_model = OLLAMA_VISION_MODEL

        # 图片压缩配置
        self.max_image_size_bytes = 10 * 1024 * 1024  # 10MB

        # 根据提供商初始化不同的客户端
        self._openai_client = None
        self._minimax_client = None

        if self.vision_provider == "ollama":
            # Ollama 使用 OpenAI 兼容接口
            try:
                from openai import OpenAI
                ollama_base_url = "http://localhost:11434/v1"
                self._openai_client = OpenAI(
                    base_url=ollama_base_url,
                    api_key=""  # Ollama 不需要 API key
                )
            except Exception:
                pass
        elif self.vision_provider == "minimax":
            # MiniMax使用 OpenAI 兼容接口
            try:
                from openai import OpenAI
                from config import MINIMAX_API_KEY, MINIMAX_BASE_URL
                # MiniMax API路径
                minimax_base_url = MINIMAX_BASE_URL.rstrip('/')
                self._minimax_client = OpenAI(
                    api_key=MINIMAX_API_KEY,
                    base_url=f"{minimax_base_url}"
                )
            except Exception:
                pass

    def _compress_image(self, image: Image.Image, max_size_bytes: int = None) -> Image.Image:
        """
        压缩图片使其小于指定大小

        Args:
            image: PIL Image 对象
            max_size_bytes: 最大文件大小（字节），默认 10MB

        Returns:
            压缩后的 PIL Image 对象
        """
        if max_size_bytes is None:
            max_size_bytes = self.max_image_size_bytes

        from io import BytesIO

        # 先检查当前图片大小
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        current_size = buffer.tell()

        if current_size <= max_size_bytes:
            return image

        # 需要压缩，逐步降低质量直到满足大小要求
        # 首先尝试缩小图片尺寸
        width, height = image.size
        compress_image = image

        # 逐步缩小尺寸（每次缩小10%）
        while width > 200 and current_size > max_size_bytes:
            new_width = int(width * 0.9)
            new_height = int(height * 0.9)
            compress_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            buffer = BytesIO()
            compress_image.save(buffer, format='PNG')
            current_size = buffer.tell()

            if current_size <= max_size_bytes:
                return compress_image

            width, height = new_width, new_height

        # 转为JPEG尝试降低质量
        if compress_image.mode not in ('RGB', 'L'):
            compress_image = compress_image.convert('RGB')

        quality = 95
        while quality >= 50 and current_size > max_size_bytes:
            buffer = BytesIO()
            compress_image.save(buffer, format='JPEG', quality=quality)
            current_size = buffer.tell()
            quality -= 5

        if current_size <= max_size_bytes:
            return compress_image

        # 最后手段：继续降低质量和尺寸
        while current_size > max_size_bytes and width > 100:
            new_width = int(width * 0.8)
            new_height = int(height * 0.8)
            compress_image = compress_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            quality = 80
            while quality >= 30 and current_size > max_size_bytes:
                buffer = BytesIO()
                compress_image.save(buffer, format='JPEG', quality=quality)
                current_size = buffer.tell()
                quality -= 10

            width, height = new_width, new_height

        # 确保返回的是RGB模式
        if compress_image.mode not in ('RGB', 'L'):
            compress_image = compress_image.convert('RGB')

        return compress_image

    def describe_image(self, image: Image.Image, prompt: str = None) -> str:
        """
        生成单张图片的描述

        Args:
            image: PIL Image 对象
            prompt: 自定义提示词

        Returns:
            图片描述文本
        """
        if not self.llm_provider and not self._openai_client and not self._minimax_client:
            return self._simple_describe(image)

        try:
            # 压缩图片以满足大小限制
            image = self._compress_image(image)

            # 将图片转换为 base64
            import base64
            from io import BytesIO

            buffer = BytesIO()
            image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # 构建视觉提示
            user_prompt = prompt or self.DEFAULT_PROMPT

            # 根据视觉模型提供商选择调用方式
            if self.vision_provider == "minimax" and self._minimax_client:
                # MiniMax视觉模型
                try:
                    response = self._minimax_client.chat.completions.create(
                        model=self.vision_model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                                    }
                                ]
                            }
                        ]
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"  MiniMax视觉模型调用失败: {e}")

            elif self.vision_provider == "ollama" and self._openai_client:
                # Ollama 视觉模型（OpenAI 兼容接口）
                try:
                    response = self._openai_client.chat.completions.create(
                        model=self.vision_model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                                    }
                                ]
                            }
                        ]
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"  Ollama OpenAI 兼容接口调用失败: {e}")

                # 回退：使用 Ollama /api/generate 接口
                try:
                    import requests
                    ollama_url = "http://localhost:11434/api/generate"
                    payload = {
                        "model": self.vision_model,
                        "prompt": user_prompt,
                        "images": [img_base64],
                        "stream": False,
                        "temperature": 0.1,
                        "max_tokens": 1000
                    }
                    response = requests.post(ollama_url, json=payload, timeout=120)
                    if response.status_code == 200:
                        result = response.json()
                        return result.get("response", "")
                    else:
                        print(f"  Ollama /api/generate 调用失败: {response.status_code}")
                except Exception as e:
                    print(f"  Ollama /api/generate 调用失败: {e}")

            # 最后回退：使用 langchain chat（可能不支持视觉）
            if self.llm_provider:
                response = self.llm_provider.chat(
                    f"请描述这张图片：{user_prompt}"
                )
                return response

        except Exception as e:
            print(f"  图片描述生成失败: {e}")

        return self._simple_describe(image)

    def describe_images_batch(self, images: List[Dict], prompt: str = None) -> List[str]:
        """
        批量生成图片描述

        Args:
            images: 图片列表
            prompt: 自定义提示词

        Returns:
            描述列表
        """
        descriptions = []

        for i, img_info in enumerate(images):
            print(f"  描述第 {i+1}/{len(images)} 张图片...")
            image = img_info['image']
            desc = self.describe_image(image, prompt)
            descriptions.append(desc)

        return descriptions

    def _simple_describe(self, image: Image.Image) -> str:
        """简单的图片描述（无视觉模型时使用）"""
        width, height = image.size
        mode = image.mode

        description = f"[图片尺寸: {width}x{height}, 颜色模式: {mode}]"

        # 尝试简单的图像分析
        if image.mode == 'RGB':
            try:
                # 获取图片主色调
                import collections
                img = image.copy()
                img = img.resize((50, 50))
                colors = img.getcolors(maxcolors=2500)
                if colors:
                    colors.sort(reverse=True)
                    top_colors = colors[:3]
                    description += f", 主色调: {', '.join([c[1] for c in top_colors])}"
            except:
                pass

        return description


class ImagePreprocessor:
    """图片预处理器 - 整合图片提取和描述"""

    def __init__(self, llm_provider=None, enable_vision: bool = True, vision_provider: str = None):
        """
        初始化图片预处理器

        Args:
            llm_provider: LLM提供商
            enable_vision: 是否启用视觉描述（需要视觉模型）
            vision_provider: 视觉模型提供商: "ollama" | "minimax"（可选，默认从config读取）
        """
        self.llm_provider = llm_provider
        self.enable_vision = enable_vision

        self.extractor = ImageExtractor(llm_provider)
        self.describer = ImageDescriber(llm_provider, vision_provider=vision_provider)

    def process_document(self, file_path: str, insert_descriptions: bool = True) -> List[Dict]:
        """
        处理文档中的图片

        Args:
            file_path: 文档路径
            insert_descriptions: 是否将描述插入文档内容

        Returns:
            图片信息列表
        """
        print(f"  提取图片 from {Path(file_path).name}...")

        # 提取图片
        images = self.extractor.extract_from_file(file_path)

        if not images:
            print(f"  未发现图片")
            return []

        print(f"  发现 {len(images)} 张图片")

        # 生成描述
        if self.enable_vision and self.llm_provider:
            try:
                descriptions = self.describer.describe_images_batch(images)
                for img_info, desc in zip(images, descriptions):
                    img_info['description'] = desc
            except Exception as e:
                print(f"  图片描述失败: {e}")
                for img_info in images:
                    img_info['description'] = self.describer._simple_describe(img_info['image'])
        else:
            for img_info in images:
                img_info['description'] = self.describer._simple_describe(img_info['image'])

        return images
