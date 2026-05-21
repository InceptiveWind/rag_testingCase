"""
Flask Web 应用 - 测试用例生成系统
提供知识库管理、测试用例生成功能
"""

import os
import sys
import secrets
import traceback
from typing import Any, Dict, List, Tuple
from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path
from werkzeug.utils import secure_filename

# ========== 监控 nul 文件的创建 ==========
import builtins
import time
original_open = builtins.open

def monitored_open(file, *args, **kwargs):
    # 检查文件名是否包含 'nul'
    filename = str(file)
    if 'nul' in filename.lower() or filename.lower() == 'nul':
        print(f"\n{'='*60}")
        print(f"!!! 警告：检测到尝试打开可能是 nul 的文件！")
        print(f"文件名: {filename}")
        print(f"时间: {time.ctime()}")
        print(f"参数: args={args}, kwargs={kwargs}")
        print(f"\n调用堆栈:")
        print('-'*60)
        traceback.print_stack()
        print(f"{'='*60}\n")
    
    return original_open(file, *args, **kwargs)

# 替换内置的 open 函数
builtins.open = monitored_open

# 也监控 Path 的 write_text 和 write_bytes
original_path_write_text = Path.write_text
original_path_write_bytes = Path.write_bytes

def monitored_write_text(self, *args, **kwargs):
    filename = str(self)
    if 'nul' in filename.lower() or filename.lower() == 'nul':
        print(f"\n{'='*60}")
        print(f"!!! 警告：检测到尝试写入可能是 nul 的文件！")
        print(f"文件名: {filename}")
        print(f"时间: {time.ctime()}")
        print(f"\n调用堆栈:")
        print('-'*60)
        traceback.print_stack()
        print(f"{'='*60}\n")
    return original_path_write_text(self, *args, **kwargs)

def monitored_write_bytes(self, *args, **kwargs):
    filename = str(self)
    if 'nul' in filename.lower() or filename.lower() == 'nul':
        print(f"\n{'='*60}")
        print(f"!!! 警告：检测到尝试写入可能是 nul 的文件！")
        print(f"文件名: {filename}")
        print(f"时间: {time.ctime()}")
        print(f"\n调用堆栈:")
        print('-'*60)
        traceback.print_stack()
        print(f"{'='*60}\n")
    return original_path_write_bytes(self, *args, **kwargs)

Path.write_text = monitored_write_text
Path.write_bytes = monitored_write_bytes

print("nul 文件监控已启动...")
# ========================================

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_base import KnowledgeBase, create_llm_provider
from utils import has_chinese
from config import (
    KNOWLEDGE_BASE_DIR,
    VECTOR_STORE_DIR,
    CASES_OUTPUT_DIR,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    LLM_PROVIDER,
    MINIMAX_MODEL,
    ARK_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    MAX_TOKENS,
    ENABLE_PREPROCESSOR,
    ENABLE_LLM_TAG,
    USE_QUERY_REWRITE,
    MAX_CONTENT_LENGTH,
    DEFAULT_PORT,
    DEFAULT_HOST,
)

app = Flask(__name__)
# 禁用模板缓存，确保每次都重新加载
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 支持的文件类型
ALLOWED_EXTENSIONS = {'txt', 'md', 'markdown', 'pdf', 'docx', 'xlsx', 'xls', 'pptx', 'ppt', 'csv', 'json', 'xmind', 'vsdx'}

# 初始化知识库
kb = None


def create_error_response(message: str, status_code: int = 200) -> Tuple[Dict[str, Any], int]:
    """
    创建标准化的错误响应
    
    Args:
        message: 错误消息
        status_code: HTTP状态码
        
    Returns:
        响应元组
    """
    return jsonify({'status': 'error', 'message': message}), status_code


def create_success_response(data: Any = None, message: str = '成功') -> Dict[str, Any]:
    """
    创建标准化的成功响应
    
    Args:
        data: 响应数据
        message: 成功消息
        
    Returns:
        响应字典
    """
    result = {'status': 'success', 'message': message}
    if data is not None:
        result['data'] = data
    return result


def log_exception(prefix: str = "Error") -> None:
    """
    记录异常堆栈信息
    
    Args:
        prefix: 日志前缀
    """
    print(f"[{prefix}] Exception occurred:")
    traceback.print_exc()

def get_knowledge_base() -> KnowledgeBase:
    """获取或初始化知识库
    
    Returns:
        KnowledgeBase 实例
    """
    global kb
    if kb is None:
        kb = KnowledgeBase()
    return kb


@app.route('/')
def index():
    """首页 - 渲染主页面"""
    try:
        llm_model = ARK_MODEL if LLM_PROVIDER == "volcano" else MINIMAX_MODEL
        return render_template('index.html', config={
            'knowledge_base_dir': str(KNOWLEDGE_BASE_DIR),
            'cases_output_dir': str(CASES_OUTPUT_DIR),
            'llm_model': llm_model
        })
    except Exception as e:
        log_exception("Index page")
        return f"页面加载失败: {str(e)}", 500


@app.route('/build', methods=['POST'])
def build_kb():
    """构建知识库
    
    Query Parameters:
        rebuild: 是否强制全量重建 ('true'/'false')
    
    Returns:
        JSON 响应
    """
    try:
        # 获取 rebuild 参数
        rebuild = request.args.get('rebuild', 'false').lower() == 'true'
        force_rebuild = rebuild  # true = 全量构建, false = 增量构建

        kb = get_knowledge_base()
        mode = "全量" if force_rebuild else "增量"
        print(f"开始{mode}构建知识库...")

        success = kb.build_knowledge_base(force_rebuild=force_rebuild)

        if success:
            return create_success_response(message=f'知识库{mode}构建成功！')
        else:
            return create_error_response('知识库构建失败，请检查文档目录')
    except Exception as e:
        log_exception("Build knowledge base")
        return create_error_response(f'构建失败: {str(e)}')


@app.route('/check', methods=['POST'])
def check_llm():
    """检查LLM连接
    
    Returns:
        JSON 响应
    """
    try:
        kb = get_knowledge_base()
        if kb.check_llm_connection():
            return create_success_response(message='LLM服务连接正常')
        else:
            return create_error_response('LLM服务连接失败')
    except Exception as e:
        log_exception("Check LLM connection")
        return create_error_response(f'检查失败: {str(e)}')


def serialize_result(result: Any) -> Dict[str, str]:
    """
    安全序列化查询结果
    
    Args:
        result: 原始查询结果
        
    Returns:
        序列化后的字典
    """
    try:
        if isinstance(result, dict):
            content_val = result.get('content')
            filepath_val = result.get('filepath')
            
            # 处理内容
            if isinstance(content_val, list):
                content_to_send = content_val
            elif content_val is not None:
                content_to_send = str(content_val)
            else:
                content_to_send = ''
                
            # 处理文件路径
            filepath_str = str(filepath_val) if filepath_val is not None else ''
            
            return {'content': content_to_send, 'filepath': filepath_str}
        else:
            # 其他情况转为字符串
            return {'content': str(result) if result else '', 'filepath': ''}
    except Exception as e:
        print(f"[Result serialization] Error: {e}")
        return {'content': str(result) if result else '', 'filepath': ''}


@app.route('/generate', methods=['POST'])
def generate():
    """生成测试用例
    
    Request JSON:
        query: 查询内容
        num_cases: 生成用例数量（默认10）
        version: 版本过滤参数
        
    Returns:
        JSON 响应
    """
    print("=" * 50)
    print("开始处理 /generate 请求")
    try:
        data = request.get_json()
        print(f"接收到的数据: {data}")

        if not data:
            return create_error_response('无效的请求数据')

        query = data.get('query', '').strip()
        num_cases = int(data.get('num_cases', 10))
        version = data.get('version', '').strip()

        print(f"query: {query}, num_cases: {num_cases}, version: {version}")

        if not query:
            return create_error_response('请输入查询内容')

        # 获取知识库并检查状态
        kb = get_knowledge_base()
        if not kb.load_knowledge_base():
            return create_error_response('知识库未构建，请先构建知识库')

        print(f"开始生成测试用例，num_cases={num_cases}...")
        
        # 生成测试用例
        try:
            if USE_QUERY_REWRITE:
                result = kb.query_with_rewrite(query, return_context=True, num_cases=num_cases, version=version)
            else:
                result = kb.query(query, return_context=True, num_cases=num_cases, version=version)
        except Exception as query_error:
            log_exception("Query execution")
            return create_error_response(f'生成测试用例失败: {str(query_error)}')

        print(f"生成完成，结果类型: {type(result)}")

        # 序列化结果
        result_data = serialize_result(result)
        
        content_len = len(result_data.get('content', ''))
        print(f"返回数据 content 长度: {content_len}")
        print("=" * 50)

        return create_success_response(data=result_data, message='测试用例生成成功')

    except Exception as e:
        log_exception("Generate test cases")
        return create_error_response(f'生成失败: {str(e)}')


@app.route('/cases')
def list_cases():
    """查看生成的测试用例
    
    Returns:
        JSON 响应，包含用例列表
    """
    try:
        cases_dir = Path(CASES_OUTPUT_DIR)

        if not cases_dir.exists():
            return jsonify({'status': 'success', 'cases': []})

        cases: List[Dict[str, Any]] = []
        for f in cases_dir.glob('*.md'):
            cases.append({
                'name': f.name,
                'path': str(f),
                'modified': f.stat().st_mtime
            })

        # 按修改时间排序
        cases.sort(key=lambda x: x['modified'], reverse=True)

        return jsonify({'status': 'success', 'cases': cases})
    except Exception as e:
        log_exception("List cases")
        return create_error_response(f'获取用例列表失败: {str(e)}')


def get_mime_type(filename: str) -> str:
    """
    根据文件扩展名获取 MIME 类型
    
    Args:
        filename: 文件名
        
    Returns:
        MIME 类型字符串
    """
    if filename.endswith('.xlsx'):
        return 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    elif filename.endswith('.xls'):
        return 'application/vnd.ms-excel'
    elif filename.endswith('.md'):
        return 'text/markdown'
    else:
        return 'application/octet-stream'


@app.route('/download/<filename>')
def download_case(filename: str):
    """下载测试用例文件
    
    Args:
        filename: 文件名
        
    Returns:
        文件下载响应或 404
    """
    try:
        case_path = CASES_OUTPUT_DIR / filename

        if not case_path.exists():
            return "文件不存在", 404

        mimetype = get_mime_type(filename)

        return send_from_directory(
            CASES_OUTPUT_DIR,
            filename,
            as_attachment=True,
            mimetype=mimetype
        )
    except Exception as e:
        log_exception("Download case")
        return f"下载失败: {str(e)}", 500


def load_file_states():
    """加载文件构建状态
    
    Returns:
        dict: 文件路径 -> 状态信息的字典
    """
    state_file = VECTOR_STORE_DIR / ".file_state.json"
    if not state_file.exists():
        return {}
    try:
        import json
        with open(state_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def get_supported_document_extensions():
    """获取支持的文档扩展名列表（不依赖完整 langchain 导入）
    
    Returns:
        list: 扩展名列表，包括点前缀，如 ['.pdf', '.docx']
    """
    return [
        '.txt', '.md', '.markdown', '.pdf', '.docx', '.xlsx', '.xls', '.pptx', '.ppt', 
        '.csv', '.json', '.xmind', '.vsdx'
    ]


@app.route('/status')
def status():
    """获取系统状态
    
    Returns:
        JSON 响应，包含系统状态信息
    """
    try:
        # 检查向量存储
        vector_dir = Path(VECTOR_STORE_DIR)
        has_vectorstore = vector_dir.exists() and any(vector_dir.iterdir())

        # 检查文档 - 只统计支持类型的文件
        docs_dir = Path(KNOWLEDGE_BASE_DIR)
        supported_exts = get_supported_document_extensions()
        supported_files = []
        if docs_dir.exists():
            for ext in supported_exts:
                supported_files.extend(docs_dir.rglob(f'*{ext}'))
        doc_count = len(supported_files)

        # 检查已入库的文档数量
        file_states = load_file_states()
        vector_doc_count = len(file_states)

        # 判断构建状态
        if vector_doc_count == 0:
            build_status = '未构建'
        elif vector_doc_count < doc_count:
            build_status = '部分构建'
        else:
            build_status = '已构建'

        llm_model = ARK_MODEL if LLM_PROVIDER == "volcano" else MINIMAX_MODEL

        return jsonify({
            'status': 'success',
            'data': {
                'has_vectorstore': has_vectorstore,
                'doc_count': doc_count,
                'vector_doc_count': vector_doc_count,
                'build_status': build_status,
                'llm_provider': LLM_PROVIDER,
                'llm_model': llm_model,
                'embedding_model': EMBEDDING_MODEL
            }
        })
    except Exception as e:
        log_exception("Get system status")
        return create_error_response(f'获取状态失败: {str(e)}')


@app.route('/versions')
def list_versions():
    """获取可用的知识库版本列表
    
    Returns:
        JSON 响应，包含版本列表
    """
    try:
        kb = get_knowledge_base()
        if not kb.load_knowledge_base():
            return jsonify({'status': 'success', 'versions': []})

        versions = kb.retriever.get_all_versions()
        return jsonify({'status': 'success', 'versions': versions})
    except Exception as e:
        log_exception("List versions")
        return create_error_response(f'获取版本列表失败: {str(e)}')


def allowed_file(filename: str) -> bool:
    """检查文件类型是否允许
    
    Args:
        filename: 文件名
        
    Returns:
        是否允许
    """
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[-1].lower() if filename.rsplit('.', 1)[-1] else ''
    return ext in ALLOWED_EXTENSIONS


import uuid  # 提前导入 uuid 模块


def generate_safe_filename(original_filename: str) -> str:
    """
    生成安全的文件名
    
    Args:
        original_filename: 原始文件名
        
    Returns:
        安全的文件名
    """
    # 包含中文字符则保留原始文件名
    if has_chinese(original_filename):
        return original_filename
        
    filename = secure_filename(original_filename)
    
    # 如果 secure_filename 处理后文件名为空或没有扩展名，生成新文件名
    if not filename or '.' not in filename:
        if '.' in original_filename:
            ext = original_filename.rsplit('.', 1)[-1].lower()
            filename = f"uploaded_file_{uuid.uuid4().hex[:8]}.{ext}"
        else:
            filename = original_filename
            
    return filename


@app.route('/upload', methods=['POST'])
def upload_file():
    """上传知识库文档
    
    Returns:
        JSON 响应
    """
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': '没有选择文件'})

        files = request.files.getlist('file')

        if not files or all(f.filename == '' for f in files):
            return jsonify({'status': 'error', 'message': '没有选择文件'})

        uploaded_count = 0
        uploaded_files: List[str] = []

        for file in files:
            if file.filename == '':
                continue

            if not allowed_file(file.filename):
                return jsonify({'status': 'error', 'message': f'不支持的文件类型: {file.filename}'})

            # 生成安全的文件名
            original_filename = file.filename
            filename = generate_safe_filename(original_filename)

            # 保存文件
            file_path = KNOWLEDGE_BASE_DIR / filename
            file.save(str(file_path))

            uploaded_count += 1
            uploaded_files.append(filename)

        return jsonify({
            'status': 'success',
            'message': f'成功上传 {uploaded_count} 个文件',
            'files': uploaded_files
        })

    except Exception as e:
        log_exception("Upload file")
        return create_error_response(f'上传失败: {str(e)}')


@app.route('/documents')
def list_documents():
    """列出知识库中的文档
    
    Returns:
        JSON 响应，包含文档列表和每个文件的构建状态
    """
    try:
        docs_dir = Path(KNOWLEDGE_BASE_DIR)

        if not docs_dir.exists():
            return jsonify({'status': 'success', 'documents': []})

        # 加载已构建文件状态
        file_states = load_file_states()
        supported_exts = get_supported_document_extensions()

        documents: List[Dict[str, Any]] = []
        for f in docs_dir.rglob('*'):
            if f.is_file():
                # 检查是否是支持的文档类型
                ext = f.suffix.lower()
                is_supported = ext in supported_exts
                file_key = str(f.resolve())
                is_built = file_key in file_states
                
                documents.append({
                    'name': f.name,
                    'path': str(f.relative_to(docs_dir)),
                    'size': f.stat().st_size,
                    'modified': f.stat().st_mtime,
                    'is_supported': is_supported,
                    'is_built': is_built
                })

        # 按修改时间排序
        documents.sort(key=lambda x: x['modified'], reverse=True)

        return jsonify({'status': 'success', 'documents': documents})
    except Exception as e:
        log_exception("List documents")
        return create_error_response(f'获取文档列表失败: {str(e)}')


@app.route('/documents/<path:filename>', methods=['DELETE'])
def delete_document(filename: str):
    """删除知识库文档（同时从向量库中删除）
    
    Args:
        filename: 文件名或路径
        
    Returns:
        JSON 响应
    """
    try:
        # 安全检查：防止路径遍历攻击
        doc_path = (KNOWLEDGE_BASE_DIR / filename).resolve()

        # 验证路径以 KNOWLEDGE_BASE_DIR 开头
        if not str(doc_path).startswith(str(KNOWLEDGE_BASE_DIR.resolve())):
            return create_error_response('非法文件路径')

        if not doc_path.exists():
            return create_error_response('文件不存在')

        # 从向量库删除
        kb = get_knowledge_base()
        deleted_chunks = 0
        if kb.load_knowledge_base():
            deleted_chunks = kb.delete_document(str(doc_path))

        # 从文件系统删除
        doc_path.unlink()

        message = f'文件已删除'
        if deleted_chunks > 0:
            message += f'，同时从向量库中删除了 {deleted_chunks} 个文档块'

        return create_success_response(message=message)
    except Exception as e:
        log_exception("Delete document")
        return create_error_response(f'删除失败: {str(e)}')


def ensure_directories() -> None:
    """确保必要的目录存在"""
    Path(KNOWLEDGE_BASE_DIR).mkdir(parents=True, exist_ok=True)
    Path(VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)
    Path(CASES_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    ensure_directories()

    print("=" * 50)
    print("测试用例生成系统 - Web版")
    print("=" * 50)
    print(f"访问地址: http://{DEFAULT_HOST}:{DEFAULT_PORT}")
    print(f"知识库目录: {KNOWLEDGE_BASE_DIR}")
    print(f"用例输出目录: {CASES_OUTPUT_DIR}")
    print("=" * 50)

    app.run(host=DEFAULT_HOST, port=DEFAULT_PORT, debug=True)
