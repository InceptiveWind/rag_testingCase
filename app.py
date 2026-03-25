"""
Flask Web 应用 - 测试用例生成系统
"""

import os
import sys
import secrets
from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path
from werkzeug.utils import secure_filename

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_base import KnowledgeBase, create_llm_provider
from config import (
    KNOWLEDGE_BASE_DIR,
    VECTOR_STORE_DIR,
    CASES_OUTPUT_DIR,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    LLM_PROVIDER,
    VOLCANO_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    MAX_TOKENS,
    ENABLE_PREPROCESSOR,
    ENABLE_LLM_TAG,
    USE_QUERY_REWRITE,
)

app = Flask(__name__)
# 禁用模板缓存，确保每次都重新加载
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 最大50MB

# 支持的文件类型
ALLOWED_EXTENSIONS = {'txt', 'md', 'markdown', 'pdf', 'docx', 'xlsx', 'xls', 'pptx', 'ppt', 'csv', 'json', 'xmind', 'vsdx'}

# 初始化知识库
kb = None

def get_knowledge_base():
    """获取或初始化知识库"""
    global kb
    if kb is None:
        kb = KnowledgeBase()
    return kb


@app.route('/')
def index():
    """首页"""
    return render_template('index.html', config={
        'knowledge_base_dir': str(KNOWLEDGE_BASE_DIR),
        'cases_output_dir': str(CASES_OUTPUT_DIR),
        'llm_model': VOLCANO_MODEL
    })


@app.route('/build', methods=['POST'])
def build_kb():
    """构建知识库"""
    try:
        # 获取 rebuild 参数
        rebuild = request.args.get('rebuild', 'false').lower() == 'true'
        force_rebuild = rebuild  # true = 全量构建, false = 增量构建

        kb = get_knowledge_base()
        mode = "全量" if force_rebuild else "增量"
        print(f"开始{mode}构建知识库...")

        success = kb.build_knowledge_base(force_rebuild=force_rebuild)

        if success:
            return jsonify({'status': 'success', 'message': f'知识库{mode}构建成功！'})
        else:
            return jsonify({'status': 'error', 'message': '知识库构建失败，请检查文档目录'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/check', methods=['POST'])
def check_llm():
    """检查LLM连接"""
    try:
        kb = get_knowledge_base()
        if kb.check_llm_connection():
            return jsonify({'status': 'success', 'message': 'LLM服务连接正常'})
        else:
            return jsonify({'status': 'error', 'message': 'LLM服务连接失败'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/generate', methods=['POST'])
def generate():
    """生成测试用例"""
    print("=" * 50)
    print("开始处理 /generate 请求")
    try:
        data = request.get_json()
        print(f"接收到的数据: {data}")

        if not data:
            return jsonify({'status': 'error', 'message': '无效的请求数据'})

        query = data.get('query', '').strip()
        num_cases = int(data.get('num_cases', 10))
        version = data.get('version', '').strip()  # 版本过滤参数

        print(f"query: {query}, num_cases: {num_cases}, version: {version}")

        if not query:
            return jsonify({'status': 'error', 'message': '请输入查询内容'})

        print("获取知识库...")
        kb = get_knowledge_base()
        print(f"知识库对象: {kb}")

        # 检查知识库是否已构建
        load_result = kb.load_knowledge_base()
        print(f"load_knowledge_base 结果: {load_result}")

        if not load_result:
            return jsonify({'status': 'error', 'message': '知识库未构建，请先构建知识库'})

        print(f"开始生成测试用例，num_cases={num_cases}...")
        # 生成测试用例（examples从配置文件中读取）
        try:
            # 根据配置决定是否使用查询改写
            if USE_QUERY_REWRITE:
                # 使用查询改写+多路召回的AdvancedRetriever
                result = kb.query_with_rewrite(query, return_context=True, num_cases=num_cases, version=version)
            else:
                # 使用普通检索
                result = kb.query(query, return_context=True, num_cases=num_cases, version=version)
        except Exception as query_error:
            print(f"kb.query 发生错误: {query_error}")
            import traceback
            traceback.print_exc()
            return jsonify({'status': 'error', 'message': f'生成测试用例失败: {str(query_error)}'})

        print(f"生成完成，结果类型: {type(result)}")

        # 确保 result 是可序列化的字典 - 使用安全转换
        result_data = {}
        try:
            if isinstance(result, dict):
                content_val = result.get('content')
                filepath_val = result.get('filepath')

                # 如果是列表（用例数组），直接传递；否则转为字符串
                if isinstance(content_val, list):
                    content_to_send = content_val
                elif content_val is not None:
                    content_to_send = str(content_val)
                else:
                    content_to_send = ''

                filepath_str = ''
                if filepath_val is not None:
                    filepath_str = str(filepath_val)

                result_data = {
                    'content': content_to_send,
                    'filepath': filepath_str
                }
            else:
                # 其他情况转为字符串
                result_data = {
                    'content': str(result) if result else '',
                    'filepath': ''
                }
        except Exception as e:
            print(f"序列化错误: {e}")
            result_data = {
                'content': str(result),
                'filepath': ''
            }

        content_len = len(result_data.get('content', ''))
        print(f"返回数据 content 长度: {content_len}")
        print("=" * 50)

        # 确保返回的JSON包含正确的字段
        return jsonify({
            'status': 'success',
            'message': '测试用例生成成功',
            'data': {
                'content': result_data['content'],
                'filepath': result_data['filepath']
            }
        })

    except Exception as e:
        import traceback
        print(f"生成失败: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/cases')
def list_cases():
    """查看生成的测试用例"""
    cases_dir = Path(CASES_OUTPUT_DIR)

    if not cases_dir.exists():
        return jsonify({'status': 'error', 'message': '测试用例目录不存在'})

    cases = []
    for f in cases_dir.glob('*.md'):
        cases.append({
            'name': f.name,
            'path': str(f),
            'modified': f.stat().st_mtime
        })

    # 按修改时间排序
    cases.sort(key=lambda x: x['modified'], reverse=True)

    return jsonify({'status': 'success', 'cases': cases})



@app.route('/download/<filename>')
def download_case(filename):
    """下载测试用例文件"""
    case_path = CASES_OUTPUT_DIR / filename

    if not case_path.exists():
        return "文件不存在", 404

    # 根据文件类型设置MIME类型
    if filename.endswith('.xlsx'):
        mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    elif filename.endswith('.xls'):
        mimetype = 'application/vnd.ms-excel'
    elif filename.endswith('.md'):
        mimetype = 'text/markdown'
    else:
        mimetype = 'application/octet-stream'

    return send_from_directory(
        CASES_OUTPUT_DIR,
        filename,
        as_attachment=True,
        mimetype=mimetype
    )


@app.route('/status')
def status():
    """获取系统状态"""
    try:
        # 检查向量存储
        vector_dir = Path(VECTOR_STORE_DIR)
        has_vectorstore = vector_dir.exists() and any(vector_dir.iterdir())

        # 检查文档
        docs_dir = Path(KNOWLEDGE_BASE_DIR)
        doc_count = len(list(docs_dir.rglob('*.*'))) if docs_dir.exists() else 0

        # 检查已入库的文档数量（从增量构建状态文件读取）
        vector_doc_count = 0
        state_file = VECTOR_STORE_DIR / ".file_state.json"
        if state_file.exists():
            try:
                import json
                with open(state_file, 'r', encoding='utf-8') as f:
                    file_states = json.load(f)
                    vector_doc_count = len(file_states)
            except:
                pass

        # 判断构建状态
        # 1. 向量库没有任何文档 -> 未构建
        # 2. 已入库文档数 < 文档目录文档数 -> 部分构建
        # 3. 已入库文档数 >= 文档目录文档数 -> 已构建
        if vector_doc_count == 0:
            build_status = '未构建'
        elif vector_doc_count < doc_count:
            build_status = '部分构建'
        else:
            build_status = '已构建'

        return jsonify({
            'status': 'success',
            'data': {
                'has_vectorstore': has_vectorstore,
                'doc_count': doc_count,
                'vector_doc_count': vector_doc_count,
                'build_status': build_status,
                'llm_provider': LLM_PROVIDER,
                'llm_model': VOLCANO_MODEL,
                'embedding_model': EMBEDDING_MODEL
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/versions')
def list_versions():
    """获取可用的知识库版本列表"""
    try:
        kb = get_knowledge_base()
        if not kb.load_knowledge_base():
            return jsonify({'status': 'success', 'versions': []})

        # 获取所有版本
        versions = kb.retriever.get_all_versions()

        return jsonify({
            'status': 'success',
            'versions': versions
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


def allowed_file(filename):
    """检查文件类型是否允许"""
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower() if filename.rsplit('.', 1)[1] else ''
    return ext in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    """上传知识库文档"""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': '没有选择文件'})

        files = request.files.getlist('file')

        if not files or all(f.filename == '' for f in files):
            return jsonify({'status': 'error', 'message': '没有选择文件'})

        uploaded_count = 0
        uploaded_files = []

        for file in files:
            if file.filename == '':
                continue

            if not allowed_file(file.filename):
                return jsonify({'status': 'error', 'message': f'不支持的文件类型: {file.filename}'})

            # 保存文件到知识库目录
            original_filename = file.filename

            # 检查是否包含中文字符（包含中文则保留原始文件名）
            def has_chinese(s):
                return any('\u4e00' <= c <= '\u9fff' for c in s)

            if has_chinese(original_filename):
                # 中文文件名直接保留原始名称
                filename = original_filename
            else:
                filename = secure_filename(original_filename)
                # 如果 secure_filename 处理后文件名为空或没有扩展名，保留原始文件名
                if not filename or '.' not in filename:
                    # 尝试从原始文件名中提取扩展名并生成有效文件名
                    if '.' in original_filename:
                        ext = original_filename.rsplit('.', 1)[-1].lower()
                        import uuid
                        filename = f"uploaded_file_{uuid.uuid4().hex[:8]}.{ext}"
                    else:
                        filename = original_filename

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
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/documents')
def list_documents():
    """列出知识库中的文档"""
    try:
        docs_dir = Path(KNOWLEDGE_BASE_DIR)

        if not docs_dir.exists():
            return jsonify({'status': 'success', 'documents': []})

        documents = []
        for f in docs_dir.rglob('*'):
            if f.is_file():
                documents.append({
                    'name': f.name,
                    'path': str(f.relative_to(docs_dir)),
                    'size': f.stat().st_size,
                    'modified': f.stat().st_mtime
                })

        # 按修改时间排序
        documents.sort(key=lambda x: x['modified'], reverse=True)

        return jsonify({'status': 'success', 'documents': documents})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/documents/<path:filename>', methods=['DELETE'])
def delete_document(filename):
    """删除知识库文档"""
    try:
        # 安全检查：防止路径遍历攻击
        # 解析路径并确保其在 KNOWLEDGE_BASE_DIR 内
        doc_path = (KNOWLEDGE_BASE_DIR / filename).resolve()

        # 验证路径以 KNOWLEDGE_BASE_DIR 开头，防止 .. 遍历
        if not str(doc_path).startswith(str(KNOWLEDGE_BASE_DIR.resolve())):
            return jsonify({'status': 'error', 'message': '非法文件路径'})

        if not doc_path.exists():
            return jsonify({'status': 'error', 'message': '文件不存在'})

        doc_path.unlink()

        return jsonify({'status': 'success', 'message': '文件已删除'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    # 确保必要的目录存在
    Path(KNOWLEDGE_BASE_DIR).mkdir(parents=True, exist_ok=True)
    Path(VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)
    Path(CASES_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("测试用例生成系统 - Web版")
    print("=" * 50)
    print(f"访问地址: http://localhost:5000")
    print(f"知识库目录: {KNOWLEDGE_BASE_DIR}")
    print(f"用例输出目录: {CASES_OUTPUT_DIR}")
    print("=" * 50)

    app.run(host='0.0.0.0', port=5000, debug=True)
