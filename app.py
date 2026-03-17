"""
Flask Web 应用 - 测试用例生成系统
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
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
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'rag-test-case-generator'
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
        kb = get_knowledge_base()
        success = kb.build_knowledge_base(force_rebuild=True)

        if success:
            return jsonify({'status': 'success', 'message': '知识库构建成功！'})
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
    try:
        data = request.get_json()
        query = data.get('query', '').strip()

        if not query:
            return jsonify({'status': 'error', 'message': '请输入查询内容'})

        kb = get_knowledge_base()

        # 检查知识库是否已构建
        if not kb.load_knowledge_base():
            return jsonify({'status': 'error', 'message': '知识库未构建，请先构建知识库'})

        # 生成测试用例
        result = kb.query(query, return_context=True)

        return jsonify({
            'status': 'success',
            'message': '测试用例生成成功',
            'result': result
        })

    except Exception as e:
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


@app.route('/case/<path:filename>')
def view_case(filename):
    """查看测试用例内容"""
    case_path = CASES_OUTPUT_DIR / filename

    if not case_path.exists():
        return "文件不存在", 404

    with open(case_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return render_template('case.html', filename=filename, content=content)


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

        # 检查测试用例
        cases_dir = Path(CASES_OUTPUT_DIR)
        case_count = len(list(cases_dir.glob('*.md'))) if cases_dir.exists() else 0

        return jsonify({
            'status': 'success',
            'data': {
                'has_vectorstore': has_vectorstore,
                'doc_count': doc_count,
                'case_count': case_count,
                'llm_provider': LLM_PROVIDER,
                'llm_model': VOLCANO_MODEL,
                'embedding_model': EMBEDDING_MODEL
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


def allowed_file(filename):
    """检查文件类型是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
            filename = secure_filename(file.filename)
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
        doc_path = KNOWLEDGE_BASE_DIR / filename

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
