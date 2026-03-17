"""
RAG测试用例生成器 - 主程序入口
"""

import argparse
import sys
import os

# 设置环境变量解决可能的兼容性问题
os.environ['TOKENIZERS_PARALLELISM'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

# 强制刷新输出
import functools
print = functools.partial(print, flush=True)

from config import KNOWLEDGE_BASE_DIR, VECTOR_STORE_DIR, CASES_OUTPUT_DIR
from knowledge_base import KnowledgeBase


def main():
    parser = argparse.ArgumentParser(
        description="RAG测试用例生成器 - 基于知识库自动生成集成测试用例"
    )

    parser.add_argument(
        '--build',
        action='store_true',
        help='构建知识库'
    )
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='强制重建知识库'
    )
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='查询并生成测试用例'
    )
    parser.add_argument(
        '--docs',
        type=str,
        default=str(KNOWLEDGE_BASE_DIR),
        help='知识库文档目录'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='检查Ollama连接状态'
    )

    args = parser.parse_args()

    # 初始化知识库
    kb = KnowledgeBase()

    # 检查LLM连接
    if args.check:
        if kb.check_llm_connection():
            print("[OK] LLM服务连接正常")
        else:
            print("[X] LLM服务连接失败，请检查配置")
            sys.exit(1)
        return

    # 构建知识库
    if args.build:
        print("开始构建知识库...")
        success = kb.build_knowledge_base(force_rebuild=args.rebuild)

        if success:
            print("[OK] 知识库构建成功")
        else:
            print("[X] 知识库构建失败")
            sys.exit(1)
        return

    # 生成测试用例
    if args.query:
        print(f"正在生成测试用例: {args.query}")
        print("-" * 50)

        try:
            result = kb.query(args.query)
            print("\n" + "=" * 50)
            print("生成的测试用例:")
            print("=" * 50)
            print(result['content'])
            print("\n" + "=" * 50)
            print(f"测试用例已保存到: {result['filepath']}")
        except Exception as e:
            print(f"生成测试用例失败: {e}")
            sys.exit(1)
        return

    # 默认显示帮助
    parser.print_help()
    print("\n" + "=" * 50)
    print("使用示例:")
    print("  python main.py --check              # 检查Ollama连接")
    print("  python main.py --build              # 构建知识库")
    print("  python main.py -q \"用户登录测试\"  # 生成测试用例")
    print("=" * 50)


if __name__ == '__main__':
    main()
