"""
增量构建管理器 - 记录文件状态，实现增量加载
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime


class IncrementalBuilder:
    """增量构建管理器"""

    def __init__(self, state_file: str = "vector_store/.file_state.json"):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.file_states: Dict[str, dict] = {}
        self._load_state()

    def _load_state(self):
        """加载状态文件"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    self.file_states = json.load(f)
            except:
                self.file_states = {}

    def _save_state(self):
        """保存状态文件"""
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.file_states, f, ensure_ascii=False, indent=2)

    def _get_file_hash(self, file_path: Path) -> str:
        """计算文件哈希（使用修改时间+大小+内容前1KB）"""
        stat = file_path.stat()
        key = f"{file_path.stat().st_mtime}_{file_path.stat().st_size}"

        # 读取文件前1KB内容用于计算哈希
        try:
            with open(file_path, 'rb') as f:
                content = f.read(1024)
                key += hashlib.md5(content).hexdigest()
        except:
            pass

        return hashlib.md5(key.encode()).hexdigest()

    def get_changed_files(self, file_paths: List[Path], enable_image_processing: bool = None) -> List[Path]:
        """
        获取需要处理的文件列表（新增或修改的文件）

        Args:
            file_paths: 所有文件路径
            enable_image_processing: 当前图片处理配置，用于判断是否需要强制重新处理

        Returns:
            需要处理的文件列表
        """
        changed_files = []
        all_known_files: Set[str] = set(self.file_states.keys())

        for file_path in file_paths:
            file_key = str(file_path)

            # 文件不存在于记录中，需要处理
            if file_key not in all_known_files:
                changed_files.append(file_path)
                continue

            # 检查文件是否已修改
            current_hash = self._get_file_hash(file_path)
            stored_hash = self.file_states[file_key].get('hash', '')

            if current_hash != stored_hash:
                changed_files.append(file_path)
                continue

            # 文件内容没变，检查图片处理配置是否变化
            # 逻辑：只有当 enable_image_processing=true 且历史为 false 时才强制重新处理
            if enable_image_processing is not None:
                stored_image_processed = self.file_states[file_key].get('image_processed', False)
                if enable_image_processing and not stored_image_processed:
                    # enable_image_processing 从 false 变为 true，需要强制重新处理
                    changed_files.append(file_path)

        return changed_files

    def mark_processed(self, file_paths: List[Path], enable_image_processing: bool = None):
        """标记文件已处理

        Args:
            file_paths: 需要标记的文件列表
            enable_image_processing: 当前图片处理配置
        """
        for file_path in file_paths:
            file_key = str(file_path)
            self.file_states[file_key] = {
                'hash': self._get_file_hash(file_path),
                'processed_at': datetime.now().isoformat(),
                'size': file_path.stat().st_size,
                'image_processed': enable_image_processing if enable_image_processing is not None else False
            }
        self._save_state()

    def clear(self):
        """清除所有状态"""
        self.file_states = {}
        self._save_state()

