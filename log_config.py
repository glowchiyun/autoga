"""
日志配置工具模块
统一管理AutoML系统的日志输出
"""
import logging
import os
import sys
from contextlib import contextmanager

# 全局配置
VERBOSE_MODE = os.environ.get('AUTOML_VERBOSE', 'FALSE').upper() == 'TRUE'
SILENT_MODE = os.environ.get('AUTOML_SILENT', 'FALSE').upper() == 'TRUE'
LOG_LEVEL = logging.DEBUG if VERBOSE_MODE else (logging.WARNING if SILENT_MODE else logging.INFO)

def setup_logging(log_file='autoML.log', level=None):
    """
    设置日志配置
    
    Parameters:
    -----------
    log_file : str
        日志文件路径
    level : int
        日志级别，如果为None则使用环境变量控制
    """
    if level is None:
        level = LOG_LEVEL
    
    # 清除现有的handlers
    logger = logging.getLogger()
    logger.handlers.clear()
    
    # 文件handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    file_handler.setLevel(level)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # 控制台handler（只在非静默模式下）
    if not SILENT_MODE:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)  # 控制台只显示警告和错误
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    logger.addHandler(file_handler)
    logger.setLevel(level)
    
    return logger

@contextmanager
def suppress_print():
    """上下文管理器：抑制print输出"""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr

@contextmanager  
def log_level(level):
    """
    临时改变日志级别的上下文管理器
    
    Example:
    --------
    with log_level(logging.WARNING):
        # 这里的代码只会输出WARNING及以上级别的日志
        some_noisy_function()
    """
    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(level)
    try:
        yield logger
    finally:
        logger.setLevel(old_level)

def log_progress(current, total, prefix='Progress', length=50):
    """
    输出进度条到日志
    
    Parameters:
    -----------
    current : int
        当前进度
    total : int
        总数
    prefix : str
        前缀文本
    length : int
        进度条长度
    """
    percent = 100 * (current / float(total))
    filled = int(length * current // total)
    bar = '█' * filled + '-' * (length - filled)
    
    if current == total or current % max(1, total // 10) == 0:
        logging.info(f'{prefix}: |{bar}| {percent:.1f}% ({current}/{total})')

class ProgressLogger:
    """进度日志记录器"""
    
    def __init__(self, total, prefix='Progress', log_interval=10):
        """
        初始化进度记录器
        
        Parameters:
        -----------
        total : int
            总任务数
        prefix : str
            日志前缀
        log_interval : int
            每多少个任务记录一次日志（百分比）
        """
        self.total = total
        self.current = 0
        self.prefix = prefix
        self.log_interval = max(1, total * log_interval // 100)
        self.start_time = None
        
    def start(self):
        """开始计时"""
        import time
        self.start_time = time.time()
        logging.info(f"{self.prefix}: 开始 (总计 {self.total} 个任务)")
        
    def update(self, n=1):
        """更新进度"""
        self.current += n
        if self.current % self.log_interval == 0 or self.current == self.total:
            percent = 100 * self.current / self.total
            
            if self.start_time:
                import time
                elapsed = time.time() - self.start_time
                if self.current > 0:
                    eta = elapsed * (self.total - self.current) / self.current
                    logging.info(f"{self.prefix}: {percent:.1f}% ({self.current}/{self.total}) "
                               f"已用时: {elapsed:.1f}s, 预计剩余: {eta:.1f}s")
                else:
                    logging.info(f"{self.prefix}: {percent:.1f}% ({self.current}/{self.total})")
            else:
                logging.info(f"{self.prefix}: {percent:.1f}% ({self.current}/{self.total})")
    
    def finish(self):
        """完成"""
        if self.start_time:
            import time
            elapsed = time.time() - self.start_time
            logging.info(f"{self.prefix}: 完成! 总用时: {elapsed:.1f}s")
        else:
            logging.info(f"{self.prefix}: 完成!")

# 使用说明字符串
USAGE_GUIDE = """
AutoML日志控制使用说明
====================

1. 设置日志模式（通过环境变量）:

   详细模式（输出所有DEBUG级别日志）:
   Windows CMD:    set AUTOML_VERBOSE=TRUE
   Windows PS:     $env:AUTOML_VERBOSE="TRUE"
   Linux/Mac:      export AUTOML_VERBOSE=TRUE

   静默模式（只输出WARNING和ERROR）:
   Windows CMD:    set AUTOML_SILENT=TRUE
   Windows PS:     $env:AUTOML_SILENT="TRUE"
   Linux/Mac:      export AUTOML_SILENT=TRUE

   普通模式（默认，输出INFO级别）:
   不设置环境变量或设置为FALSE

2. 在代码中使用:

   from log_config import setup_logging, log_level, suppress_print
   
   # 初始化日志
   setup_logging('my_log.log')
   
   # 临时改变日志级别
   with log_level(logging.WARNING):
       noisy_function()
   
   # 抑制print输出
   with suppress_print():
       function_with_prints()

3. 查看日志:
   日志文件: autoML.log
   控制台: 只显示WARNING和ERROR（非静默模式）
"""

if __name__ == "__main__":
    print(USAGE_GUIDE)
