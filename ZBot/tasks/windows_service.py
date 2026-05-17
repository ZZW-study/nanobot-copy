import win32serviceutil
import win32service
import win32event
import subprocess
import sys
import time
from pathlib import Path

class CeleryWorkerService(win32serviceutil.ServiceFramework):
    # 服务基本信息
    _svc_name_ = "AgentCeleryWorker"
    _svc_display_name_ = "Agent Celery Worker"
    _svc_description_ = "Agent 记忆系统 Celery 后台服务（Worker+Beat），开机自启执行定时任务"

    def __init__(self, args):
        """初始化 Windows 服务句柄和子进程状态。"""
        super().__init__(args)
        # 停止事件
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        # 进程句柄
        self.worker_process = None
        self.beat_process = None
        
        # 使用 Path 类处理路径
        self.service_file_path = Path(__file__).resolve()
        # 从 ZBot/tasks/windows_service.py 往上找2层，到 ZBOT 根目录
        self.project_root = self.service_file_path.parent.parent.parent
        # 日志目录
        self.log_dir = self.project_root / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def SvcDoRun(self):
        """⚠️ Windows服务的启动入口（固定写法，不能改名！）"""
        self.ReportServiceStatus(win32service.SERVICE_START_PENDING)
        
        # 关键：把项目根目录加入Python路径，确保能import ZBot包
        project_root_str = str(self.project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)

        # 获取当前Python解释器路径（避免虚拟环境问题）
        python_exe = sys.executable

        try:
            # 1. 启动 Celery Worker（执行任务的进程）
            self.worker_process = subprocess.Popen(
                [
                    python_exe, "-m", "celery",
                    "-A", "ZBot.tasks.celery_app",
                    "worker",
                    "--loglevel=info",
                    "--pool=solo",
                    "--logfile", str(self.log_dir / "worker.log"),
                    "--pidfile", str(self.log_dir / "worker.pid")
                ],
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW
            )

            # 2. 启动 Celery Beat（定时任务触发器）
            self.beat_process = subprocess.Popen(
                [
                    python_exe, "-m", "celery",
                    "-A", "ZBot.tasks.celery_app",
                    "beat",
                    "--loglevel=info",
                    "--logfile", str(self.log_dir / "beat.log"),
                    "--pidfile", str(self.log_dir / "beat.pid"),
                    "--schedule", str(self.log_dir / "beat-schedule.db")
                ],
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW
            )

            # 标记服务为运行中
            self.ReportServiceStatus(win32service.SERVICE_RUNNING)
            
            # 循环等待停止信号，同时检查子进程是否异常退出
            while True:
                wait_result = win32event.WaitForSingleObject(self.stop_event, 1000)
                if wait_result == win32event.WAIT_OBJECT_0:
                    break
                if (self.worker_process.poll() is not None) or (self.beat_process.poll() is not None):
                    break

        except Exception as e:
            # 服务启动异常，写入日志
            error_log = self.log_dir / "service_error.log"
            with open(error_log, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 服务启动异常：{str(e)}\n")
        finally:
            self._stop_all_processes()

    def SvcStop(self):
        """Windows服务的停止入口（固定写法）"""
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.stop_event)
        self._stop_all_processes()
        self.ReportServiceStatus(win32service.SERVICE_STOPPED)

    def _stop_all_processes(self):
        """安全停止Celery子进程，避免残留"""
        for proc in [self.worker_process, self.beat_process]:
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()

if __name__ == "__main__":
    win32serviceutil.HandleCommandLine(CeleryWorkerService)
