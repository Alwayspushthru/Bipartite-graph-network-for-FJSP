import os
import time
from torch.utils.tensorboard import SummaryWriter


def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0 and int(os.environ.get("LOCAL_RANK", "0")) == 0


class TBLogger:
    def __init__(self, log_dir="runs", run_name="exp", enabled=True):
        self.enabled = enabled and is_main_process()
        self.writer = None
        self.log_dir = None
        if self.enabled:
            timestamp = time.strftime("%Y-%m-%d_%H%M", time.localtime())
            run_name = run_name or "exp"
            self.log_dir = os.path.join(log_dir, run_name, timestamp)
            self.writer = SummaryWriter(self.log_dir)

    def add_scalar(self, tag, value, global_step=None, update_step=None):
        if not self.writer:
            return
        step = global_step if global_step is not None else update_step
        if step is None:
            self.writer.add_scalar(tag, value)
        else:
            self.writer.add_scalar(tag, value, step)

    def add_histogram(self, tag, values, global_step=None, update_step=None):
        if not self.writer:
            return
        step = global_step if global_step is not None else update_step
        if step is None:
            self.writer.add_histogram(tag, values)
        else:
            self.writer.add_histogram(tag, values, step)

    def flush(self):
        if self.writer:
            self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.close()