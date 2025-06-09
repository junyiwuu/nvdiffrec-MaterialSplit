import logging
import os
import sys
from tabulate import tabulate

from torch.utils.tensorboard import SummaryWriter
import omegaconf
from omegaconf import OmegaConf


def config_logging(cfg_logging, out_dir = None):
    file_level = cfg_logging.get("file_level", 10)
    console_level = cfg_logging.get("console_level", 20)

    log_formatter = logging.Formatter(cfg_logging["format"]) # 这个字段是必须项，所以不用 .get(),防止silent failure

    root_logger = logging.getLogger()  # 拿到logging最顶层的控制器
    root_logger.handlers.clear()  # clear all handler, like FileHandler, StreamHandler

    root_logger.setLevel(min(file_level, console_level))

    # set file handler
    if out_dir is not None:
        _logging_file = os.path.join(
            out_dir, cfg_logging.get("filename", "logging.log") # default log filename is loggig.log
        )

        file_handler = logging.FileHandler(_logging_file)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(file_level)
        root_logger.addHandler(file_handler)

    # set terminal handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(console_level)
    root_logger.addHandler(console_handler)

    # avoid pollution by packages, they might have different logging level
    logging.getLogger("PIL").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)



class MyTrainingLogger: 
    '''tensorboard'''
    writer: SummaryWriter    # class level , just a type hint , not an assignment, not initialize anything
    # self.writer = None
    is_initialized = False

    def __init__(self) -> None:
        pass


    # initialized the tb logger
    def set_dir(self, tb_log_dir):
        # initialize the writer, and set the flag
        if self.is_initialized:
            raise ValueError("Do not initialize writter twice")
        self.writer = SummaryWriter(tb_log_dir)
        self.is_initialized = True

    def log_dic(self, scalar_dic, global_step, walltime=None):
        for k, v in scalar_dic.items():
            self.writer.add_scalar(k, v, global_step=global_step, walltime=walltime)
            # walltime: actual timestamp 
        return
    
    def log_img(self, img, global_step):
        # self.writer.add_image("val/predic", img_opt, global_step=global_step)
        self.writer.add_image("val/predict_vs_ref", img, global_step=global_step)
        return
    



# global instance
tb_logger = MyTrainingLogger()



def eval_dict_to_text(val_metrics: dict, dataset_name: str, sample_list_path: str):
    # 写进evaluation metricxs里面的
    eval_text = f"Evaluation metrics: \n\
        on dataset: {dataset_name}\n\
        over samples in : {sample_list_path}\n "

    eval_text += tabulate([val_metrics.keys(), val_metrics.values()])
    return eval_text  # 得到后直接写进文件的






def recursive_load_config(config_path: str) -> OmegaConf:
    conf = OmegaConf.load(config_path)

    output_conf = OmegaConf.create({})

    # Load base config. Later configs on the list will overwrite previous
    base_configs = conf.get("base_config", default_value=None)
    if base_configs is not None:
        assert isinstance(base_configs, omegaconf.listconfig.ListConfig)
        for _path in base_configs:
            assert (
                _path != config_path
            ), "Circulate merging, base_config should not include itself."
            _base_conf = recursive_load_config(_path)
            output_conf = OmegaConf.merge(output_conf, _base_conf)

    # Merge configs and overwrite values
    output_conf = OmegaConf.merge(output_conf, conf)

    return output_conf

