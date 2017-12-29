import os
import logging
import datetime


LOG_LEVEL = logging.getLevelName(os.environ.get('LOG_LEVEL') or 'WARN')
LOG_DIR = 'log'

logger = logging.getLogger('statoil-iceberg')
logger.setLevel(LOG_LEVEL)
fh = logging.FileHandler(os.path.sep.join([
    LOG_DIR if os.path.isdir(LOG_DIR) and os.access(LOG_DIR, os.W_OK) else '/tmp',
    datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d')
]))
fh.setLevel(LOG_LEVEL)

ch = logging.StreamHandler()
ch.setLevel(LOG_LEVEL)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)