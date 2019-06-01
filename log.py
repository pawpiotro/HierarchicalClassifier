import logging
import logging.config


# Logowanie
def getLogger(name):
    log_file_path = './logs/%s.log' % name
    logging.config.fileConfig('logs/conf/logging.conf',
                              defaults={'logfilename':
                                        log_file_path})
    logger = logging.getLogger(name)
    return logger
