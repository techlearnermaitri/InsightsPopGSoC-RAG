import logging

def setup_logger(name="InsightsPopRAG"):
    logger=logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch=logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter=logging.Formatter('[%(asctime)s] [%(levelname)s] -- [%(message)s]')
    ch.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(ch)   
        
        return logger
    
logger=setup_logger()

# Test logs removed for cleaner deployment