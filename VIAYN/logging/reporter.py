# -*- coding: utf-8 -*-
"""
Created on Thu May  7 00:48:31 2020

@author: suhai
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Any, Optional
class Reporter:
    __instance__: Optional[logging.RootLogger] = None
    def __init__(self,log_file: str ='logfile.log'):
        if (Reporter.__instance__ is None):
            Reporter.__instance__ = logging.getLogger()
        #### which messages to report
        Reporter.__instance__.setLevel(logging.DEBUG)
        #############################
        self.file_handler:logging.FileHandler = logging.FileHandler(log_file)
        self.formatter:logging.Formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
        self.file_handler.setFormatter(self.formatter)
        self.remove_handlers()
        # add file handler to logger
        Reporter.__instance__.addHandler(self.file_handler)
        ## Testing logging
        self.pulse()
        ##################

    def remove_handlers(self):
        Reporter.__instance__.propagate = False
        while Reporter.__instance__.handlers:
            Reporter.__instance__.handlers.pop()
            
    def pulse(self):
        self.warning("TESTING STARTED")
        self.debug('A debug message')
        self.info('An info message')
        self.warning('A warning message')
        self.error('An error has happened.')
        self.warning("TESTING ENDED")
        
    def critical(self,text: str):
        if (Reporter.__instance__ is not None):
            Reporter.__instance__.critical(text)
    def debug(self,text: str):
        if (Reporter.__instance__ is not None):
            Reporter.__instance__.debug(text)
    def info(self,text: str):
        if (Reporter.__instance__ is not None):
            Reporter.__instance__.info(text)
    def warning(self,text: str):
        if (Reporter.__instance__ is not None):
            Reporter.__instance__.warning(text)
    def error(self,text: str):
        if (Reporter.__instance__ is not None):
            Reporter.__instance__.error(text)
        
    def shutdown(self):
        self.warning("LOGGER SHUTDOWN\n\n")
        self.remove_handlers()
        logging.shutdown()
        
    def report_vector(self,vector: List[Any],file: str,column: str):
        append: bool
        try:
            res = pd.read_csv(file+'.csv')
            append = True
        except:
            res = pd.DataFrame(data=np.asarray(vector),columns=[column])
            append = False

        if (append):
            res = pd.concat([res,pd.DataFrame(data=np.asarray(vector),columns=[column])],axis=1)
        res.to_csv(file+'.csv',index=False)
