# -*- coding: utf-8 -*-
"""
Created on Thu May  7 00:48:31 2020

@author: suhai
"""
import logging
import pandas as pd
import numpy as np
from typing import List, Any, Optional, TypeVar, Dict

T = TypeVar("T")

class Reporter:
    __instance__: Optional[logging.Logger] = None
    __fileHandlers__: Dict[str,logging.FileHandler] = {}
    __counts__ : Dict[str,int] = {}
        
    def __init__(self,log_file: str ='logfile.log'):
        if (Reporter.__instance__ is None):
            Reporter.__instance__ = logging.getLogger()
            #### which messages to report
            Reporter.__instance__.setLevel(logging.DEBUG)
            #############################
        self.formatter:logging.Formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
        self.fileId = log_file
        if (log_file not in Reporter.__fileHandlers__.keys()):
            Reporter.__fileHandlers__[log_file] = logging.FileHandler(log_file)
            Reporter.__counts__[log_file] = 1
            self.remove_handlers()
            # add file handler to logger
            Reporter.__fileHandlers__[log_file].setFormatter(self.formatter)
            Reporter.__instance__.addHandler(Reporter.__fileHandlers__[log_file])
            ## Testing logging
            self.pulse()
            ##################
        else:
            Reporter.__counts__[log_file] += 1

    def remove_handlers(self):
        Reporter.__instance__.propagate = False
        while Reporter.__instance__.handlers:
            Reporter.__instance__.handlers.pop()

    def set_correct_handler(self):
        if (Reporter.__instance__ is not None):
            self.remove_handlers()
            # add file handler to logger
            Reporter.__instance__.addHandler(
                Reporter.__fileHandlers__[self.fileId]
            )

    def pulse(self):
        self.warning("TESTING STARTED")
        self.debug('A debug message')
        self.info('An info message')
        self.warning('A warning message')
        self.error('An error has happened.')
        self.warning("TESTING ENDED")
        
    def critical(self,text: str):
        if (Reporter.__instance__ is not None):
            self.set_correct_handler()
            Reporter.__instance__.critical(text)
            
    def debug(self,text: str):
        if (Reporter.__instance__ is not None):
            self.set_correct_handler()
            Reporter.__instance__.debug(text)
            
    def info(self,text: str):
        if (Reporter.__instance__ is not None):
            self.set_correct_handler()
            Reporter.__instance__.info(text)
            
    def warning(self,text: str):
        if (Reporter.__instance__ is not None):
            self.set_correct_handler()
            Reporter.__instance__.warning(text)
            
    def error(self,text: str):
        if (Reporter.__instance__ is not None):
            self.set_correct_handler()
            Reporter.__instance__.error(text)

    def shutdown(self):
        self.warning("LOGGER SHUTDOWN\n")
        Reporter.__counts__[self.fileId] -= 1
        if (Reporter.__counts__[self.fileId] <= 0):
            del Reporter.__fileHandlers__[self.fileId]
            del Reporter.__counts__[self.fileId]
    
    def shutdownALL(self):
        for i in Reporter.__fileHandlers__:
            self.remove_handlers()
            # add file handler to logger
            Reporter.__instance__.addHandler(
                Reporter.__fileHandlers__[i]
            )
            self.warning("LOGGER SHUTDOWN\n")
        self.remove_handlers()
        logging.shutdown()
       
        
    def report_vector(self, vector: List[T],file: str,column: str):
        try:
            res: pd.DataFrame = pd.read_csv(file+'.csv')
            res = pd.concat([res,pd.DataFrame(data=np.asarray(vector),columns=[column])],axis=1)
        except:
            res = pd.DataFrame(data=np.asarray(vector),columns=[column])
            
        res.to_csv(file+'.csv',index=False)
