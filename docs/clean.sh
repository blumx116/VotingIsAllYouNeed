#!/bin/bash

cd source
shopt -s extglob
rm -v !("conf.py"|"index.rst")
shopt -u extglob
cd ..