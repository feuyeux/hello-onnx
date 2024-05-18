#!/bin/bash
cd "$(
    cd "$(dirname "$0")" >/dev/null 2>&1
    pwd -P
)/" || exit
set -e
#
mvn clean test -Djava.library.path="D:\\coding\\cuda\\11.8\\bin" -Dtest=HelloOrtTests#textLoadOnnx
