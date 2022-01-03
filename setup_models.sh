#!/usr/bin/env bash
mkdir pretrained_models
cd pretrained_models

# Object by object
wget -v -O Chair_models.tar.gz -L https://berkeley.box.com/shared/static/airmfckrkc9sh5xe26m0hwjsu9zw0jbg.gz
tar -xzf Chair_models.tar.gz
rm Chair_models.tar.gz


wget -v -O Table_models.tar.gz -L https://berkeley.box.com/shared/static/kb64gjzmbdb8dwfyub7rd0z0i2fyok4d.gz
tar -xzf Table_models.tar.gz
rm Table_models.tar.gz

cd ..
