#!/usr/bin/env bash
mkdir semantic_srn_data
cd semantic_srn_data

# Object by object
wget -v -O Chair.tar.gz -L https://berkeley.box.com/shared/static/2mphopu9eqld2ve160dzeluon57uwvhs.gz
tar -xzf Chair.tar.gz
rm Chair.tar.gz


wget -v -O Table.tar.gz -L https://berkeley.box.com/shared/static/mmz6odslcw3my6aeo8o7d2xtxpl0s1qw.gz
tar -xzf Table.tar.gz
rm Table.tar.gz

cd ..
