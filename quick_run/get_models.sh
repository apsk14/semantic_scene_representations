#!/usr/bin/env bash
mkdir update_linear_30/
cd update_linear_30/
wget -v -O epoch_2500_iter_005000.pth -L https://berkeley.box.com/shared/static/k99rr72u8if9ggkvq0j66loospyhcg3j

cd ..
mkdir test_vanilla_50
cd test_vanilla_50/

wget -v -O epoch_0026_iter_080000.pth -L https://berkeley.box.com/shared/static/gmhbwowj449hxho27gbzsnfca2itku4s

cd ..
