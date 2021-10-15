#! /bin/bash

cwd=$PWD

# rl forward
echo -e "\e[93m download RL model s1536_f1869509.pth \e[0m"
cd $cwd/rdpg_torch_forward
gdown --id 1E4SBpsKXjwX3q8U4gR2cBOXZ_NUIbuf0

# rl vae
echo -e "\e[93m download VAE model 0726_1557.pth \e[0m"
cd $cwd/vae
gdown --id 1B2ugYD11vKhcSiJR3loldEmOtpV3OnbQ

# rl cgan
echo -e "\e[93m download cGAN model 0827_1851.pth \e[0m"
cd $cwd/cgan
gdown --id 1KSgA1O-BKuRzav8Ew-bALgKJY490qZKO

# rl contrastive
echo -e "\e[93m download contrastive mse model mse.pth \e[0m"
cd $cwd/rdpg_torch_forward
gdown --id 1gffTWAI3hOGE09FjF0kc_of5xLprdtK1

# rl goal
echo -e "\e[93m download rl goal s0214_f435052.pth \e[0m"
cd $cwd/goal
gdown --id 1OZb9QXisx_-fLj0i0LdxDwW1Jonqvu3E

echo -e "\e[93m download rl goal d4pg \e[0m"
cd $cwd/goal
gdown --id 1COpa_Z3ZYIZFurFIxHAIsN9J9XGW7V2Q
unzip policy.zip
rm policy.zip

echo -e "\e[93m download rl goal mmwave d4pg \e[0m"
cd $cwd/goal
gdown --id 1ntwYOA0QkG9WN6sD-3tKuoBirxc2CKWo
unzip policy_mmwave.zip
rm policy_mmwave.zip

echo -e "\e[93m radar transformer cgn \e[0m"
cd $cwd/transformer
gdown --id 1DHpi4r74FgMIoWZkQqNZMn4E4EOQgxoA

echo -e "\e[93m radar transformer radar encoder \e[0m"
cd $cwd/transformer
gdown --id 1ggZ3JdnlXc8atXW77y-rQwMolOge4GjX


cd $cwd
