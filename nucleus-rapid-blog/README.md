# Nucleus_Rapid_Blog
This is the link to download the Imagenet validation data: https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

This is the command for standard file extraction and formatting:

```
cd ../ && mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```
Environment can be created with:
```
conda create --name <env_name> --file requirements.txt python=3.7.10
```
