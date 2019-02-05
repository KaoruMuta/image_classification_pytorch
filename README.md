# image_classification_pytorch

Training and dataloading scripts in PyTorch  
By using these scripts, you can also calculate multiple AUC score and confusion matrix.  

# Details

* `data`: working directory
* `train.txt`, `test.txt`, `val.txt`: they have a path to imagefile (e.g. data/dog/dog1.jpg)
* `dataloader.py`, `trainer.py`: scripts

Please put these scripts and text files on working directory.  
And you run this following command.  
`cd data`  
`python trainer.py -t -o ./output -b 16 -e 100 -d 100 --lr 0.0001`  

If you would like to use more useful function, please customize my statement.  
Maybe it is easy to revise my code.  

# Contact

If you have any problem, please let me know.  

# Acknowledgement

Inspired by him (https://github.com/shoaibahmed), I produce the file scripts of training and dataloading in PyTorch.  
Thanks for your help.
