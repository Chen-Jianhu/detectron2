# SSD in Detectron2
**SSD: Single Shot MultiBox Detector**

Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg

[[`SSD`](https://github.com/weiliu89/caffe/tree/ssd)][[`arXiv`](https://arxiv.org/abs/1512.02325)][[`BibTeX`](#CitingSSD)]

<img src="http://img.cjh.zone/20201025100036.png" alt="20201025100036" width=70%>



In this repository, I implement SSD in Detectron2.


## Installation
Install Detectron2 following [the instructions](https://detectron2.readthedocs.io/tutorials/install.html).


## Training

To train a model:
```bash
cd /path/to/detectron2/projects/SSD/labs/[model_name]
python ../../train_net.py --config-file ./config.yaml
```

For example, to train model `ssd_vgg16_size300` on 8 GPUs, one should execute:
```bash
cd /path/to/detectron2/projects/SSD/labs/ssd_vgg16_size300
python ../../train_net.py --config-file ./config.yaml --num-gpus 8
```


## Evaluation

Model evaluation can be done similarly:
```bash
cd /path/to/detectron2/projects/SSD/labs/[model_name]
python ../../train_net.py --config-file ./config.yaml --eval-only MODEL.WEIGHTS /path/to/model_checkpoint
```


## Results on MS-COCO in Detectron2

|Model|AP|AP50|AP75|APs|APm|APl|download|
|-----|--|----|----|---|---|---|--------|
|[SSD300](labs/ssd_vgg16_size300)|      |      |      |      |      |      | <a href="http://det.cjh.zone/detectron2/projects/SSD/labs/ssd_vgg16_size512/output/model_final.pkl">model</a>&nbsp;\|&nbsp;<a href="http://det.cjh.zone/detectron2/projects/SSD/labs/ssd_vgg16_size512/output/metrics.json">metrics</a> |
|[SSD512](labs/ssd_vgg16_size500)|      |      |      |      |      |      | <a href="http://det.cjh.zone/detectron2/projects/SSD/labs/ssd_vgg16_size512/output/model_final.pkl">model</a>&nbsp;\|&nbsp;<a href="http://det.cjh.zone/detectron2/projects/SSD/labs/ssd_vgg16_size512/output/metrics.json">metrics</a> |
|[SSD300\*](labs/ssd_vgg16_size300_expand_aug)|      |      |      |      |      |      |<a href="http://det.cjh.zone/detectron2/projects/SSD/labs/ssd_vgg16_size512/output/model_final.pkl">model</a>&nbsp;\|&nbsp;<a href="http://det.cjh.zone/detectron2/projects/SSD/labs/ssd_vgg16_size512/output/metrics.json">metrics</a>|
|[SSD512\*](labs/ssd_vgg16_size512_expand_aug)|      |      |      |      |      |      |<a href="http://det.cjh.zone/detectron2/projects/SSD/labs/ssd_vgg16_size512/output/model_final.pkl">model</a>&nbsp;\|&nbsp;<a href="http://det.cjh.zone/detectron2/projects/SSD/labs/ssd_vgg16_size512/output/metrics.json">metrics</a>|

**Note:**

- I first train the model with 1e-3
learning rate for 160k iterations, and then continue training for 40k iterations with 1e-4 and 40k iterations with 1e-5.

- SSD300* and SSD512* are the models that are trained **with the image expansion data augmentation trick**.


## Results on MS-COCO in Paper

<img src="http://img.cjh.zone/20201106203001.png" alt="20201106203001" width=70%>


## <a name="CitingSSD"></a>Citing SSD

If you use SSD, please use the following BibTeX entry.

```
@inproceedings{liu2016ssd,
  title = {{SSD}: Single Shot MultiBox Detector},
  author = {Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
  booktitle = {ECCV},
  year = {2016}
}
```