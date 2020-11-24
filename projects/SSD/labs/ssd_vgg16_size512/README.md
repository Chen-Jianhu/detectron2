# ssd_vgg16_size512  

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.272
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.468
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.283
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.104
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.309
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.414
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.252
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.382
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.153
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.453
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.584
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 27.226 | 46.772 | 28.323 | 10.367 | 30.878 | 41.359 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 36.797 | bicycle      | 19.195 | car            | 28.289 |  
| motorcycle    | 30.246 | airplane     | 45.709 | bus            | 54.186 |  
| train         | 50.873 | truck        | 26.476 | boat           | 12.934 |  
| traffic light | 12.151 | fire hydrant | 47.670 | stop sign      | 51.819 |  
| parking meter | 32.671 | bench        | 14.867 | bird           | 21.387 |  
| cat           | 51.482 | dog          | 47.770 | horse          | 40.002 |  
| sheep         | 34.469 | cow          | 38.832 | elephant       | 50.566 |  
| bear          | 55.931 | zebra        | 52.162 | giraffe        | 52.102 |  
| backpack      | 6.305  | umbrella     | 24.110 | handbag        | 5.565  |  
| tie           | 14.763 | suitcase     | 18.587 | frisbee        | 40.026 |  
| skis          | 10.550 | snowboard    | 20.125 | sports ball    | 22.325 |  
| kite          | 21.359 | baseball bat | 15.109 | baseball glove | 20.404 |  
| skateboard    | 32.687 | surfboard    | 20.886 | tennis racket  | 29.305 |  
| bottle        | 17.888 | wine glass   | 17.706 | cup            | 27.344 |  
| fork          | 14.352 | knife        | 5.966  | spoon          | 5.747  |  
| bowl          | 29.134 | banana       | 14.181 | apple          | 13.178 |  
| sandwich      | 28.547 | orange       | 21.496 | broccoli       | 17.090 |  
| carrot        | 11.468 | hot dog      | 18.178 | pizza          | 39.629 |  
| donut         | 29.204 | cake         | 23.838 | chair          | 16.476 |  
| couch         | 32.365 | potted plant | 14.502 | bed            | 35.344 |  
| dining table  | 21.444 | toilet       | 45.661 | tv             | 44.994 |  
| laptop        | 46.173 | mouse        | 45.905 | remote         | 12.716 |  
| keyboard      | 33.671 | cell phone   | 21.197 | microwave      | 45.615 |  
| oven          | 26.176 | toaster      | 11.764 | sink           | 21.707 |  
| refrigerator  | 36.724 | book         | 5.653  | clock          | 36.676 |  
| vase          | 21.897 | scissors     | 17.470 | teddy bear     | 32.227 |  
| hair drier    | 1.485  | toothbrush   | 4.619  |                |        |
