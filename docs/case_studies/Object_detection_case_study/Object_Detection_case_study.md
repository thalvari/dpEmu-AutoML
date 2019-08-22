# Object Detection case study

## Comparison of models from FaceBook's Detectron-project and YOLOv3

We compared the performance of models from FaceBook's Detectron project and YOLOv3 model from Joseph Redmon, when different error sources were added. The models from FaceBook's Detectron project were FasterRCNN, MaskRCNN and RetinaNet.

## Data

We used 118 287 jpg images (COCO train2017) to train the models. 5000 images (COCO val2017) were used to calculate the mAP-50 scores.

Detectron's model zoo had pretrained weights for FasterRCNN, MaskRCNN and RetinaNet. YOLOv3's weights were trained by us, using the Kale cluster of University of Helsinki. The training took approximately five days when two NVIDIA Tesla V100 GPUs were used. 

## Error types (Filters) used in the case study

* Gaussian blur

* Added rain

* Added snow

* JPEG compression

* Resolution change

### Gaussian blur filter

The error parameter here is the standard deviation (std) for the Gaussian distribution.

#### Example images using the filter:

##### std: 0.0

![std 0.0](Blur_Gaussian/20190729-150653-727543.jpg)

##### std: 1.0

![std 1.0](Blur_Gaussian/20190729-150700-771777.jpg)

##### std: 2.0

![std 0.0](Blur_Gaussian/20190729-150707-503684.jpg)

##### std: 3.0

![std 1.0](Blur_Gaussian/20190729-150714-401435.jpg)

#### The results of Gaussian Blur filter

![Gaussian Blur](../../../results/object_detection/gaussian_blur_filter/20190807-230431-155708.png)

### Rain filter

The error parameter here is the probability of rain.

#### Example images using the filter:

##### probability: 0.0001

![probability 10^-4](Rain/20190729-151307-080828.jpg)

##### probability: 0.001

![probability 10^-3](Rain/20190729-151314-483299.jpg)

##### probability: 0.01

![probability 10^-2](Rain/20190729-151323-269028.jpg)

##### probability: 0.1

![probability 10^-1](Rain/20190729-151330-649152.jpg)

#### The results of Rain filter 

![Rain](../../../results/object_detection/rain_filter/20190806-173029-848262.png)

### Snow filter

The error parameter here is the probability of snow. The other parameters had static values as follows: 
"snowflake_alpha": 0.4, "snowstorm_alpha": 0

#### Example images using the filter:

##### probability: 0.0001

![probability 10^-4](Snow/20190729-151434-149765.jpg)

##### probability: 0.001

![probability 10^-3](Snow/20190729-151443-736282.jpg)

##### probability: 0.01

![probability 10^-2](Snow/20190729-151452-361038.jpg)

##### probability: 0.1

![probability 10^-1](Snow/20190729-151507-952953.jpg)

#### The results of Snow filter

![Snow](../../../results/object_detection/snow_filter/20190807-035949-375428.png)

### JPEG Compression

The error parameter here is the quality of JPEG-compression. The higher the value, the better quality the picture has.

#### Example images using the filter:

##### quality: 10

![quality 10](JPEG_Compression/20190729-150821-361183.jpg)

##### quality: 20

![quality 20](JPEG_Compression/20190729-150831-366993.jpg)

##### quality: 30

![quality 30](JPEG_Compression/20190729-150839-587541.jpg)

##### quality: 100

![quality 100](JPEG_Compression/20190729-150847-940301.jpg)

#### The results of JPEG Compression filter

![JPEG Compression](../../../results/object_detection/jpeg_compression/20190806-035613-845958.png)

### Resolution

The error parameter makes the resolution k times smaller.

#### Example images using the filter:

##### Value of k: 1

![k 1](Resolution/20190729-151611-205148.jpg)

##### Value of k: 2

![k 2](Resolution/20190729-151621-167993.jpg)

##### Value of k: 3

![k 3](Resolution/20190729-151630-067637.jpg)

##### Value of k: 4

![k 4](Resolution/20190729-151639-036737.jpg)

#### The results of Resolution filter

![Resolution](../../../results/object_detection/reduced_resolution/20190808-032055-393558.png)