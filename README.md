# ImageClassifier With Tensowflow

Developing an Image Classifier with Deep Learning using Python


1. In the Jupyter notebook the model is built.
2. predict.py is an application to use the model.
  
### Basic usage:

  `$ python predict.py ./test_images/orchid.jpg my_model.h5`

### Options:

* Return the top 3 most likely classes:
  `$ python predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3`

* Use a label_map.json file to map labels to flower names:
  `$ python predict.py ./test_images/orchid.jpg my_model.h5 --category_names label_map.json`
