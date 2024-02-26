# Analysis of line charts through CNNs
This repository contains the source code for the implementation of a neural network model to achieve multiple tasks over line chart images. In particular, 3 tasks have been seeked:
- Localizing legend
- Localizing axis
- Counting the number of lines

### Frameworks and languages
The whole project has been implemented locally with `Python 3.8`, the [`Keras/TensorFlow2`](https://www.tensorflow.org/) platfrom for preprocessing, building and training the model, [`Matplotlib`](https://matplotlib.org/) library for the visualization of the dataset and the usage of [`Google Colab`](https://colab.research.google.com/) for speeding up the process.

### Dataset
The study has been conducted over the images contained inside the [`FigureSeer`](https://prior.allenai.org/projects/figureseer) dataset, and contained 927 line chart images of different size, each one characterized by a legend and a number of lines between 2 and 10.

### Description of the study
As a first study, a custom model for the task of legend localization has been implemented in [`legend_localization_custom_model.ipynb`](https://github.com/LeoGori/line_chart_analysis/blob/main/legend_localization_custom_model.ipynb) with a standard preprocessing. Secondly, the same custom model has been compared with vgg16 pre-trained model, with the presence of a custom preprocessing in [`legend_localization_model.ipynb`](https://github.com/LeoGori/line_chart_analysis/blob/main/legend_localization_model.ipynb) . The fine tuning approach outperformed the custom model and therefore it has been selected for the rest of the study.

Notebooks [`2_task_vgg_fine_tuning.ipynb`](https://github.com/LeoGori/line_chart_analysis/blob/main/2_task_vgg_fine_tuning.ipynb) and [`2_task_resnet_fine_tuning.ipynb` ](https://github.com/LeoGori/line_chart_analysis/blob/main/2_task_resnet_fine_tuning.ipynb) implement the mulitask models for the localization of legend and axes over the line graph images, respectively fine tuning pre-trained model vgg16 and ResNet50. Such models have similar performance, however vgg16 showed less overfitting behavior, therefore it has been chosen for the rest of the study.

Notebook [`get_num_lines_from_legend.ipynb`](https://github.com/LeoGori/line_chart_analysis/blob/main/get_num_lines_from_legend.ipynb) implements the line counting task through the vgg16 fine-tuned model, by processing the legend-only images (cropped out by the original dataset). This approach led to interesting results, however imperfect in the case line entries inside the legend were placed horizontally.

Notebook [`3_task_fine_tuning_model.ipynb`](https://github.com/LeoGori/line_chart_analysis/blob/main/3_task_fine_tuning_model.ipynb) imlements the model that predicts over all the aforementioned tasks. This approach, however, showed an overfitting behavior for the line counting task, which evidently degraded the performance for the legend localization task.

### Results
Being the localization task a bounding-box regression problem, the accuracy of each model has been analyzed based on the number of predictions that reported satisfying values of IoU metric (which describes the amount of groundtruth area the prediction fulfills), over the whole sets. The gatehred results show the performance of the vgg16 fine-tuned model, for each of the afore-mentioned scenarios.

Legend localization:
>    | IoU  | Training set accuracy | Validation set accuracy | Test set accuracy |
>    |------|-----------------------|-------------------------|-------------------|
>    | >0.6 | 95,27%                | 82,55%                  | 86,02%            |
>    | >0.7 | 88,01%                | 71,81%                  | 71,51%            |
>    | >0.8 | 68,58%                | 42,95%                  | 44,09%            |


Legend and axis localization:
>| Task                | IoU  | Training set accuracy | Validation set accuracy | Test set accuracy |
>|---------------------|------|-----------------------|-------------------------|-------------------|
>|                     | >0.6 | 93,58%                | 74,50%                  | 84,95%            |
>| Legend localization | >0.7 | 85,30%                | 60,40%                  | 68,28%            |
>|                     | >0.8 | 61,66%                | 32,21%                  | 37,10%            |
>|                     | >0.6 | 98,14%                | 97,32%                  | 97,31%            |
>| Axis localization   | >0.7 | 97,80%                | 91,95%                  | 96,77%            |
>|                     | >0.8 | 94,09%                | 83,89%                  | 90,32%            |

Legend-axis localization and line counting:
>| Task                | criterion | Training set accuracy | Validation set accuracy | Test set accuracy |
>|---------------------|-----------|-----------------------|-------------------------|-------------------|
>|                     | IoU>0.6   | 73,31%                | 52,35%                  | 57,53%            |
>| Legend localization | IoU>0.7   | 46,96%                | 26,85%                  | 33,33%            |
>|                     | IoU>0.8   | 17,06%                | 8,05%                   | 12,37%            |
>|                     | IoU>0.6   | 97,97%                | 96,64%                  | 97,85%            |
>| Axis localization   | IoU>0.7   | 95,78%                | 92,62%                  | 94,09%            |
>|                     | IoU>0.8   | 91,05%                | 81,88%                  | 83,87%            |
>| Line counting       | None      | 100%                  | 74,50%                  | 71,51%            |