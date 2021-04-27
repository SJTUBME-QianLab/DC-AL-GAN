# DC-AL-GAN
This repository holds the TensorFlow code for the paper

**DC-AL GAN: Pseudoprogression and true tumor progression of glioblastomamultiform image classiﬁcation based on DCGAN and AlexNet** 

All the materials released in this library can ONLY be used for RESEARCH purposes and not for commercial use.

# Author List
Meiyu Li, Hailiang Tang, Michael D. Chan, Xiaobo Zhou, Xiaohua Qian*

# Abstract
Purpose: Pseudoprogression (PsP) occurs in 20–30% of patients with glioblastoma multiforme (GBM) after receiving the standard treatment. PsP exhibits similarities in shape and intensity to the true tumor progression (TTP) of GBM on the follow‐up magnetic resonance imaging (MRI). These similarities pose challenges to the differentiation of these types of progression and hence the selection of the appropriate clinical treatment strategy.

Methods: To address this challenge, we introduced a novel feature learning method based on deep convolutional generative adversarial network (DCGAN) and AlexNet, termed DC‐AL GAN, to discriminate between PsP and TTP in MRI images. Due to the adversarial relationship between the generator and the discriminator of DCGAN, high‐level discriminative features of PsP and TTP can be derived for the discriminator with AlexNet. We also constructed a multifeature selection module to concatenate features from different layers, contributing to more powerful features used for effectively discriminating between PsP and TTP. Finally, these discriminative features from the discriminator are used for classification by a support vector machine (SVM). Tenfold cross‐validation (CV) and the area under the receiver operating characteristic (AUC) were applied to evaluate the performance of this developed algorithm.

Results: The accuracy and AUC of DC‐AL GAN for discriminating PsP and TTP after tenfold CV were 0.920 and 0.947. We also assessed the effects of different indicators (such as sensitivity and specificity) for features extracted from different layers to obtain a model with the best classification performance.

Conclusions: The proposed model DC‐AL GAN is capable of learning discriminative representations from GBM datasets, and it achieves desirable PsP and TTP classification performance superior to other state‐of‐the‐art methods. Therefore, the developed model would be useful in the diagnosis of PsP and TTP for GBM.

# Required
Our code is based on **Python**.

# Citation
Please cite the following paper if you use this repository in your research.
```
@inproceedings{
  title     = {DC‐AL GAN: Pseudoprogression and true tumor progression of glioblastoma multiform image classification based on DCGAN and AlexNet},
  author    = {Meiyu Li, Hailiang Tang, Michael D. Chan, Xiaobo Zhou, Xiaohua Qian*},
  journal   = {Medical Physics},
  month     = {December}，
  year      = {2019},
}
```

# Contact
For any question, feel free to contact
```
Xiaohua Qian: xiaohua.qian@sjtu.edu.cn
```
