
## Variational Prototyping-Encoder: One-Shot Learning with Prototypical Images

IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2019

<img src="./teaser/concept_train_and_test.png" width="1000">

We tackle an open-set graphic symbol recognition problem by one-shot classification with prototypical images as a single training example for each novel class. We take an approach to learn a generalizable embedding space for novel tasks. We propose a new approach called variational prototyping-encoder (VPE) that learns the image translation task from real-world input images to their corresponding prototypical images as a meta-task. As a result, VPE learns image similarity as well as prototypical concepts which differs from widely used metric learning based approaches. Our experiments with diverse datasets demonstrate that the proposed VPE performs favorably against competing metric learning based one-shot methods. Also, our qualitative analyses show that our meta-task induces an effective embedding space suitable for unseen data representation.

### Citation
Please cite our [paper](https://arxiv.org/abs/1904.08482) in your publications if it helps your research:
    
    @InProceedings{Kim_2019_CVPR,
      author = {Kim, Junsik and Oh, Tae-Hyun and Lee, Seokju and Pan, Fei and Kweon, In So},
      title = {Variational Prototyping-Encoder: One-Shot Learning with Prototypical Images},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      month = {June},
      year = {2019}
    }

### Usage
 
This is a PyTorch implementation of "Variational Prototyping-Encoder: One-Shot Learning with Prototypical Images
" CVPR 2019.

1) Clone the repository. The default folder name is 'VPE'

    ```Shell
    git clone https://github.com/mibastro/VPE.git
    ```

2. Download the datasets used in our paper from [here](https://docs.google.com/forms/d/e/1FAIpQLSc_DSQDJMnOOu2yiTg_1OwH8NionChgvwqc7h0Fqk6FUz7NkA/viewform?usp=sf_link).
The datasets used in our paper are modified from the existing datasets.
Please cite the dataset papers if you use it for your research. ([Belgalogos](http://www-sop.inria.fr/members/Alexis.Joly/BelgaLogos/BelgaLogos.html), [FlickrLogos-32](http://www.multimedia-computing.de/flickrlogos/), [GSTRB](http://www.multimedia-computing.de/flickrlogos/), [TT100K](https://cg.cs.tsinghua.edu.cn/traffic-sign/)  )

    - Organize the file structure as below.
    ```Shell
    |__ VPE
        |__ code
        |__ db
            |__ belga
            |__ flickr32
            |__ toplogo10
            |__ GTSRB
            |__ TT100K
            |__ exp_list
    ```
    - Training and test splits are defined as text files in 'VPE/db/exp_list' folder.

3. Set the global repository path in 'VPE/code/config.json'.

4. Run main*.py to train and test the code.

### Contact
mibastro@gmail.com
