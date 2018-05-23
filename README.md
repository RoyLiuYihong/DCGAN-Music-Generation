# DCGAN for Music Generation

Quick and dirty experiment for music generation with piano roll using deep convolutional generative adversarial network.


## Instructions

- Dataset:
    Create a directory for your dataset in dataset/ like dataset/mymidi/.

- Preprocessing
    ```shell
    python3 preprocess.py mymidi # generates dataset/mymidi.pkl
    ```

- Training and generating
    ```shell
    ipython3
    ```

    In IPython:

    ```python
    run main

    # Train on dataset/mymidi.pkl with learning ratio
    train('dataset/mymidi.pkl', ratio=(1, 0.1))

    save('train.sess')
    load('train.sess')

    # Generate output/all.png and output/all.mid
    generate(save_png=True, save_mid=True)
    ```
