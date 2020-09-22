# TchAIkovsky -- Music Composition Neural Network

Transformer Neural Network for the generation of classical music performances on piano.

Currently uses PyTorch to implement this. Previously used LSTMs in Tensorflow (with not so good results)

I am using the Tensorflow Magenta MAESTRO v2 dataset which can be found [here](https://magenta.tensorflow.org/datasets/maestro#v200). 
> In previous iterations of this project it seems I completely destroyed the intended train/validation/test split. I randomly sample
> instead before beginning training.

Uses a lovely program for converting MIDI files to a more easily manipulated CSV format which can be found [here](https://www.fourmilab.ch/webtools/midicsv/)

These CSV files are then tokenized as described in [this paper](https://link.springer.com/article/10.1007/s00521-018-3758-) 
with the aim of creating expressive performances rather than robotic performance common in MIDI

I train the Transformer network ( [Attention is All You Need. Vaswani et al.](https://arxiv.org/abs/1706.03762) ) on these tokenized sequences 
like any other seq2seq NLP problem and then convert them back to MIDI files using aforementioned lovely programs.

Currently struggles to fit the dataset generally. More experimentation needed. Please be patient.
