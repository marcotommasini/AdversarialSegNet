# Analysis of loss function impact on Real-time Domain Adaptation in Semantic Segmentation

This project aims at clarifying what is the impact
of the loss function in the accuracy of a Real-time Domain
Adaptation in Semantic Segmentation model, using adversarial
networks, and with class imbalance. By comparing three different
loss types and comparing the results, it became clear that using
different functions to calculate the loss while training produces
severely different analysis.

To run this project:
1. Download all of it's files
2. Open the training.ipynb file and change the directory parameter to your installation folder
3. If you want to train the model or validade a checkpoint, you need to set the "Analysis mode" argument to False.
4. To train the model without the use of a checkpoint you need to set the "Use Checkpoints" parameter on the Main() to "False".
5. If something happened to your execution and you want to use the checkpoint, set the "Use Checkpoints" parameter on the Main() to "True".
6. When calling the Main() you can set lists with different parameters in order to train the network in different modes automatically.
7. This project was meant to run on a GPU, running it on the CPU will not be possible.
8. The checkpoints are saved in the checkpoint folder.
9. The results of the validations are saved in .txt files inside the validations folder.
