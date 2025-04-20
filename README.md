# ViT-Chess-Position_Analysis

This project explores the use of Vision Transformers (ViTs) for classifying individual chess pieces from images of chessboard squares. Unlike traditional approaches that rely on convolutional neural networks (CNNs) to directly predict board states, this method introduces a preprocessing pipeline that segments full-board images into square-level inputs, enabling finegrained classification. The model is trained on a diverse dataset of synthetically generated chess positions featuring various board and piece styles. Results show that the ViT architecture achieves high classification accuracy while requiring less training data and time, thanks to its superior generalization capabilities. Additionally, attention maps offer valuable interpretability into the model’s decision-making process.

<table>
  <thead>
    <tr>
      <th style="border: 1px solid; text-align: center;">Epochs</th>
      <th style="border: 1px solid; text-align: center;">Training Set Size</th>
      <th style="border: 1px solid; text-align: center;">Accuracy(%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid; text-align: center;">10</td>
      <td style="border: 1px solid; text-align: center;">10</td>
      <td style="border: 1px solid; text-align: center;">83.71</td>
    </tr>
    <tr>
      <td style="border: 1px solid; text-align: center;">10</td>
      <td style="border: 1px solid; text-align: center;">50</td>
      <td style="border: 1px solid; text-align: center;">88.32</td>
    </tr>
    <tr>
      <td style="border: 1px solid; text-align: center;">10</td>
      <td style="border: 1px solid; text-align: center;">100</td>
      <td style="border: 1px solid; text-align: center;">97.91</td>
    </tr>
    <tr>
      <td style="border: 1px solid; text-align: center;">10</td>
      <td style="border: 1px solid; text-align: center;">500</td>
      <td style="border: 1px solid; text-align: center;">99.92</td>
    </tr>
    <tr>
      <td style="border: 1px solid; text-align: center;">10</td>
      <td style="border: 1px solid; text-align: center;">1000</td>
      <td style="border: 1px solid; text-align: center;">99.95</td>
    </tr>
  </tbody>
</table>


Dataset used can be found here: https://www.kaggle.com/datasets/koryakinp/chess-positions
