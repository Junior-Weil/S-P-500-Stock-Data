#include "../include/NeuralNetwork.hpp"

//Stochastic Gradient Descent
NeuralNetwork::NeuralNetwork(std::vector<uint> topology, Scalar learningRate) {
    this->topology = topology;
    this->learningRate = learningRate;

    for (uint i = 0; i < topology.size(); i++){
      // initialize neuron layer
      // Each layer in the neural network is an array of neurons, we store each of these layers as a vector such that each element in this vector stores the activation value of neuron in that layer (note that an array of these layers is the neural network itself. Now in line 13, we add an extra bias neuron to each layer except in the output layer (line 11)
      if (i == topology.size() - 1){
        neuronLayers.push_back(new RowVector(topology[i]));
      } else {
        neuronLayers.push_back(new RowVector(topology[i] + 1));
      }

      // initialize cache (sum of weighted inputs from the previous layer) and delta vectors (same dimension as neuronLayer Vec)
      // cacheLayers.push_back(new RowVector(neuronLayers.size()));
      // deltas.push_back(new RowVector(neuronLayers.size()));
      cacheLayers.push_back(new RowVector(neuronLayers.back()->size()));
      deltas.push_back(new RowVector(neuronLayers.back()->size()));

      // vector.back() gives the handle to the recently added element
      // coeffRef gives the reference of value at that place
      // (using this as we are using pointers here)
      if (i != topology.size() - 1) {
        neuronLayers.back()->coeffRef(topology[i]) = 1.0;
        cacheLayers.back()->coeffRef(topology[i]) = 1.0;
      }

      // initialize weights matrix
      if (i > 0) {
        if (i != topology.size() - 1) {
          weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
          weights.back()->setRandom();
          weights.back()->col(topology[i]).setZero();
          weights.back()->coeffRef(topology[i - 1], topology[i]) = 1.0;
        } else {
          weights.push_back(new Matrix(topology[i - 1] + 1, topology[i]));
          weights.back()->setRandom();
        }
      }
    }
  };

  Scalar activationFunction(Scalar x) {
      return tanhf(x);
  }

  Scalar activationFunctionDerivative(Scalar x) {
      return 1 - tanhf(x) * tanhf(x);
  }

  // void NeuralNetwork::propagateForward(RowVector& input) {
  //   // set the input to input layer
  //   // block returns a part of the given vector or matrix
  //   // block takes 4 args: startRow, startCol, blockRows, blockCols
  //   neuronLayers.front()->block(0,0,1, neuronLayers.front()->size() - 1) = input;

  //   for (uint i = 1; i < topology.size(); i++) {
  //     (*neuronLayers[i]) = (*neuronLayers[i-1]) * (*weights[i - 1]);
  //     neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr(&activationFunction);
  //       //[](Scalar x) {return activationFunction(x);}); // could also just pass &activationFunction instead of lambda for function pointer
  //   }
  // }

    void NeuralNetwork::propagateForward(RowVector& input) {
    // set the input to input layer
    // block returns a part of the given vector or matrix
    // block takes 4 args: startRow, startCol, blockRows, blockCols
    auto* in = neuronLayers.front();
    in->block(0,0,1, in->size() - 1) = input; // exclude bias column
    // neuronLayers.front()->block(0,0,1, neuronLayers.front()) = input;

    for (uint i = 1; i < topology.size(); i++) {
      // z = a_{i-1} * W_{i-1}
      *cacheLayers[i] = (*neuronLayers[i-1]) * (*weights[i-1]);

      // a = f(z) on the first topology[i] entries
      auto a_block = neuronLayers[i]->block(0, 0, 1, topology[i]);
      a_block = cacheLayers[i]->block(0, 0, 1, topology[i]).unaryExpr(&activationFunction);

      // re-assert bias for hiddden layers (layer before output)
      if (i != topology.size() - 1) {
        neuronLayers[i]->coeffRef(topology[i]) = 1.0; // bias neuron
      }
    }
  }

  // void NeuralNetwork::calcErrors(RowVector& output) {
  //   // calculate the errors made by neurons of last layer
  //   (*deltas.back()) = output - (*neuronLayers.back());

  //   // error calculation of hidden layers is different
  //   // begin with last hidden layer
  //   // continue till the first hidden layer
  //   for (uint i = topology.size() - 2; i > 0; i--) {
  //     (*deltas[i]) = (*deltas[i + 1]) * (weights[i]->transpose());
  //   }
  // }
  void NeuralNetwork::calcErrors(RowVector& output) {
    // L = last layer index
    uint L = topology.size();

    // delta_L = (target - a_L) * f'(z_L)
    auto aL = neuronLayers[L]->block(0, 0, 1, topology[L]);
    auto zL = cacheLayers[L]->block(0, 0, 1, topology[L]);
    deltas[L]->block(0, 0, 1, topology[L]) = (output - aL).binaryExpr(zL, [](Scalar diff, Scalar z) {return diff * activationFunctionDerivative(z);});

    // hidden layers: l = L-1 .. 1
    for (int l = (int)L - 1; l > 0; --l) {
      // backprop through W_l (drop bias column from W_l^T)
      Matrix WT = weights[l]->transpose(); // shape: (n_l+1) x (n_{l})
      RowVector next = deltas[l+1]->block(0, 0, 1, topology[l+1]);
      RowVector pre = (next * WT.block(0, 0, topology[l+1], topology[l])); // ignore bias row

      auto zl = cacheLayers[l]->block(0, 0, 1, topology[l]);
      deltas[l]->block(0, 0, 1, topology[l]) = pre.binaryExpr(zl, [](Scalar v, Scalar z) {return v * activationFunctionDerivative(z);});

      // keep bias slot as 0 (no gradient for bias activation)
      deltas[l]->coeffRef(topology[l]) = 0.0;
    }
  }

  // void NeuralNetwork::updateWeights() {
  //   // topology.size() - 1 = weights.size()
  //   for (uint i = 0; i < topology.size() - 1; i++) {
  //     //iterate over the different layers (first hidden to output layer)
  //     // if this layer is the output layer, there is no bias neuron, num of neurons specified = number of cols
  //     // if not output layer, there is a bias neuron and number of neurons specified = number of cols - 1
  //     if (i != topology.size() - 2) {
  //       for (uint c = 0; c < weights[i]->cols() - 1; c++){
  //         for (uint r = 0; r < weights[i]->rows(); r++) {
  //           weights[i]->coeffRef(r, c) += learningRate + deltas[i + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c)) + neuronLayers[i]->coeffRef(r);
  //         }
  //       }
  //     } else {
  //      for (uint c = 0; c < weights[i]->cols(); c++){
  //         for (uint r = 0; r < weights[i]->rows(); r++) {
  //           weights[i]->coeffRef(r, c) += learningRate + deltas[i + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c)) + neuronLayers[i]->coeffRef(r);

  //         }
  //       }
  //     }
  //   }
  // }

  void NeuralNetwork::updateWeights() {
    for (uint l = 0; l < topology.size() - 1; ++l) {
      // a_l includes bias; delta_{l+1} excludes bias if next lauer is output
      uint cols_next = topology[l+1]; // no bias in output layer count
      auto a_l = neuronLayers[l]->block(0, 0, 1, weights[l]->rows()); // rows = n_l + 1 when hidden
      auto d_next = neuronLayers[l+1]->block(0, 0, 1, cols_next);

      // W_l (shape (n_l+1) x (n_{l+1})) += lr * a_l^T * d_next
      *weights[l] += (a_l.transpose() * d_next) * learningRate;

      // if next layer has bias (hidden), keep last column (bias column) zero except bias-self if used
      if (l != topology.size() - 2) {
        weights[l]->col(cols_next).setZero(); // no learning into the bias neuron
        weights[l]->coeffRef(topology[l], cols_next) = 1.0; // keep bias pass-through if convention used
      }
    }
  }

  void NeuralNetwork::propagateBackward(RowVector& output) {
    calcErrors(output);
    updateWeights();
  }

  void NeuralNetwork::train(std::vector<RowVector*> input_data, std::vector<RowVector*> output_data) {
    for (uint i = 0; i < input_data.size(); i++){
      std::cout << "Input to neural network is: " << *input_data[i] << std::endl;
      propagateForward(*input_data[i]);
      std::cout << "Expected output is: " << *output_data[i] << std::endl;
      std::cout << "Output produced is: " << *neuronLayers.back() << std::endl;
      propagateBackward(*output_data[i]);
      std::cout << "MSE: " << std::sqrt((*deltas.back()).dot((*deltas.back())) / deltas.back()->size()) << std::endl;
    }
  }


// void ReadCSV(std::string filename, std::vector<RowVector*>& data) {
//   data.clear();
//   std::ifstream file(filename);
//   std::string line, word;
//   //determine number of columns in file
//   getline(file, line, '\n');
//   std::stringstream ss(line);
//   std::vector<Scalar> parsed_vec;
//   while (getline(ss, word, ',')) {
//     parsed_vec.push_back(Scalar(std::stof(&word[0])));
//   }
//   uint cols = parsed_vec.size();
//   data.push_back(new RowVector(cols));
//   for (uint i = 0; i < cols; i++) {
//     data.back()->coeffRef(1, i) = parsed_vec[i];
//   }

//   //read the file
//   if (file.is_open()) {
//     while (getline(file, line, '\n')) {
//       std::stringstream ss(line);
//       data.push_back(new RowVector(1, cols));
//       uint i = 0;
//       while (getline(ss, word, ',')) {
//         data.back()->coeffRef(i) = Scalar(std::stof(&word[0]));
//         i++;
//       }
//     }
//   }
// }

void ReadCSV(std::string filename, std::vector<RowVector*>& data) {
  data.clear();
  std::ifstream file(filename);
  std::string line;

  if (!file.is_open()) return;

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string word;
    std::vector<Scalar> vals;

    while (std::getline(ss, word, ',')) {
      vals.push_back(static_cast<Scalar>(std::stof(word)));
    }

    if (vals.empty()) continue;

    RowVector* row = new RowVector(1, vals.size());

    for (size_t i = 0; i < vals.size(); ++i) row->coeffRef(i) = vals[i];
    data.push_back(row);
  }
}
