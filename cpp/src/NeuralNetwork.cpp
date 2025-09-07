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
      cacheLayers.push_back(new RowVector(neuronLayers.size()));
      deltas.push_back(new RowVector(neuronLayers.size()));

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

void NeuralNetwork::propagateForward(RowVector& input) {
  // set the input to input layer
  // block returns a part of the given vector or matrix
  // block takes 4 args: startRow, startCol, blockRows, blockCols
  neuronLayers.front()->block(0,0,1, neuronLayers.front()->size() - 1) = input;

  for (uint i = 1; i < topology.size(); i++) {
    (*neuronLayers[i]) = (*neuronLayers[i-1]) * (*weights[i - 1]);
    neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr(std::ptr_fun(activationFunction));
  }
}


