#include "cpp/include/NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(std::vector<uint> topology, Scalar learningRate) {
    this->topology = topology;
    this->learningRate = learningRate;

    for (uint i = 0; i < topology.size(); i++){
      // initialize neuron layer
      if (i == topology.size() - 1){
        neuronLayers.push_back(new RowVector(topology[i]));
      } else {
        neuronLayers.push_back(new RowVector(topology[i] + 1));
      }

      // initialize cache and delta vectors
      cacheLayers.push_back(new RowVector(neuronLayers.size()));
      deltas.push_back(new RowVector(neuronLayers.size()));

      
    }
};
