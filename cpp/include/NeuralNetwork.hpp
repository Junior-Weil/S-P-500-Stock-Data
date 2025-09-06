#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <vector>

typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;

class NeuralNetwork {
  public:

    // Constructor
    NeuralNetwork(std::vector<uint> topology, Scalar learningRate = Scalar(0.005));

    // Forward Propagation of data
    void propagateForward(RowVector& input);

    // Backward Propagation of errors made by neurons
    void proagateeBackward(RowVector& output);

    // Calculate errors made by neurons in each layer
    void calcErrors(RowVector& output);

    // Update eights of connections
    void updateWeights();

    // Train NN give an array of data points
    void train(std::vector<RowVector*> data);

    //storage objects for working of neural network
    /*
          use pointers when using std::vector<Class> as std::vector<Class> calls destructor of
          Class as soon as it is pushed back! when we use pointers it can't do that, besides
          it also makes our neural network class less heavy!! It would be nice if you can use
          smart pointers instead of usual ones like this
    */

    std::vector<RowVector*> neuronLayers; // stores different layer of out network

    std::vector<RowVector*> cacheLayers; // stores the unactivated (activation fn not yet applied) values of layers

    std::vector<RowVector*> deltas; // error contribution of each neurons

    std::vector<Matrix*> weights; // the connection of weights itself

    std::vector<uint> topology;
    
    Scalar learningRate;

};
