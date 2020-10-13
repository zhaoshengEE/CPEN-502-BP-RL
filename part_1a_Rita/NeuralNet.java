package Sarb;

import java.util.Arrays;

public class NeuralNet implements NeuralNetInterface {
  int numInputs = 0;
  int numHidden = 0;
  double learningRate = 0.0;
  double momentumTerm = 0.0;
  double sigmoidLowerBound = 0;
  double sigmoidUpperBound = 1;
  boolean useBipolarHiddenNeurons = false;

  double[][] hiddenWeights;
  double[][] prevHiddenWeights;
  double[] hiddenBiases;
  double[] hiddenOutputs;
  double[] outputWeights;
  double[] prevOutputWeights;
  double outputBias = bias;

  public NeuralNet(int argNumInputs, int argNumHidden, double argLearningRate, double argMomentumTerm, double argA,
      double argB, boolean argUseBipolarHiddenNeurons) {
    numInputs = argNumInputs;
    numHidden = argNumHidden;
    learningRate = argLearningRate;
    momentumTerm = argMomentumTerm;
    sigmoidLowerBound = argA;
    sigmoidUpperBound = argB;
    useBipolarHiddenNeurons = argUseBipolarHiddenNeurons;

    hiddenWeights = new double[numInputs][numHidden];
    prevHiddenWeights = new double[numInputs][numHidden];
    hiddenBiases = new double[numHidden];
    hiddenOutputs = new double[numHidden];
    outputWeights = new double[numHidden];
    prevOutputWeights = new double[numHidden];
  }

  public double train(double[] x, double argValue) {
    double net_output = outputFor(x);
    updateWeights(x, net_output, argValue);
    return net_output - argValue;
  }

  public double outputFor(double[] x) {
    if (x.length != numInputs) {
      System.err.println("Wrong Input Number!");
      return 0.0;
    }
    // calc hiddenOutputs
    for (int j = 0; j < numHidden; ++j) {
      double temp = 0;
      for (int i = 0; i < numInputs; ++i) {
        temp += x[i] * hiddenWeights[i][j];
      }
      if (useBipolarHiddenNeurons) {
        hiddenOutputs[j] = sigmoid(temp + hiddenBiases[j]);
      } else {
        hiddenOutputs[j] = customSigmoid(temp + hiddenBiases[j]);
      }
    }
    // calc output
    double output = 0.0;
    for (int j = 0; j < numHidden; ++j) {
      output += outputWeights[j] * hiddenOutputs[j];
    }
    if (useBipolarHiddenNeurons) {
      output = sigmoid(output + outputBias);
    } else {
      output = customSigmoid(output + outputBias);
    }
    return output;
  }

  public double sigmoid(double x) {
    return 2 / (1 + Math.pow(Math.E, -x)) - 1;
  }

  public double customSigmoid(double x) {
    return (sigmoidUpperBound - sigmoidLowerBound) / (1 + Math.pow(Math.E, -x)) - sigmoidLowerBound;
  }

  public double derivationSigmoid(double x) {
    if (useBipolarHiddenNeurons) {
      return (1 - x * x) / 2;
    } else {
      return x * (1 - x);
    }
  }

  public void initializeWeights() {
    for (int i = 0; i < numInputs; ++i) {
      for (int j = 0; j < numHidden; ++j) {
        hiddenWeights[i][j] = Math.random() - 0.5;
      }
    }
    for (int i = 0; i < numHidden; ++i) {
      outputWeights[i] = Math.random() - 0.5;
    }
    Arrays.fill(hiddenBiases, bias);
    outputBias = bias;
  }

  public void zeroWeights() {
    for (int i = 0; i < numInputs; ++i) {
      for (int j = 0; j < numHidden; ++j) {
        hiddenWeights[i][j] = 0.0;
      }
    }
    for (int i = 0; i < numHidden; ++i) {
      outputWeights[i] = 0.0;
    }
    Arrays.fill(hiddenBiases, bias);
    outputBias = bias;
  }

  private void updateWeights(double[] x, double netOutput, double argValue) {
    double[] hiddenErrorSignal = new double[numHidden];
    double outputErrorSignal = 0;
    // Calc outputErrorSignal
    outputErrorSignal = argValue - netOutput;
    // Calc hiddenErrorSignal
    for (int i = 0; i < numHidden; ++i) {
      hiddenErrorSignal[i] = outputErrorSignal * outputWeights[i];
    }
    // update hiddenWeights
    for (int i = 0; i < numInputs; ++i) {
      for (int j = 0; j < numHidden; ++j) {
        hiddenWeights[i][j] += momentumTerm * (hiddenWeights[i][j] - prevHiddenWeights[i][j])
            + learningRate * hiddenErrorSignal[j] * derivationSigmoid(hiddenOutputs[j]) * x[i];
        prevHiddenWeights[i][j] = hiddenWeights[i][j];
      }
    }
    // update hiddenBiases
    for (int i = 0; i < numHidden; ++i) {
      hiddenBiases[i] += learningRate * hiddenErrorSignal[i] * derivationSigmoid(hiddenOutputs[i]) * 1;
    }
    // update outputWeights
    for (int i = 0; i < numHidden; ++i) {
      outputWeights[i] += momentumTerm * (outputWeights[i] - prevOutputWeights[i])
          + learningRate * outputErrorSignal * derivationSigmoid(netOutput) * hiddenOutputs[i];
      prevOutputWeights[i] = outputWeights[i];
    }
    // update outputBias
    outputBias += learningRate * outputErrorSignal * derivationSigmoid(netOutput) * 1;
  }

  public String printHiddenWeights() {
    String ret = "[ ";
    for (double[] row : hiddenWeights) {
      ret += Arrays.toString(row) + " ";
    }
    ret += "]\n";
    return ret;
  }
}
