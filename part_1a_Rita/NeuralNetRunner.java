package Sarb;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class NeuralNetRunner {
  // Sample training set. The XOR.
   double[][] xorTrainingSet = { { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };
   double[] xorTargetSet = { -1, 1, 1, -1 };
//  double[][] xorTrainingSet = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
//  double[] xorTargetSet = { 0, 1, 1, 0 };

  NeuralNet nn = new NeuralNet(
          2,
          4,
          0.2,
          0.9,
          -1,
          1,
          true);

  private double totalError() {
    double sumError = 0.0;
    double output = 0.0;
    for (int i = 0; i < xorTargetSet.length; ++i) {
      output = nn.outputFor(xorTrainingSet[i]);
      sumError += 0.5 * Math.pow((xorTargetSet[i] - output), 2);
    }
    return sumError;
  }

  private int train(boolean showErrorAtEachEpoch, boolean showHiddenWeightsAtEachEpoch) {
    int numEpochs = 500000;
    int epochsToReachTarget = 0;
    boolean targetReached = false;
    double target = 0.05;

    nn.initializeWeights();

    FileWriter fileWriter = null;
    File file = new File("data.csv");
    try {
      fileWriter = new FileWriter(file);
    } catch (IOException e) {
      e.printStackTrace();
    }
    try {
      fileWriter.write("epochs, error\n");
    } catch (IOException e) {
      e.printStackTrace();
    }
    for (int i = 0; i < numEpochs; ++i) {
      for (int j = 0; j < xorTargetSet.length; ++j) {
        nn.train(xorTrainingSet[j], xorTargetSet[j]);
      }
      if (showErrorAtEachEpoch) {
        System.out.println("Error at epoch " + i + " : " + totalError());
      }
      String outputString = i + ", " + totalError() + "\n";
      try {
        fileWriter.write(outputString);
      } catch (IOException e) {
        e.printStackTrace();
      }
      if (showHiddenWeightsAtEachEpoch) {
        System.out.print("Hidden weights at epoch " + i + " : " + nn.printHiddenWeights());
      }
      if (!targetReached) {
        if (totalError() < target) {
          epochsToReachTarget = i;
          targetReached = true;
          break;
        }
      }
    }
    try {
      if (fileWriter != null) {
        fileWriter.close();
      }
    } catch (IOException e) {
      System.err.println("Error when closing the Stream");
    }

    if (targetReached) {
      System.out.println("Target reached at " + epochsToReachTarget + " epochs ");
      return epochsToReachTarget;
    } else {
      System.out.println("Target not reached!");
      return -1;
    }
  }

  public static void main(String[] args) {
    // Scanner reader = new Scanner(System.in);
    // System.out.print("Enter the number of trails you want to run: ");
    // int numTrails = reader.nextInt();
    // System.out.print("Do you want to see errors at each epoch y/n? ");
    // String showErrors = reader.next();
    // System.out.print("Do you want to see the hidden weights at each epoch y/n?
    // ");
    // String showHiddenWeights = reader.next();
    // reader.close();
    int numTrails = 2000;
    String showErrors = "y";
    String showHiddenWeights = "n";

    int numConverged = 0;
    int sum = 0;
    int epochs = 0;
    for (int i = 0; i < numTrails; ++i) {
      NeuralNetRunner myTester = new NeuralNetRunner();
      epochs = myTester.train(showErrors.equals("y"), showHiddenWeights.equals("y"));
      if (epochs != -1) {
        ++numConverged;
        sum += epochs;
      }
    }
    if (numConverged > 0) {
      System.out.println("Average convergence rate = " + (int) sum / numConverged);
    }
    // File xorWeights = new File("xorWeights.txt");
    // myTester.nn.save(xorWeights);
  }
}
