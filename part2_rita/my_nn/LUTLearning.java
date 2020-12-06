package my_nn;

import robocode.RobocodeFileOutputStream;

import java.io.*;
import java.util.Random;

public class LUTLearning implements LUTInterface {
    // variables to be tuned
    public double learningRate = 0.1;
    public double discountRate = 0.9;
    public double explorationRate = 0.2;
    public boolean offPolicy = true;

    // States Arguments
    public int[][][][][][] States =
            new int[enumDistance.values().length]
                    [enumEnergy.values().length]
                    [enumDistance.values().length]
                    [enumDegree.values().length]
                    [enumDegree.values().length]
                    [enumEnergy.values().length];
    public final int statesNum = enumDistance.values().length *
            enumEnergy.values().length *
            enumDistance.values().length *
            enumDegree.values().length *
            enumDegree.values().length *
            enumEnergy.values().length;
    public final int actionsNum = enumActions.values().length;
    public int previousState;
    public int previousAction;
    // Look Up Table
    public double[][] LUTable;

    public LUTLearning() {
        LUTable = new double[statesNum][actionsNum];
        initialiseLUT(); // set all function values to be zero
    }

    public int getEnergy(double energy) {
        if (energy < 30.0) {
            return 0;
        } else if (energy < 60.0) {
            return 1;
        } else {
            return 2;
        }
    }

    public int getOpponentDistance(double distance) {
        // Four kinds of distance: close, near, far, very far
        if (distance < 100) {
            return 0;
        } else if (distance < 400) {
            return 1;
        } else {
            return 2;
        }
    }

    public int getCenterDistance(double distance) {
        if (distance < 100) {
            return 0;
        } else if (distance < 300) {
            return 1;
        } else {
            return 2;
        }
    }

    // Bearing: [-PI, PI)
    public int getBearing(double bearing) {
        int b = 0;
        if (bearing >= -Math.PI && bearing < -Math.PI / 2) {
            b = 0;
        } else if (bearing >= -Math.PI / 2 && bearing < 0) {
            b = 1;
        } else if (bearing >= 0 && bearing < Math.PI / 2) {
            b = 2;
        } else if (bearing >= Math.PI / 2 && bearing < Math.PI) {
            b = 3;
        }
        return b;
    }

    // Heading: [0, 2PI)
    public int getHeading(double heading) {
        return getBearing(heading - Math.PI);
    }

    @Override
    public void initialiseLUT() {
        // Initialize table to zero
        for (int i = 0; i < statesNum; ++i) {
            for (int j = 0; j < actionsNum; ++j) {
                LUTable[i][j] = 0.0;
            }
        }

        // Initialize States
        int count = 0;
        for (int i = 0; i < enumDistance.values().length; ++i) {
            for (int j = 0; j < enumEnergy.values().length; ++j) {
                for (int k = 0; k < enumDistance.values().length; ++k) {
                    for (int l = 0; l < enumDegree.values().length; ++l) {
                        for (int m = 0; m < enumDegree.values().length; ++m) {
                            for (int n = 0; n < enumEnergy.values().length; ++n) {
                                States[i][j][k][l][m][n] = count++;
                            }
                        }
                    }
                }
            }
        }
    }

    double max(double[] arr) {
        double max = arr[0];
        for (double num : arr) {
            if (num > max) {
                max = num;
            }
        }
        return max;
    }

    int argmax(double[] arr) {
        int max = 0;
        for (int i = 0; i < arr.length; ++i) {
            if (arr[i] > arr[max]) {
                max = i;
            }
        }
        return max;
    }

    @Override
    public int indexFor(double[] state) {
        int disToCenter = getCenterDistance(state[0]);
        int egoEnergy = getEnergy(state[1]);
        int enemyDistance = getOpponentDistance(state[2]);
        int enemyBearing = getBearing(state[3]);
        int enemyHeading = getHeading(state[4]);
        int enemyEnergy = getEnergy(state[5]);
        return States[disToCenter][egoEnergy][enemyDistance][enemyBearing][enemyHeading][enemyEnergy];
    }

    @Override
    public double outputFor(double[] state) {
        return 0;
    }

    @Override
    public double train(double[] state, double reward) {
        int currentState = indexFor(state);
        int currentAction = selectAction(currentState);
        return learning(currentState, currentAction, reward);
    }

    public double learning(int currentState, int currentAction, double reward) {
        double previousQ = LUTable[previousState][previousAction];
        double currentQ;
        if (offPolicy) { // QLearning
            currentQ = (1 - learningRate) * previousQ
                    + learningRate * (reward + discountRate * max(LUTable[currentState]));
        } else { // Sarsa
            currentQ = (1 - learningRate) * previousQ
                    + learningRate * (reward + discountRate * LUTable[currentState][currentAction]);
        }
        LUTable[previousState][previousAction] = currentQ;
        previousState = currentState;
        previousAction = currentAction;
        return 0;
    }

    public int selectAction(int state) {
        double random = Math.random();
        if (explorationRate > random) {
            Random ran = new Random();
            return ran.nextInt(actionsNum);
        } else { // Pure greedy
            return argmax(LUTable[state]);
        }
    }

    public void load(File file) {
        BufferedReader read = null;
        try {
            read = new BufferedReader(new FileReader(file));
            for (int i = 0; i < statesNum; ++i) {
                for (int j = 0; j < actionsNum; ++j) {
                    LUTable[i][j] = Double.parseDouble(read.readLine());
                }
            }
        } catch (IOException | NumberFormatException e) {
            initialiseLUT();
        } finally {
            try {
                if (read != null)
                    read.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public void save(File file) {
        PrintStream write = null;
        try {
            write = new PrintStream(new RobocodeFileOutputStream(file));
            for (int i = 0; i < statesNum; ++i) {
                for (int j = 0; j < actionsNum; ++j) {
                    write.println(LUTable[i][j]);
                }
            }

            if (write.checkError())
                System.out.println("Could not save the data!");
            write.close();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (write != null)
                    write.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public void load(String argFileName) throws IOException {

    }

    public enum enumEnergy {low, medium, high}

    public enum enumDistance {veryClose, near, far}

    public enum enumDegree {zero, one, two, three}

    public enum enumActions {advance, retreat, advanceLeft, advanceRight, retreatLeft, retreatRight, fire}

}