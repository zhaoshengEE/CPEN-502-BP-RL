package myrobot;

import NeuralNet.NeuralNet;
import robocode.*;

import java.awt.*;
import java.util.*;

public class Yang_Robot extends AdvancedRobot {

    // DYZ
    private int trainInterval = 50;
    private static int trainRoundNum = 0;
    private int testInterval = 100;
    private static int testRoundNum = 0;
    private static boolean underTraining = true;

    public enum OperationMode {scan, performAction}
    public enum Action {ahead, back, aheadLeft, aheadRight, backLeft, backRight, turnGunFire}

    private Double currentMyEnergy = 100.0;
    private Double currentEnemyEnergy = 100.0;
    private Double currentDistanceToEnemy = 500.0;
    private Double currentDistanceToCenter = 500.0;
    private Action currentAction = Action.ahead;

    private Double previousMyEnergy = 25.0;
    private Double previousEnemyEnergy = 25.0;
    private Double previousDistanceToEnemy = 50.0;
    private Double previousDistanceToCenter = 50.0;
    private Action previousAction = Action.back;

    private OperationMode operationMode = OperationMode.scan;

    private static int totalNumRounds = 0;
    private static int numRoundsTo50 = 0;
    private static int numWins = 0;
    private static double winningRate = 0.0;

    private double gamma = 0.75;
    private double alpha = 0.5;
    private double epsilon = 0.5;

    private double bestQ = 0.0;
    private double currentQ = 0.0;
    private double previousQ = 0.0;

    private final double instantPenalty = -0.25;
    private final double terminalPenalty = -0.5;
    private final double instantReward = 1.0;
    private final double terminalReward = 2.0;
    private double reward = 0.0;

    private double xCenter;
    private double yCenter;

    private double my_location_X = 0.0;
    private double my_location_Y = 0.0;
    private double my_energy = 0.0;

    private double enemy_Bearing = 0.0;
    private double enemy_Distance = 0.0;
    private double enemy_energy = 0.0;

    private boolean onPolicy = false;

    private boolean terminalRewardOnly = false;

    private static LogFile logFile = null;

    Queue<Double[]> replayMemory = new LinkedList<Double[]>();

    private static NeuralNet net = new NeuralNet(new int[]{5, 20, 1});
    private Double errorTarget = 0.02;
    private Double learningRate = 0.005;
    private Double momentum = 0.9;

    public void run() {
        setBulletColor(Color.BLACK);
        setGunColor(Color.GRAY);
        setBodyColor(Color.BLUE);
        setRadarColor(Color.CYAN);

        xCenter = getBattleFieldWidth() / 2;
        yCenter = getBattleFieldHeight() / 2;

        net.initializeWeight(-2.0, 2.0);
        net.binary = true;

        if (logFile == null) {
            logFile = new LogFile(getDataFile("log.dat"));
            logFile.stream.printf("gamma,   %2.2f\n", gamma);
            logFile.stream.printf("alpha,   %2.2f\n", alpha);
            logFile.stream.printf("epsilon, %2.2f\n", epsilon);
            logFile.stream.printf("badInstantReward, %2.2f\n", instantPenalty);
            logFile.stream.printf("badTerminalReward, %2.2f\n", terminalPenalty);
            logFile.stream.printf("goodInstantReward, %2.2f\n", instantReward);
            logFile.stream.printf("goodTerminalReward, %2.2f\n\n", terminalReward);
        }

        while (true) {
            //[DYZ] if (totalNumRounds > 9000) epsilon = 0.0;
            epsilon = underTraining ? 0.5 : 0;

            switch (operationMode) {
                case scan: {
                    reward = 0.0;
                    turnRadarRight(90);
                    break;
                }
                case performAction: {
                    if (Math.random() <= epsilon)
                        currentAction = selectRandomAction();
                    else currentAction = selectBestAction(
                            my_energy,
                            enemy_energy,
                            enemy_Distance,
                            getDistanceToCenter(my_location_X, my_location_Y, xCenter, yCenter)
                    );

                    switch (currentAction) {
                        case ahead: {
                            setAhead(100);
                            execute();
                            break;
                        }
                        case back: {
                            setBack(100);
                            execute();
                            break;
                        }
                        case aheadLeft: {
                            setTurnLeft(20);
                            setAhead(100);
                            execute();
                            break;
                        }
                        case aheadRight: {
                            setTurnRight(20);
                            setAhead(100);
                            execute();
                            break;
                        }
                        case turnGunFire: {
                            turnGunRight(getHeading() - getGunHeading() + enemy_Bearing);
                            fire(3);
                            break;
                        }
                        case backLeft: {
                            setTurnLeft(20);
                            setBack(100);
                            execute();
                            break;
                        }
                        case backRight: {
                            setTurnRight(20);
                            setBack(100);
                            execute();
                            break;
                        }
                    }
                    Double[] x = new Double[]{
                            previousMyEnergy,
                            previousEnemyEnergy,
                            previousDistanceToEnemy,
                            previousDistanceToCenter,
                            Double.valueOf(previousAction.ordinal())
                    };
                    Double[][] input = {x};
                    Double[][] target = {{computeQ(reward, onPolicy)}};
                    //[DYZ]System.out.println("training started");
//                    Double[] memory = new Double[6];

                    if (underTraining) { // DYZ
                        net.train(input, target, errorTarget, learningRate, momentum);
                        //[DYZ]System.out.println("intermediate training done");
                    }
                    operationMode = OperationMode.scan;
                }
            }
        }
    }

    @Override
    public void onScannedRobot(ScannedRobotEvent e) {
        my_location_X = Double.valueOf(getX());
        my_location_Y = Double.valueOf(getY());
        my_energy = Double.valueOf(getEnergy());
        enemy_Bearing = Double.valueOf(e.getBearing());
        enemy_Distance = e.getDistance();
        enemy_energy = e.getEnergy();

        previousMyEnergy = currentMyEnergy;
        previousEnemyEnergy = currentEnemyEnergy;
        previousDistanceToEnemy = currentDistanceToEnemy;
        previousDistanceToCenter = currentDistanceToCenter;
        previousAction = currentAction;

        currentMyEnergy = my_energy;
        currentEnemyEnergy = enemy_energy;
        currentDistanceToEnemy = enemy_Distance;
        currentDistanceToCenter = getDistanceToCenter(my_location_X, my_location_Y, xCenter, yCenter);

        operationMode = OperationMode.performAction;
    }

    public Action selectRandomAction() {
        Random random = new Random();
        return Action.values()[random.nextInt(Action.values().length)];
    }

    public Action selectBestAction(double myEnergy, double enemyEnergy, double enemyDistance, double centerDistance) {
        double maxReward = -Double.MAX_VALUE;
        Action bestAction = null;
        double e1 = myEnergy;
        double d1 = enemyEnergy;
        double e2 = enemyDistance;
        double d2 = centerDistance;

        for (int i = 0; i < Action.values().length; i++) {
            Double[] x = new Double[]{e1, d1, e2, d2, Double.valueOf(i)};
            if (net.feedForward(x)[0] > maxReward){
                maxReward = net.feedForward(x)[0];
                bestAction = Action.values()[i];
            }

        }
        return bestAction;
    }

    public double getDistanceToCenter(double my_location_X, double my_location_Y, double xCenter, double yCenter) {
        return Math.sqrt(Math.pow((my_location_X - xCenter), 2) + Math.pow((my_location_Y - yCenter), 2));
    }

    public Double computeQ(double reward, boolean onPolicy) {
        Action bestAction = selectBestAction(my_energy, enemy_energy, enemy_Distance, getDistanceToCenter(my_location_X, my_location_Y, xCenter, yCenter));
        Double[] previousStateAction = new Double[]{
                previousMyEnergy,
                previousEnemyEnergy,
                previousDistanceToEnemy,
                previousDistanceToCenter,
                Double.valueOf(previousAction.ordinal())
        };
        Double[] currentStateAction = new Double[]{
                currentMyEnergy,
                currentEnemyEnergy,
                currentDistanceToEnemy,
                currentDistanceToCenter,
                Double.valueOf(currentAction.ordinal())
        };
        Double[] bestStateAction = new Double[]{
                currentMyEnergy,
                currentEnemyEnergy,
                currentDistanceToEnemy,
                currentDistanceToCenter,
                Double.valueOf(bestAction.ordinal())
        };

        previousQ = net.feedForward(previousStateAction)[0];
        currentQ = net.feedForward(currentStateAction)[0];
        bestQ = net.feedForward(bestStateAction)[0];

        return (onPolicy) ? previousQ + alpha * (reward + gamma * currentQ - previousQ)
                : previousQ + alpha * (reward + gamma * bestQ - previousQ);
    }

    @Override
    public void onBulletHit(BulletHitEvent e) {
        if (terminalRewardOnly == false)
            reward = instantReward;
    }

    @Override
    public void onHitByBullet(HitByBulletEvent e) {
        if (terminalRewardOnly == false)
            reward = instantPenalty;
    }

    @Override
    public void onBulletMissed(BulletMissedEvent e) {
        if (terminalRewardOnly == false)
            reward = instantPenalty;
    }

    @Override
    public void onHitRobot(HitRobotEvent e) {
        if (terminalRewardOnly == false)
            reward = instantPenalty;
    }

    @Override
    public void onHitWall(HitWallEvent e) {
        if (terminalRewardOnly == false)
            reward = instantPenalty;
    }

    @Override
    public void onWin(WinEvent e) {
        reward = terminalReward;
        numWins ++;
        if (underTraining) { // DYZ
            Double[] x = new Double[]{
                    previousMyEnergy,
                    previousEnemyEnergy,
                    previousDistanceToEnemy,
                    previousDistanceToCenter,
                    Double.valueOf(previousAction.ordinal())
            };
            Double[][] input = {x};
            Double[][] target = {{computeQ(reward, onPolicy)}};
            System.out.println("training called by onWin started");
            net.train(input, target, errorTarget, learningRate, momentum);
            System.out.println("training called by onWin done");
        }

        // DYZ
        /*
        if (numRoundsTo50 < 50) {
            numRoundsTo50 += 1;
            totalNumRounds += 1;
            numWins += 1;
        } else {
            winningRate = 100.0 * numWins / numRoundsTo50;
            logFile.stream.printf("Winning rate: %2.1f\n ", winningRate);
            logFile.stream.flush();
            numRoundsTo50 = 0;
            numWins = 0;
        } */
    }

    @Override
    public void onDeath(DeathEvent e) {
        reward = terminalPenalty;
        if (underTraining) { // DYZ
            Double[] x = new Double[]{
                    previousMyEnergy,
                    previousEnemyEnergy,
                    previousDistanceToEnemy,
                    previousDistanceToCenter,
                    Double.valueOf(previousAction.ordinal())
            };
            Double[][] input = {x};
            Double[][] target = {{computeQ(reward, onPolicy)}};
            System.out.println("training called by onDeath started");
            net.train(input, target, errorTarget, learningRate, momentum);

            System.out.println("training called by onDeath done");
        }

        // DYZ
        /*
        if (numRoundsTo50 < 50) {
            numRoundsTo50 += 1;
            totalNumRounds += 1;
        } else {
            winningRate = 100.0 * numWins / numRoundsTo50;
            logFile.stream.printf("Winning rate: %2.1f\n ", winningRate);
            logFile.stream.flush();
            numRoundsTo50 = 0;
            numWins = 0;
        } */
    }

    // DYZ
    @Override
    public void onRoundEnded(RoundEndedEvent event) {
        if (underTraining) {
            trainRoundNum ++;
            System.out.println("under training mode, epsilon: " + epsilon + ", trainRoundNum: " + trainRoundNum);
            if (trainRoundNum == trainInterval) {
                winningRate = 100.0 * numWins / trainRoundNum;
                logFile.stream.printf("Winning rate under train mode: %2.1f\n ", winningRate);
                trainRoundNum = 0;
                underTraining = false; // switch to test mode
                numWins = 0; // reset numWins in case test mode also needs to calculate win rate.
            }
        } else {  // test mode
            testRoundNum ++;
            System.out.println("under test mode, epsilon: " + epsilon + ", testRoundNum: " + testRoundNum + ", numWins: " + numWins);
            if (testRoundNum == testInterval) {
                winningRate = 100.0 * numWins / testRoundNum;
                testRoundNum = 0;
                underTraining = true; // switch to train mode
                //logFile.stream.printf("Winning rate under test mode: %2.1f\n ", winningRate);
                logFile.stream.flush();
                numWins = 0; // reset numWins in case training mode also needs to calculate win rate.
            }
        }
    }
}