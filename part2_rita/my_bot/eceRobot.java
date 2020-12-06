package my_bot;

import my_nn.LUTLearning;
import robocode.*;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Date;

public class eceRobot extends AdvancedRobot {
    // Action
    enum enumOperationalMode {scan, execute}

    static enumOperationalMode operationalMode = enumOperationalMode.scan;
    static int xMid;
    static int yMid;

    static int totalNumRounds;
    static int roundCount;
    static final int countRounds = 50;
    static int winCount;
    static double reward;
    static boolean useIntermediateReward = true; // false for only terminal reward
    static LUTLearning rl = new LUTLearning();

    double egoX;
    double egoY;
    double egoEnergy;
    double egoHeading;
    double egoSpeed;
    double enemySpeed;
    double enemyBearing;
    double enemyDistance;
    double enemyEnergy;
    double enemyHeading;
    double positiveTerminalReward = 50;
    double negativeTerminalReward = -30;
    double positiveInstantReward = 5;
    double negativeInstantReward = -3;
    final double turnRadians = Math.PI / 4.0;
    final double moveDistance = 100;

    static String logFilename = eceRobot.class.getSimpleName() + "-" + System.currentTimeMillis()/1000L + ".log";
    static LogFile log = null;

    public void run() {
        xMid = (int) getBattleFieldWidth() / 2;
        yMid = (int) getBattleFieldHeight() / 2;
        //        loadTable();
        if (log == null) {
            log = new LogFile(getDataFile(logFilename));
            log.stream.printf("learningRate, %2.2f\n", rl.learningRate);
            log.stream.printf("discountRate, %2.2f\n", rl.discountRate);
            log.stream.printf("explorationRate, %2.2f\n", rl.explorationRate);
            log.stream.printf("positiveTerminalReward, %2.2f\n", positiveTerminalReward);
            log.stream.printf("negativeTerminalReward, %2.2f\n", negativeTerminalReward);
            log.stream.printf("positiveInstantReward, %2.2f\n", positiveInstantReward);
            log.stream.printf("negativeInstantReward, %2.2f\n\n", negativeInstantReward);
        }
        setAdjustGunForRobotTurn(true);
        setAdjustRadarForGunTurn(true);
        setAdjustRadarForRobotTurn(true);

        while (true) {
            if (operationalMode == enumOperationalMode.scan) {
                reward = 0.0;
                turnRadarLeftRadians(Math.PI / 2.0);
            } else if (operationalMode == enumOperationalMode.execute) {
                double[] state = getState();
                int currentState = rl.indexFor(state);
                int currentAction = rl.selectAction(currentState);
                switch (currentAction) {
                    case 0: {
                        setAhead(moveDistance);
                        execute();
                        break;
                    }
                    case 1: {
                        setBack(moveDistance);
                        execute();
                        break;
                    }
                    case 2: {
                        setTurnLeftRadians(turnRadians);
                        setAhead(moveDistance);
                        execute();
                        break;
                    }
                    case 3: {
                        setTurnRightRadians(turnRadians);
                        setAhead(moveDistance);
                        execute();
                        break;
                    }
                    case 4: {
                        setTurnLeftRadians(turnRadians);
                        setBack(moveDistance);
                        execute();
                        break;
                    }
                    case 5: {
                        setTurnRightRadians(turnRadians);
                        setBack(moveDistance);
                        execute();
                        break;
                    }
                    case 6: {
                        turnGunLeftRadians(getGunHeadingRadians() - getHeadingRadians() - enemyBearing);
                        fire(3);
                        break;
                    }
                }
                rl.learning(currentState, currentAction, reward);
                operationalMode = enumOperationalMode.scan;
            }
        }
    }

    private double[] getState() {
        return new double[]{getDistanceToCenter(egoX, egoY), egoEnergy, enemyDistance, enemyBearing, enemyHeading, enemyEnergy};
    }

    private double getDistanceToCenter(double x, double y) {
        return Math.sqrt(Math.pow(x - xMid, 2.0) + Math.pow(y - yMid, 2.0));
    }

    public void onScannedRobot(ScannedRobotEvent e) {
        enemyBearing = e.getBearingRadians();
        enemyHeading = e.getHeadingRadians();
        enemySpeed = e.getVelocity();
        enemyDistance = e.getDistance();
        enemyEnergy = e.getEnergy();
        egoX = getX();
        egoY = getY();
        egoEnergy = getEnergy();
        egoHeading = getHeadingRadians();
        egoSpeed = getVelocity();//always 0 when scan
        operationalMode = enumOperationalMode.execute;
    }

    double normalizeBearing(double argValue) {
        if (argValue > Math.PI)
            argValue -= 2 * Math.PI;
        if (argValue < -Math.PI)
            argValue += 2 * Math.PI;
        return argValue;
    }

    double normalizeHeading(double argValue) {
        if (argValue > 2 * Math.PI)
            argValue -= 2 * Math.PI;
        if (argValue < 0)
            argValue += 2 * Math.PI;
        return argValue;
    }

    // robot bullet hit
    public void onBulletHit(BulletHitEvent e) {
        if (useIntermediateReward) {
            reward += positiveInstantReward;
        }
    }

    // robot bullet missed
    public void onBulletMissed(BulletMissedEvent e) {
        if (useIntermediateReward) {
            reward += negativeInstantReward;
        }
    }

    // robot hit by bullet
    public void onHitByBullet(HitByBulletEvent e) {
        if (useIntermediateReward) {
            reward += negativeInstantReward;
        }
    }

    // robot hit wall
    public void onHitWall(HitWallEvent e) {
        if (useIntermediateReward) {
            reward += negativeInstantReward;
        }
    }

    // enemy robot die
    public void onRobotDeath(RobotDeathEvent e) {
        reward += positiveTerminalReward;
        double[] state = getState();
        int currentState = rl.indexFor(state);
        int currentAction = rl.selectAction(currentState);
        rl.learning(currentState, currentAction, reward);
        operationalMode = enumOperationalMode.scan;
        enemyDistance = 1000;
    }

    // Terminal rewards
    public void onWin(WinEvent event) {
        winCount++;
    }

    public void onDeath(DeathEvent event) {
        reward += negativeTerminalReward;
        double[] state = getState();
        int currentState = rl.indexFor(state);
        int currentAction = rl.selectAction(currentState);
        rl.learning(currentState, currentAction, reward);
    }

    public void onRoundEnded(RoundEndedEvent e) {
        operationalMode = enumOperationalMode.scan;
        logResult();
    }

    private void logResult() {
        roundCount++;
        totalNumRounds++;
        if (roundCount == countRounds) {
            double winningRate = (double) (winCount) / roundCount;
            log.stream.printf("Round, exploration Rate, winning Rate: %d, %f, %f\n", totalNumRounds, rl.explorationRate, winningRate);
            winCount = 0;
            roundCount = 0;
            if (totalNumRounds >= 5000) {
                rl.explorationRate = 0;
            }
            if (totalNumRounds >= 6000) {
               rl.learningRate = 0;
            }
        }
        saveTable();
    }

    public void loadTable() {
        try {
            rl.load(getDataFile(logFilename.replaceAll(".log","-lut.csv")));
        } catch (Exception e) {
            out.println("Load Error!" + e);
        }
    }

    public void saveTable() {
        try {
            rl.save(getDataFile(logFilename.replaceAll(".log","-lut.csv")));
        } catch (Exception e) {
            out.println("Save Error!" + e);
        }
    }
}