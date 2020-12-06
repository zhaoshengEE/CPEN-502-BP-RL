package NeuralNet;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class Test {
    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader("/Users/ydu/Downloads/CPEN502/LUT.dat"));
        String line;
        int lineNum = 0;
        while((line = reader.readLine()) != null){
            lineNum++;
        }
        reader.close();
        Double[][] lutTrainingInput = new Double[lineNum][5];
        Double[][] lutTrainingTarget = new Double[lineNum][1];
        BufferedReader br = new BufferedReader(new FileReader("/Users/ydu/Downloads/CPEN502/LUT.dat"));
        lineNum = 0;
        while((line = br.readLine()) != null){
            String[] data = line.split(",");
            for (int i = 0; i < data.length - 2; i++){
                lutTrainingInput[lineNum][i] = Double.valueOf(data[i]);
            }
            lutTrainingTarget[lineNum][0] = Double.valueOf(data[data.length - 2]) / 3.0;
            lineNum++;
        }
        br.close();
        Double[] aveInput = new Double[5];
        Double[] stdInput = new Double[5];
        for (int i = 0; i < 5; i++){
            Double temp = 0.0;
            for (int j = 0; j < lineNum; j++){
                temp += lutTrainingInput[j][i];
            }
            aveInput[i] = temp / (lineNum + 1);
        }
        for (int i = 0; i < 5; i++){
            Double temp = 0.0;
            for (int j = 0; j < lineNum; j++){
                temp += Math.pow(lutTrainingInput[j][i] - aveInput[i], 2);
            }
            stdInput[i] = Math.sqrt(temp / (lineNum));
        }
        for (int i = 0; i < 5; i++){
            for (int j = 0; j < lineNum; j++){
                lutTrainingInput[j][i] = (lutTrainingInput[j][i] - aveInput[i]) / stdInput[i];
            }
        }
//        Double sumTarget = 0.0;
//        Double aveTarget = 0.0;
//        Double stdTarget = 0.0;
//        for (int i = 0; i < lineNum; i++){
//            sumTarget += lutTrainingTarget[i][0];
//        }
//        aveTarget = sumTarget / (lineNum + 1);
//        Double temp = 0.0;
//        for (int i = 0; i < lineNum; i++){
//            temp += Math.pow(lutTrainingTarget[i][0] - aveTarget, 2);
//        }
//        stdTarget = Math.sqrt(temp / (lineNum));
//
//        for (int i = 0; i < lineNum; i++){
//            lutTrainingTarget[i][0] = (lutTrainingTarget[i][0] - aveTarget) / stdTarget;
//        }

        //construct a neural network and the layers can be more than 3 i.e. it can be [2,4,4,4,1]
        NeuralNet network = new NeuralNet(new int[]{5, 20, 1});
        //switch this for bipolar or binary input, true -> binary input
        network.binary = true;
        //initialize the weight with Random numbers -> [-0.5, 0.5]
        network.initializeWeight(2.0, -2.0);
        //total error target -> 0.05, learning rate -> 0.2 and momentum -> 0.9
        network.train(lutTrainingInput, lutTrainingTarget, 0.5, 0.005, 0.9);
    }
}
