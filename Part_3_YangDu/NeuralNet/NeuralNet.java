package NeuralNet;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuralNet{
    public boolean binary = false;
    private List<Layer> neuralNet = new ArrayList<>();

    public NeuralNet(int[] networkSize){
        for (int i = 0; i < networkSize.length - 1; i++) {
            neuralNet.add(new Layer(networkSize[i] + 1));
        }
        neuralNet.add(new Layer(networkSize[networkSize.length - 1]));
        for (int i = 0; i < neuralNet.size() - 1; i++){
            for (int neuronThisLayer = 0; neuronThisLayer < neuralNet.get(i).getNumberOfNeurons(); neuronThisLayer++){
                for (int neuronNextLayer = 0; neuronNextLayer < neuralNet.get(i + 1).getNumberOfNeurons(); neuronNextLayer++){
                    neuralNet.get(i).getNeuron(neuronThisLayer).addWeight(null);
                    neuralNet.get(i).getNeuron(neuronThisLayer).addPreWeight(null);
                }
            }
        }
    }

    public void initializeWeight(Double randMax, Double ranMin){
        for (int i = 0; i < neuralNet.size() - 1; i++){
            for (int neuronThisLayer = 0; neuronThisLayer < neuralNet.get(i).getNumberOfNeurons(); neuronThisLayer++){
                for (int neuronNextLayer = 0; neuronNextLayer < neuralNet.get(i + 1).getNumberOfNeurons(); neuronNextLayer++){
                    Double rand = randomWeight(ranMin, randMax);
                    neuralNet.get(i).getNeuron(neuronThisLayer).setWeight(neuronNextLayer, rand);
                    neuralNet.get(i).getNeuron(neuronThisLayer).setPreWeight(neuronNextLayer, rand);
                }
                if (i != neuralNet.size() - 2){
                    neuralNet.get(i).getNeuron(neuronThisLayer).setWeight(neuralNet.get(i + 1).getNumberOfNeurons() - 1, 0.0);
                    neuralNet.get(i).getNeuron(neuronThisLayer).setPreWeight(neuralNet.get(i + 1).getNumberOfNeurons() - 1, 0.0);
                }
            }
        }
    }

    public Double[] feedForward(Double[] input){
        for (int i = 0; i < input.length; i++){
            neuralNet.get(0).getNeuron(i).setValue(input[i]);
        }
        neuralNet.get(0).getNeuron(input.length).setValue(1.0);
        for (int layer = 1; layer < neuralNet.size(); layer++){
            for (int neuronThisLayer = 0; neuronThisLayer < neuralNet.get(layer).getNumberOfNeurons(); neuronThisLayer++){
                Double value = 0.0;
                for (int neuronPreviousLayer = 0; neuronPreviousLayer < neuralNet.get(layer - 1).getNumberOfNeurons(); neuronPreviousLayer++){
                    value += neuralNet.get(layer - 1).getNeuron(neuronPreviousLayer).getWeight(neuronThisLayer) * neuralNet.get(layer - 1).getNeuron(neuronPreviousLayer).getValue();
                }
                value = customSigmoid(value);
                neuralNet.get(layer).getNeuron(neuronThisLayer).setValue(value);
            }
            if (layer != neuralNet.size() - 1) {
                neuralNet.get(layer).getNeuron(neuralNet.get(layer).getNumberOfNeurons() - 1).setValue(1.0);
            }
        }
        Double[] output = new Double[neuralNet.get(neuralNet.size() - 1).getNumberOfNeurons()];
        for (int i = 0; i < output.length; i++){
            output[i] = neuralNet.get(neuralNet.size() - 1).getNeuron(i).getValue();
        }
        return output;
    }

    public void backpropagation (Double[] target, Double learningRate, Double momentum) {
        for (int layer = neuralNet.size() - 1; layer > 0; layer--) {
            if (layer == neuralNet.size() - 1) {
                for (int neuron = 0; neuron < neuralNet.get(layer).getNumberOfNeurons(); neuron++) {
                    Double error = neuralNet.get(layer).getNeuron(neuron).getValue() - target[neuron];
                    Double delta = error * customSigmoidDerivative(neuralNet.get(layer).getNeuron(neuron).getValue());
                    neuralNet.get(layer).getNeuron(neuron).setError(delta);
                }
                //this way of updating on the go is better
                for (int neuron = 0; neuron < neuralNet.get(layer - 1).getNumberOfNeurons(); neuron++){
                    for (int neuronNextLayer = 0; neuronNextLayer < neuralNet.get(layer).getNumberOfNeurons(); neuronNextLayer++){
                        Double weightChange = (neuralNet.get(layer - 1).getNeuron(neuron).getWeight(neuronNextLayer) - neuralNet.get(layer - 1).getNeuron(neuron).getPreWeight(neuronNextLayer));
                        Double delta = momentum * weightChange - learningRate * neuralNet.get(layer - 1).getNeuron(neuron).getValue() * neuralNet.get(layer).getNeuron(neuronNextLayer).getError();
                        neuralNet.get(layer - 1).getNeuron(neuron).setPreWeight(neuronNextLayer, neuralNet.get(layer - 1).getNeuron(neuron).getWeight(neuronNextLayer));
                        neuralNet.get(layer - 1).getNeuron(neuron).setWeight(neuronNextLayer, neuralNet.get(layer - 1).getNeuron(neuron).getWeight(neuronNextLayer) + delta);
                    }
                }
            }
            else {
                for (int neuron = 0; neuron < neuralNet.get(layer).getNumberOfNeurons(); neuron++) {
                    Double sumError = 0.0;
                    for (int neuronNextLayer = 0; neuronNextLayer < neuralNet.get(layer + 1).getNumberOfNeurons(); neuronNextLayer++) {
                        sumError += neuralNet.get(layer + 1).getNeuron(neuronNextLayer).getError() * neuralNet.get(layer).getNeuron(neuron).getWeight(neuronNextLayer);
                    }
                    Double delta = sumError * customSigmoidDerivative(neuralNet.get(layer).getNeuron(neuron).getValue());
                    neuralNet.get(layer).getNeuron(neuron).setError(delta);
                }
                for (int neuron = 0; neuron < neuralNet.get(layer - 1).getNumberOfNeurons(); neuron++){
                    for (int neuronNextLayer = 0; neuronNextLayer < neuralNet.get(layer).getNumberOfNeurons() - 1; neuronNextLayer++){
                        Double weightChange = (neuralNet.get(layer - 1).getNeuron(neuron).getWeight(neuronNextLayer) - neuralNet.get(layer - 1).getNeuron(neuron).getPreWeight(neuronNextLayer));
                        Double delta = momentum * weightChange - learningRate * neuralNet.get(layer - 1).getNeuron(neuron).getValue() * neuralNet.get(layer).getNeuron(neuronNextLayer).getError();
                        neuralNet.get(layer - 1).getNeuron(neuron).setPreWeight(neuronNextLayer, neuralNet.get(layer - 1).getNeuron(neuron).getWeight(neuronNextLayer));
                        neuralNet.get(layer - 1).getNeuron(neuron).setWeight(neuronNextLayer, neuralNet.get(layer - 1).getNeuron(neuron).getWeight(neuronNextLayer) + delta);
                    }
                }
            }
        }
    }

    public Double randomWeight(Double min, Double max){
        Random random = new Random();
        return min + random.nextDouble() * (max - min);
    }

    public Double sigmoid(Double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public Double customSigmoid(Double x) {
        if (binary){
            return  sigmoid(x);
        }
        else{
            return (2 / (1 + Math.exp(-x)) - 1);
        }
    }

    public Double sigmoidDerivative(Double x){
        return x * (1 - x);
    }

    public Double customSigmoidDerivative(Double x){
        if (binary){
            return sigmoidDerivative(x);
        }
        else{
            return (1 - x) * (1 + x) * 0.5;
        }
    }

    public Double Error(Double[] output, Double[] target) {
        Double totalError = 0.0;
        for (int i = 0; i < output.length; i++){
            totalError += Math.pow((target[i] - output[i]), 2);
        }
        return totalError / 2;
    }

    public void train(Double[][] input, Double[][] target, Double errorTarget, Double learningRate, Double momentum){
        int epoch = 0;
        Double error;
        do{
            error = 0.0;
            for (int i = 0; i < input.length; i++){
                Double[] output = feedForward(input[i]);
                error += Error(output, target[i]);
                backpropagation(target[i], learningRate, momentum);
            }
            System.out.println("epoch " + epoch + " error is " + error);
            epoch++;
        }while (error > errorTarget);
    }

    public void writeToFile(Double[][] data, String type) throws IOException {
        int row = data.length;
        int column = data[0].length;
        try{
            FileWriter fileWriter = new FileWriter(type);
            for (int i = 0; i < row; i++){
                for (int j = 0; j < column; j++){
                    fileWriter.write(data[i][j] + " ");
                }
                fileWriter.write("\n");
            }
            fileWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}


