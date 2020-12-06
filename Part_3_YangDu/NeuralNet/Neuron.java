package NeuralNet;

import java.util.ArrayList;
import java.util.List;

public class Neuron {

    private Double value = null;

    private List<Double> weightArray = new ArrayList<>();

    private List<Double> preWeightArray = new ArrayList<>();

    private Double error = null;

    public Double getValue() {
        return this.value;
    }

    public void setValue(Double value) {
        this.value = value;
    }

    public void addWeight(Double weight){
        weightArray.add(weight);
    }

    public void setWeight(int index, Double weight){
        weightArray.set(index, weight);
    }

    public Double getWeight(int index){
        if (index >= 0 && index < weightArray.size()){
            return weightArray.get(index);
        }
        else{
            return null;
        }
    }

    public void addPreWeight(Double weight) {
        preWeightArray.add(weight);
    }

    public void setPreWeight(int index, Double weight) {
        preWeightArray.set(index, weight);
    }

    public Double getPreWeight(int index){
        if (index >= 0 && index < preWeightArray.size()){
            return preWeightArray.get(index);
        }
        else{
            return null;
        }
    }

    public Double getError() {
        return this.error;
    }

    public void setError(Double error) {
        this.error = error;
    }
}
