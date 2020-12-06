package NeuralNet;

import java.util.ArrayList;
import java.util.List;

public class Layer {

    private List<Neuron> Layer = new ArrayList<>();

    public Layer(int numOfNeurons){
        for (int i = 0; i < numOfNeurons; i++){
            Layer.add(new Neuron());
        }
    }

    public Neuron getNeuron(int index){
        if (index <= Layer.size() && index >= 0){
            return Layer.get(index);
        }
        else {
            return null;
        }
    }

    public int getNumberOfNeurons(){
        return Layer.size();
    }
}
