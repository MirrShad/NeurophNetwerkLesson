

import java.util.ArrayList;
import java.util.List;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.transfer.Linear;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.LayerFactory;
import org.neuroph.util.NeuralNetworkFactory;
import org.neuroph.util.NeuralNetworkType;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.util.random.NguyenWidrowRandomizer;


public class MlPerceptron extends NeuralNetwork<BackPropagation> {

    private static final long serialVersionUID = 2L;

    public MlPerceptron(List<Integer> neuronsInLayers) {
        NeuronProperties neuronProperties = new NeuronProperties();
        neuronProperties.setProperty("transferFunction", TransferFunctionType.SIGMOID);
        this.createNetwork(neuronsInLayers, neuronProperties);
    }

    public MlPerceptron(int... neuronsInLayers) {
        NeuronProperties neuronProperties = new NeuronProperties();

        neuronProperties.setProperty("transferFunction", TransferFunctionType.SIGMOID);
        List<Integer> neuronsInLayersVector = new ArrayList<>();
        for (int i = 0; i < neuronsInLayers.length; i++) {
            neuronsInLayersVector.add(new Integer(neuronsInLayers[i]));
        }
        this.createNetwork(neuronsInLayersVector, neuronProperties);
    }

    public MlPerceptron(TransferFunctionType transferFunctionType, int... neuronsInLayers) {
        NeuronProperties neuronProperties = new NeuronProperties();
        neuronProperties.setProperty("transferFunction", transferFunctionType);
        List<Integer> neuronsInLayersVector = new ArrayList<>();
        for (int i = 0; i < neuronsInLayers.length; i++) {
            neuronsInLayersVector.add(new Integer(neuronsInLayers[i]));
        }
        this.createNetwork(neuronsInLayersVector, neuronProperties);
    }

    public MlPerceptron(List<Integer> neuronsInLayers, TransferFunctionType transferFunctionType) {
        NeuronProperties neuronProperties = new NeuronProperties();
        neuronProperties.setProperty("transferFunction", transferFunctionType);
        this.createNetwork(neuronsInLayers, neuronProperties);
    }

    public MlPerceptron(List<Integer> neuronsInLayers, NeuronProperties neuronProperties) {
        this.createNetwork(neuronsInLayers, neuronProperties);
    }

    protected void createNetwork(List<Integer> neuronsInLayers, NeuronProperties neuronProperties) {


        this.setNetworkType(NeuralNetworkType.MULTI_LAYER_PERCEPTRON);


        NeuronProperties inputNeuronProperties = new NeuronProperties(InputNeuron.class, Linear.class);
        Layer layer = LayerFactory.createLayer(neuronsInLayers.get(0), inputNeuronProperties);
        layer.addNeuron(new BiasNeuron());
        this.addLayer(layer);
        Layer prevLayer = layer;
        for (int layerIdx = 1; layerIdx < neuronsInLayers.size(); layerIdx++) {
            Integer neuronsNum = neuronsInLayers.get(layerIdx);
            layer = LayerFactory.createLayer(neuronsNum, neuronProperties);
            this.addLayer(layer);
            if(layerIdx != neuronsInLayers.size()-1)
            	layer.addNeuron(new BiasNeuron());
            if (prevLayer != null) {
                ConnectionFactory.fullConnect(prevLayer, layer);
            }

            prevLayer = layer;
        }
        NeuralNetworkFactory.setDefaultIO(this);
        this.setLearningRule(new BackPropagation());
        this.randomizeWeights(new NguyenWidrowRandomizer(-0.7, 0.7));
    }
}
